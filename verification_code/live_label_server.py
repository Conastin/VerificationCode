"""Live human-in-the-loop labeling server for portal.example.com.

Based on the replay probe finding: a fetched captcha stays bound to its
session (JSESSIONID + lt/execution/csrf) for >=10 minutes, so low-confidence
captures can be HELD un-submitted, shown to a human, and the human's label
submitted for server verification - every accepted label is human-written
AND server-confirmed golden data.

Architecture:
- background worker: fresh capture -> model predict
    conf >= --auto-threshold  -> auto-submit (server verdict), archive
    conf <  threshold         -> park in pending queue (session kept alive)
- web UI (http://localhost:8765): shows pending images; human types the code;
  submit -> server verdict -> golden (human + server confirmed) or reject
- pending items older than --ttl seconds are dropped (binding expires)

Usage:
    python -m verification_code.live_label_server --model v9
Then open http://localhost:8765 in a browser.
"""

import os
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image

from .portal_login_validate import (classify, fetch_captcha, fetch_login_page,
                                    make_opener, recognize, submit_login)
from .train_universal import BLANK, FOLD_CHARSET, CrnnCaptcha, ctc_greedy, preprocess, set_img_h


def _logaddexp(a: float, b: float) -> float:
    m = a if a > b else b
    return m + math.log1p(math.exp(-abs(a - b)))


def ctc_topk(logits: torch.Tensor, k: int = 3, beam_width: int = 20) -> list[str]:
    """Standard CTC prefix beam search; returns top-k decoded strings.

    Used for the typo-guard: the human's input is compared against these
    candidates before the single submission attempt is burned.
    """
    logp = torch.log_softmax(logits[0].float(), dim=-1).cpu()
    T, C = logp.shape
    beams: dict[tuple, list[float]] = {(): [0.0, -1e18]}  # [p_blank, p_no_blank]

    def merge(dst, key, val, idx):
        entry = dst.setdefault(key, [-1e18, -1e18])
        entry[idx] = _logaddexp(entry[idx], val)

    for t in range(T):
        nxt: dict[tuple, list[float]] = {}
        row = logp[t]
        for prefix, (pb, pnb) in beams.items():
            last = prefix[-1] if prefix else None
            total = _logaddexp(pb, pnb)
            merge(nxt, prefix, total + row[BLANK].item(), 0)
            for c in range(C):
                if c == BLANK:
                    continue
                if c == last:
                    merge(nxt, prefix, pnb + row[c].item(), 1)
                else:
                    merge(nxt, prefix + (FOLD_CHARSET[c],),
                          pb + row[c].item(), 1)
        beams = dict(sorted(nxt.items(),
                            key=lambda kv: -_logaddexp(kv[1][0], kv[1][1])
                            )[:beam_width])
    ranked = ["".join(prefix) for prefix, _ in
              sorted(beams.items(), key=lambda kv: -_logaddexp(kv[1][0], kv[1][1]))]
    non_empty = [r for r in ranked if r][:k]
    while len(non_empty) < k:  # pathological all-blank case
        non_empty.append("")
    return non_empty[:k]

ROOT = Path("data/portal_dataset")
AUTO_TH = 0.97
TTL_S = 480  # conservative under the measured 10-minute binding
PENDING: deque = deque()          # held sessions awaiting human labels
PENDING_LOCK = threading.Lock()
STATS = {"auto_ok": 0, "auto_wrong": 0, "human_ok": 0, "human_wrong": 0,
         "expired": 0}
STATS_LOCK = threading.Lock()
BATCH = datetime.now().strftime("live_%Y%m%d_%H%M%S")


def archive(kind: str, name: str, data: bytes, label: str) -> None:
    out = ROOT / kind / BATCH
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_bytes(data)
    if kind == "server_confirmed":
        with (out / "labels.txt").open("a", encoding="utf-8") as handle:
            handle.write(f"{name} {label}\n")
    else:
        # rejected submissions must NOT land in labels.txt (training loader
        # convention) - record under an explicitly-unsafe filename instead
        with (out / "failed_submissions_DO_NOT_TRAIN.txt").open(
                "a", encoding="utf-8") as handle:
            handle.write(f"{name} {label}\n")


class Pending:
    def __init__(self, opener, page, data, pred, conf, top3) -> None:
        self.opener = opener
        self.page = page
        self.data = data
        self.pred = pred
        self.conf = conf
        self.top3 = top3
        self.created = time.time()
        self.id = f"{int(self.created * 1000) % 10**9}"
        self.done = False


def recognize_full(model, data: bytes, device):
    """Greedy prediction + beam top-3 for the typo-guard."""
    with Image.open(io.BytesIO(data)) as image:
        arr = preprocess(image.convert("RGB"), model.img_h)
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        (pred, conf), = ctc_greedy(logits)
        top3 = ctc_topk(logits)
    return pred, conf, top3


def worker(model, device, stop_evt: threading.Event, interval: float) -> None:
    while not stop_evt.is_set():
        try:
            opener = make_opener()
            page = fetch_login_page(opener)
            if not page["lt"]:
                raise RuntimeError("no lt")
            data = fetch_captcha(opener, page["lt"])
            pred, conf, top3 = recognize_full(model, data, device)
            if conf >= AUTO_TH:
                _, resp = submit_login(opener, page, os.environ.get("PORTAL_USER", "testuser01"),
                                       os.environ.get("PORTAL_PASSWORD", ""), pred)
                verdict, _ = classify(resp)
                with STATS_LOCK:
                    if verdict == "captcha_ok":
                        STATS["auto_ok"] += 1
                        archive("server_confirmed", f"{pred}_{STATS['auto_ok']:05d}.jpg",
                                data, pred)
                    else:
                        STATS["auto_wrong"] += 1
                        archive("server_failed", f"fail_{STATS['auto_wrong']:05d}.jpg",
                                data, pred)
            else:
                # hold: do NOT submit; wait for a human within TTL
                with PENDING_LOCK:
                    # drop stale items first
                    while PENDING and time.time() - PENDING[0].created > TTL_S:
                        STATS["expired"] += 1
                        PENDING.popleft()
                    PENDING.append(Pending(opener, page, data, pred, conf, top3))
        except Exception as error:  # noqa: BLE001
            print(f"worker: {error}", flush=True)
        stop_evt.wait(interval)


PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>portal 实时标注</title>
<style>
 body{font-family:Consolas,"Microsoft YaHei";background:#222;color:#eee;display:flex;
      flex-direction:column;align-items:center;padding:20px}
 #stat{color:#9ecbff;margin-bottom:10px}
 img{border:2px solid #555;background:#fff}
 #row{display:flex;gap:14px;align-items:center;margin-top:14px}
 input{font-size:28px;width:190px;text-align:center;letter-spacing:8px;padding:6px}
 button{font-size:16px;padding:8px 20px;cursor:pointer}
 #msg{margin-top:10px;min-height:26px;font-size:18px}
 .ok{color:#6f6}.bad{color:#f66}.skip{color:#aaa}
</style></head><body>
<div id="stat">等待中: <span id="qn">0</span> | 自动通过 <span id="aok">0</span> |
自动失败 <span id="abad">0</span> | 人工确认 <span id="hok">0</span> |
人工被拒 <span id="hbad">0</span></div>
<img id="img" width="440" height="187">
<div id="meta" style="color:#888;margin-top:6px"></div>
<div id="row"><input id="code" maxlength="6" autofocus autocomplete="off"
  spellcheck="false" placeholder="输入后回车">
<button id="skip">跳过</button></div>
<div id="msg"></div>
<script>
let cur = null, armed = false;
async function poll(){ for(;;){ await new Promise(r=>setTimeout(r,1500));
  try{ const j = await (await fetch('/api/pending')).json();
    document.getElementById('qn').textContent=j.queue;
    document.getElementById('aok').textContent=j.stats.auto_ok;
    document.getElementById('abad').textContent=j.stats.auto_wrong;
    document.getElementById('hok').textContent=j.stats.human_ok;
    document.getElementById('hbad').textContent=j.stats.human_wrong;
    if(j.item && (!cur || j.item.id!==cur.id)){ show(j.item); } }catch(e){} } }
function show(it){ cur=it; armed=false;
  document.getElementById('img').src='data:image/jpeg;base64,'+it.img;
  document.getElementById('meta').textContent=
    `模型候选: ${it.top3.join(' / ')}  (conf ${it.conf})  #${it.id}`;
  document.getElementById('msg').textContent='';
  const c=document.getElementById('code'); c.value=''; c.focus(); }
async function submit(){ if(!cur) return;
  const code=document.getElementById('code').value.trim().toUpperCase();
  const m=document.getElementById('msg');
  if(code.length!==4){ armed=false; m.className='bad';
    m.textContent='✗ 长度应为 4 个字符，请检查'; return; }
  if(!cur.top3.includes(code) && !armed){ armed=true; m.className='skip';
    m.textContent='⚠ 与模型候选差异大——确认无误请再按一次回车提交，或直接修改';
    return; }
  armed=false;
  const r=await (await fetch('/api/submit', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id:cur.id, code})})).json();
  if(r.verdict==='captcha_ok'){m.className='ok';m.textContent='✓ 服务器确认，已入库';}
  else if(r.verdict==='captcha_wrong'){m.className='bad';m.textContent='✗ 服务器拒绝（标注错误或已超时）';}
  else {m.className='skip';m.textContent='? '+r.verdict;}
  cur=null; setTimeout(async()=>{ const j=await (await fetch('/api/pending')).json();
    if(j.item) show(j.item); }, 400); }
document.getElementById('code').addEventListener('keydown',e=>{
  if(e.key==='Enter'){e.preventDefault();submit();}
  if(e.key==='Escape'){armed=false;}});
document.getElementById('code').addEventListener('input',()=>{armed=false;});
document.getElementById('skip').onclick=()=>{cur=null;
  document.getElementById('msg').className='skip';
  document.getElementById('msg').textContent='已跳过（等待服务器自然过期）';};
poll();
</script></body></html>"""


def make_handler(model, device):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _json(self, obj, status=200):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/pending":
                with PENDING_LOCK:
                    while PENDING and time.time() - PENDING[0].created > TTL_S:
                        STATS["expired"] += 1
                        PENDING.popleft()
                    item = None
                    for p in PENDING:
                        item = p
                        break
                    payload = {
                        "queue": len(PENDING),
                        "stats": dict(STATS),
                        "item": None if item is None else {
                            "id": item.id,
                            "img": base64.b64encode(item.data).decode(),
                            "pred": item.pred,
                            "conf": round(item.conf, 3),
                            "top3": item.top3,
                        },
                    }
                self._json(payload)

        def do_POST(self):
            if self.path == "/api/submit":
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length))
                with PENDING_LOCK:
                    target = None
                    for p in PENDING:
                        if p.id == req["id"]:
                            target = p
                            break
                    if target is not None:
                        PENDING.remove(target)
                if target is None:
                    self._json({"verdict": "expired", "hint": "队列中已无此项"})
                    return
                _, resp = submit_login(target.opener, target.page,
                                       os.environ.get("PORTAL_USER", "testuser01"),
                                       os.environ.get("PORTAL_PASSWORD", ""), req["code"])
                verdict, hint = classify(resp)
                with STATS_LOCK:
                    if verdict == "captcha_ok":
                        STATS["human_ok"] += 1
                        archive("server_confirmed",
                                f"H{req['code']}_{STATS['human_ok']:05d}.jpg",
                                target.data, req["code"])
                    else:
                        STATS["human_wrong"] += 1
                        archive("server_failed",
                                f"Hfail_{STATS['human_wrong']:05d}.jpg",
                                target.data, req["code"])
                self._json({"verdict": verdict, "hint": hint})

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live human-in-the-loop labeler.")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/universal_crnn_ft_portal_v7.pt"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=1.2,
                        help="seconds between background captures")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    set_img_h(ckpt.get("img_h", 32))
    model = CrnnCaptcha(img_h=ckpt.get("img_h", 32),
                        lstm_hidden=ckpt.get("lstm_hidden", 192),
                        lstm_layers=ckpt.get("lstm_layers", 2)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"model={args.checkpoint} auto-threshold={AUTO_TH} ttl={TTL_S}s "
          f"batch={BATCH}")

    stop_evt = threading.Event()
    threading.Thread(target=worker, args=(model, device, stop_evt,
                                          args.interval),
                     daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(model, device))
    print(f"open http://localhost:{args.port}  (Ctrl+C 停止)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop_evt.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
