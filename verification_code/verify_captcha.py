"""Interactive manual verification of captured CAPTCHAs.

Shows each captured CAPTCHA enlarged with the recognizer's prediction.
Low-confidence images come first. Keyboard workflow:

- type 4 characters + Enter  -> correct the label, move to next
- Enter (empty input)        -> accept the prediction, move to next
- Left / Right arrows        -> go to previous / next image
- q + Enter                  -> save and quit

Every accepted/corrected label is written to labels_corrected.txt
immediately, so quitting early loses nothing. Labels for images you never
visited keep the recognizer's prediction.

Usage:
    python -m verification_code.verify_captcha --dir data/real_captcha
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manually verify captured CAPTCHAs.")
    parser.add_argument("--dir", type=Path, default=Path("data/real_captcha"))
    parser.add_argument("--out", type=Path, default=None,
                        help="output labels file (default: <dir>/labels_corrected.txt)")
    parser.add_argument("--max-conf", type=float, default=1.0,
                        help="only show images with confidence below this value; "
                             "labels for skipped images keep the prediction")
    parser.add_argument("--chars", default="",
                        help="also show images whose predicted label contains any of "
                             "these characters (e.g. MN for confusable M/N pairs); "
                             "shown set = (conf < max-conf) OR (label contains char)")
    return parser


class VerifierApp:
    def __init__(self, root: tk.Tk, directory: Path, out_path: Path,
                 max_conf: float, max_chars: str = "") -> None:
        self.root = root
        self.directory = directory
        self.out_path = out_path
        self.max_conf = max_conf
        self.max_chars = max_chars

        # (actual_filename, predicted_label, min_confidence) sorted by confidence asc
        # row: source_file, label, predicted_file, min_confidence (skip header)
        manifest = self.directory / "capture_manifest.csv"
        self.items: list[tuple[str, str, float]] = []
        if manifest.is_file():
            with manifest.open(encoding="utf-8") as handle:
                for row in csv.reader(handle):
                    if len(row) >= 3 and row[1] and row[2] and row[1] != "label":
                        try:
                            conf = float(row[3]) if len(row) > 3 else 1.0
                        except ValueError:
                            conf = 1.0
                        if (self.directory / row[2]).is_file():
                            self.items.append((row[2], row[1], conf))
        if not self.items:
            raise SystemExit(f"no manifest entries in {manifest}")
        self.items.sort(key=lambda item: item[2])
        # Shown set = (confidence below threshold) OR (prediction contains a
        # confusable character); everything else keeps the prediction.
        chars = set(self.max_chars)
        self.visible = [
            i for i, item in enumerate(self.items)
            if item[2] < max_conf or any(c in item[1] for c in chars)
        ]
        if not self.visible:
            print(f"no images to review (conf<{max_conf} or contains {self.max_chars!r})")
            raise SystemExit(0)

        # labels already saved in a previous run (keeps progress across restarts)
        self.labels: dict[str, str] = {}
        if out_path.is_file():
            with out_path.open(encoding="utf-8") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) == 2:
                        self.labels[parts[0]] = parts[1]

        self.index = 0
        self._photo: ImageTk.PhotoImage | None = None
        self._build_widgets()
        self.root.title("CAPTCHA 核对")
        self.root.bind("<Return>", self._on_enter)
        self.root.bind("<Left>", lambda _event: self._go(-1))
        self.root.bind("<Right>", lambda _event: self._go(1))
        # Jump to the first unverified image so a resumed run continues where
        # it left off instead of re-showing everything from the start.
        self.index = next(
            (v for v in self.visible if self.items[v][0] not in self.labels),
            len(self.visible) - 1,
        )
        self._show()

    def _build_widgets(self) -> None:
        self.info_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.info_var, font=("Consolas", 14)).pack(pady=6)
        self.image_label = tk.Label(self.root)
        self.image_label.pack()
        self.entry = tk.Entry(self.root, font=("Consolas", 18), justify="center", width=12)
        self.entry.pack(pady=8)
        tk.Label(
            self.root,
            text="输入 4 个字符+回车 = 修改标签；直接回车 = 接受预测；←/→ 切换；q+回车 = 保存退出",
            font=("Microsoft YaHei", 10),
        ).pack(pady=4)
        self.entry.focus_set()

    def _show(self) -> None:
        filename, predicted, conf = self.items[self.visible[self.index]]
        current = self.labels.get(filename, predicted)
        marked = "✓" if self.labels.get(filename) else "·"
        self.info_var.set(
            f"{marked} {self.index + 1}/{len(self.visible)} (需核对)  {filename}  "
            f"预测: {predicted}  置信度: {conf:.3f}  当前标签: {current}"
        )
        image_path = self.directory / filename
        if not image_path.is_file():
            self.info_var.set(f"{self.index + 1}/{len(self.visible)}  {filename}  [文件缺失，按 → 跳过]")
            self._photo = None
            self.image_label.config(image=self._photo)
            self.entry.delete(0, tk.END)
            return
        image = Image.open(image_path).resize((480, 160), Image.NEAREST)
        self._photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=self._photo)
        self.entry.delete(0, tk.END)
        self.entry.insert(0, current)
        self.entry.select_range(0, tk.END)

    def _on_enter(self, _event=None) -> None:
        text = self.entry.get().strip().upper()
        if text == "Q":
            self._save()
            self.root.destroy()
            return
        filename, _predicted, _conf = self.items[self.visible[self.index]]
        if len(text) == 4:
            self.labels[filename] = text
            self._save()
            if self.index + 1 < len(self.visible):
                self._go(1)
            else:
                messagebox.showinfo("完成", f"全部核对完成，已保存到 {self.out_path}")
                self.root.destroy()
        else:
            messagebox.showwarning("输入无效", "请输入 4 个字符（字母/数字），或按 Q 退出")

    def _go(self, delta: int) -> None:
        self.index = max(0, min(len(self.visible) - 1, self.index + delta))
        self._show()

    def _save(self) -> None:
        rows: list[tuple[str, str]] = []
        for filename, predicted, _conf in self.items:
            rows.append((filename, self.labels.get(filename, predicted)))
        with self.out_path.open("w", encoding="utf-8", newline="\n") as handle:
            for filename, label in rows:
                handle.write(f"{filename} {label}\n")


def main() -> int:
    args = build_parser().parse_args()
    out_path = args.out or (args.dir / "labels_corrected.txt")
    root = tk.Tk()
    VerifierApp(root, args.dir, out_path, args.max_conf, args.chars)
    root.mainloop()
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
