"""Tiny static server for the browser demo, with COOP/COEP headers so
onnxruntime-web can use its threaded WASM build.

Usage:
    python server.py [port]
Then open http://localhost:8000/browser/
"""

from __future__ import annotations

import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        # Correct MIME for .onnx (otherwise browsers may refuse to fetch it)
        if self.path.endswith(".onnx"):
            self.send_header("Content-Type", "application/octet-stream")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:  # quieter logs
        print(f"[{self.log_date_time_string()}] {self.address_string()} {format % args}")


def main() -> int:
    root = Path(__file__).resolve().parent.parent  # project root
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    handler = lambda *args, **kwargs: Handler(*args, directory=str(root), **kwargs)  # noqa: E731
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"serving {root} on http://127.0.0.1:{port}/browser/")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
