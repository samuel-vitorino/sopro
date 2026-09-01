#!/usr/bin/env python3
"""Static demo server with the headers required for WASM threads.

  python serve.py [--port 8123]
"""
import argparse
import functools
import http.server
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent
RUNTIME_URL = "https://cdn.jsdelivr.net/npm/@soprotts/onnx-web@0.3.0/src/index.js"


class Handler(http.server.SimpleHTTPRequestHandler):
    model_dir = None

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_GET(self):
        clean = self.path.split("?", 1)[0].split("#", 1)[0]
        if self.model_dir is not None and clean in ("/", "/index.html"):
            body = (ROOT / "index.html").read_text(encoding="utf-8").replace(RUNTIME_URL, "/onnx-web/index.js").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def translate_path(self, path):
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean == "/backend.js":
            return str(ROOT / "backend-onnx.js")
        if self.model_dir is not None and clean.startswith("/model/"):
            return str(self._inside(self.model_dir, clean.removeprefix("/model/")))
        if self.model_dir is not None and clean.startswith("/onnx-web/"):
            return str(self._inside(ROOT.parents[1] / "web" / "src", clean.removeprefix("/onnx-web/")))
        return super().translate_path(path)

    @staticmethod
    def _inside(root, relative):
        root = root.resolve()
        candidate = (root / unquote(relative)).resolve()
        return candidate if candidate.is_relative_to(root) else root / ".missing"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cert", default=None)
    parser.add_argument("--key", default=None)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()
    Handler.model_dir = args.model_dir.resolve() if args.model_dir is not None else None
    if Handler.model_dir is not None and not (Handler.model_dir / "manifest.json").is_file():
        parser.error(f"model directory has no manifest.json: {Handler.model_dir}")
    handler = functools.partial(Handler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    scheme = "http"
    if args.cert and args.key:
        import ssl

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.cert, args.key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    suffix = "?model=/model" if Handler.model_dir is not None else ""
    print(f"{scheme}://{args.host}:{args.port}/{suffix}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
