"""Training Studio: start, change, evaluate and watch training in the browser.

    python scripts/studio.py      ->   http://127.0.0.1:8800
"""

import argparse
import threading
import webbrowser

from trex.studio.server import serve

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    if not a.no_open:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{a.port}")).start()
    serve(a.port)
