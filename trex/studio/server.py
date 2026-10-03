"""The Studio backend: a small local web server. The page (page.html) does all the interaction.

It starts/stops training jobs (scripts/train_ppo.py), runs evaluations (scripts/eval_json.py), lets a job's
settings be changed while it runs (control.json), and plays a game server-side so the browser can watch.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import numpy as np                                             # noqa: E402
import pygame                                                  # noqa: E402

from trex.lab import draw                                      # noqa: E402
from trex.lab.session import OBS_NAMES, Config, Session, find_checkpoints   # noqa: E402
from trex.lab.trainer import KNOBS, NEW_ONLY, PRESETS          # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
EVALS = RUNS / "_evals"
LIVE = {"lr", "ent", "clip", "scen", "alive", "passb", "death"}
INT_FLAGS = {"--frames", "--max-frames", "--eval-every", "--n-envs", "--n-steps", "--batch-size", "--epochs",
             "--net-width", "--net-layers", "--seed"}
JOBS: dict[str, subprocess.Popen] = {}


def knobs():
    out = []
    for key, label, vals, default, flag, fmt, hint in KNOBS:
        out.append(dict(key=key, label=label, flag=flag, default=default, help=hint, live=key in LIVE, new_only=key in NEW_ONLY,
                        options=[dict(v=v, text=fmt(v)) for v in vals]))
    return dict(knobs=out, presets=PRESETS)


def read_csv(path: Path):
    rows = []
    if path.exists():
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    rows.append({k: float(v) for k, v in r.items() if k != "frac_cap"})
                except ValueError:
                    pass
    return rows


def ps_jobs():
    """{run name: pid} for live train_ppo.py processes (also ones started outside the Studio)."""
    out = {}
    try:
        txt = subprocess.run(["ps", "-axo", "pid,command"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return out
    for line in txt.splitlines():
        if "train_ppo.py" in line and "--name" in line and "multiprocessing" not in line:
            m = re.search(r"--name\s+(\S+)", line)
            if m:
                out[m.group(1)] = int(line.split()[0])
    return out


def state():
    live = ps_jobs()
    runs = []
    for d in sorted(RUNS.glob("*/eval.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
        name = d.parent.name
        pj = d.parent / "params.json"
        params = json.loads(pj.read_text()) if pj.exists() else None
        log = RUNS / f"{name}.log"
        tail = log.read_text().splitlines()[-6:] if log.exists() else []
        runs.append(dict(name=name, rows=read_csv(d), changes=read_csv_text(d.parent / "changes.csv"), params=params,
                         running=name in live, age=round(time.time() - d.stat().st_mtime), tail=tail))
    ev = []
    for f in sorted(EVALS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:40]:
        try:
            ev.append(dict(id=f.stem, **json.loads(f.read_text())))
        except ValueError:
            pass
    ck = [dict(c, label=f"{c['run']}/{c['name']}") for c in find_checkpoints(RUNS)]
    return dict(runs=runs, evals=ev, ckpts=ck, now=time.strftime("%H:%M:%S"))


def read_csv_text(path: Path):
    out = []
    if path.exists():
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    out.append(dict(frames=float(r["frames"]), what=r["what"]))
                except (ValueError, KeyError):
                    pass
    return out


# ---- actions ----------------------------------------------------------------------------------------
def start_training(body):
    name = re.sub(r"[^A-Za-z0-9_-]", "", body.get("name") or "") or "run_" + time.strftime("%H%M%S")
    if (RUNS / name / "eval.csv").exists() and name in ps_jobs():
        return dict(error=f"a job named {name} is already running")
    cmd = [sys.executable, "scripts/train_ppo.py", "--name", name]
    vals, resume = body.get("values", {}), body.get("resume")
    for key, _l, _v, default, flag, _f, _h in KNOBS:
        if key in NEW_ONLY and resume:
            continue
        v = float(vals.get(key, default))
        cmd += [flag, str(int(v)) if flag in INT_FLAGS else repr(v)]
    if resume:
        cmd += ["--resume", str(resume)]
    RUNS.mkdir(exist_ok=True)
    JOBS[name] = subprocess.Popen(cmd, cwd=ROOT, stdout=open(RUNS / f"{name}.log", "w"), stderr=subprocess.STDOUT)
    return dict(ok=True, name=name)


def stop_training(body):
    pid = ps_jobs().get(body.get("name"))
    if pid:
        os.kill(pid, 15)
    return dict(ok=bool(pid))


def control(body):
    d = RUNS / body["name"]
    ctl = d / "control.json"
    cur = json.loads(ctl.read_text()) if ctl.exists() else {"seq": 0, "set": {}}
    cur["seq"] = cur.get("seq", 0) + 1
    cur["set"] = {k: float(v) for k, v in body.get("set", {}).items() if k in LIVE}
    cur["save_now"] = bool(body.get("save_now"))
    ctl.write_text(json.dumps(cur))
    return dict(ok=True)


def start_eval(body):
    EVALS.mkdir(parents=True, exist_ok=True)
    eid = time.strftime("%H%M%S") + "_" + re.sub(r"[^A-Za-z0-9]", "", str(body["model"]))[-14:]
    cmd = [sys.executable, "scripts/eval_json.py", "--model", str(body["model"]), "--out", str(EVALS / f"{eid}.json"),
           "--episodes", str(int(body.get("episodes", 50))), "--cap", str(int(body.get("cap", 20000))),
           "--hz", str(float(body.get("hz", 120))), "--jitter", str(float(body.get("jitter", 0.3))),
           "--latency", str(int(body.get("latency", 0))), "--scenario", str(int(body.get("scenario", 0)))]
    subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dict(ok=True, id=eid)


# ---- watching a game --------------------------------------------------------------------------------
class Watch:
    surf = None

    def __init__(self):
        self.sess = None
        self.models = {}
        self.f = None

    def new(self, b):
        cfg = Config(hz=float(b.get("hz", 120)), jitter=float(b.get("jitter", 0.3)), latency=int(b.get("latency", 0)),
                     scenario=bool(b.get("scenario")), seed=int(b.get("seed", 0)))
        kind, model, name = b.get("agent", "rule"), None, ""
        if kind == "ppo":
            path = b["model"]
            if path not in self.models:
                from stable_baselines3 import PPO
                self.models[path] = PPO.load(path, device="cpu")
            model, name = self.models[path], path
        self.sess = Session(kind, cfg, model, name)
        self.boxes = bool(b.get("boxes"))
        if self.surf is None:
            pygame.font.init()
            self.f = draw.fonts()
            self.surf = pygame.Surface((1200, 300))
        return self.snapshot()

    def step(self, n, act, boxes):
        s = self.sess
        if s is None:
            return dict(error="no game")
        self.boxes = boxes
        for _ in range(max(1, min(n, 600))):
            s.step(act)
            if s.done:
                break
        return self.snapshot()

    def snapshot(self):
        s = self.sess
        draw.draw_game(self.surf, s, 0, 0, 2, self.boxes, "", self.f)
        buf = io.BytesIO()
        pygame.image.save(self.surf, buf, "x.png")
        e = s.engine
        r = lambda a: None if a is None else [round(float(x), 2) for x in a]
        return dict(png=base64.b64encode(buf.getvalue()).decode(), score=e.score, speed=round(e.speed, 1), done=s.done,
                    crash=s.crash_text, probs=r(s.probs), h1=r(s.h1), h2=r(s.h2), obs=r(s.obs), obs_names=OBS_NAMES,
                    held=s.held, kind=s.kind, decisions=s.decisions, events=list(s.events)[-9:],
                    flags=dict(jumping=bool(e.jumping), ducking=bool(e.ducking), fast_fall=bool(e.speed_drop), status=e.status),
                    rule_action=getattr(s, "rule_action", None))


WATCH = Watch()


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/state":
                self._send(state())
            elif u.path == "/api/knobs":
                self._send(knobs())
            elif u.path == "/api/watch/step":
                self._send(WATCH.step(int(q.get("n", 2)), int(q.get("act", 0)), q.get("boxes") == "1"))
            else:
                self._send((Path(__file__).parent / "page.html").read_bytes(), "text/html; charset=utf-8")
        except Exception as ex:                                  # surface the problem in the page instead of hanging
            self._send(dict(error=f"{type(ex).__name__}: {ex}"))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        try:
            fn = {"/api/train": start_training, "/api/stop": stop_training, "/api/control": control,
                  "/api/eval": start_eval, "/api/watch/new": WATCH.new}[self.path]
            self._send(fn(body))
        except Exception as ex:
            self._send(dict(error=f"{type(ex).__name__}: {ex}"))

    def log_message(self, *a):
        pass


def serve(port=8800):
    os.chdir(ROOT)
    ThreadingHTTPServer.allow_reuse_address = True
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Training Studio: http://127.0.0.1:{port}", flush=True)
    srv.serve_forever()
