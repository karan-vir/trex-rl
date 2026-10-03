"""Backend for the guided RLHF lesson (learn.html)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from trex.rlhf import core
from trex.rlhf.experiments import BY_ID, CARDS, EXP_DIR, read_rows
from trex.rlhf.reward_model import accuracy_vs_judge, load_rm, save_rm, step_rewards, train_reward_model

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "runs" / "rlhf" / "default"
TASKS: dict[str, dict] = {}
PENDING: dict[str, dict] = {}
_models: dict = {}
_rng = np.random.default_rng()
DEFAULT_REF = "runs/ppo_d/ckpt_007M"


def _model(path):
    if path not in _models:
        from stable_baselines3 import PPO
        _models[path] = PPO.load(str(ROOT / path), device="cpu")
    return _models[path]


def project() -> dict:
    p = PROJ / "project.json"
    return json.loads(p.read_text()) if p.exists() else {"ref": DEFAULT_REF}


def set_ref(body):
    PROJ.mkdir(parents=True, exist_ok=True)
    cur = project(); cur["ref"] = body["ref"]; cur["chosen"] = True
    (PROJ / "project.json").write_text(json.dumps(cur))
    return {"ok": True}


def _ps(pattern: str) -> dict:
    out = {}
    try:
        txt = subprocess.run(["ps", "-axo", "pid,command"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return out
    for line in txt.splitlines():
        if pattern in line and "multiprocessing" not in line:
            out[line.split(None, 1)[1]] = int(line.split()[0])
    return out


def checkpoints_with_scores(ckpts):
    """Attach the training-time evaluation (mean / worst 10%) to every checkpoint that has one."""
    evals = {}
    for r in (ROOT / "runs").glob("*/eval.csv"):
        evals[r.parent.name] = read_rows(r)
    out = []
    for c in ckpts:
        rows = evals.get(c["run"], [])
        m = c["M"]
        hit = next((r for r in rows if m is not None and abs(r["frames"] - m * 1e6) < 3e5), None)
        out.append(dict(c, mean=hit["mean"] if hit else None, p10=hit["p10"] if hit else None,
                        path=c["path"][:-4] if c["path"].endswith(".zip") else c["path"]))
    return out


def rlhf_runs():
    live = _ps("train_rlhf")
    runs = []
    for d in sorted((ROOT / "runs").glob("*/params.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        params = json.loads(d.read_text())
        if params.get("kind") != "rlhf" or d.parent.name.startswith("exp_"):
            continue
        n = d.parent.name
        runs.append(dict(name=n, params=params, eval=read_rows(d.parent / "eval.csv"), proxy=read_rows(d.parent / "rlhf.csv"),
                         tail=(ROOT / "runs" / f"{n}.log").read_text().splitlines()[-6:] if (ROOT / "runs" / f"{n}.log").exists() else [],
                         running=any(f"--name {n}" in c for c in live), has_final=(d.parent / "final.zip").exists(),
                         age=round(time.time() - (d.parent / "eval.csv").stat().st_mtime) if (d.parent / "eval.csv").exists() else 0))
    return runs


def experiments_state():
    live = _ps("rlhf_experiment")
    out = {}
    for c in CARDS:
        p = EXP_DIR / f"{c['id']}.json"
        d = json.loads(p.read_text()) if p.exists() else None
        out[c["id"]] = dict(result=d, running=any(c["id"] in cmd for cmd in live))
    return out


def state(ckpts):
    rm_meta = None
    if (PROJ / "rm.json").exists():
        rm_meta = json.loads((PROJ / "rm.json").read_text())
    return dict(project=project(), prefs=core.prefs_summary(PROJ), rm=rm_meta, tasks=TASKS, runs=rlhf_runs(),
                ckpts=checkpoints_with_scores(ckpts), cards=[{k: v for k, v in c.items()} for c in CARDS], experiments=experiments_state())


# ---- human feedback --------------------------------------------------------------------------------------
def pair_next(body):
    ref = _model(project()["ref"])
    pair = core.make_pair(ref, _rng, informative=0.6)
    pid = f"{int(time.time() * 1000)}"
    PENDING[pid] = pair
    for k in list(PENDING)[:-20]:
        PENDING.pop(k, None)
    fa = core.replay(pair["spec"], pair["A"], render=True)
    fb = core.replay(pair["spec"], pair["B"], render=True)
    return dict(id=pid, A=fa["frames"], B=fb["frames"], speed=pair["spec"]["speed"], danger=pair["spec"]["scenario"])


def pair_label(body):
    pair = PENDING.get(body["id"])
    if not pair:
        return {"error": "that pair expired; get a new one"}
    core.add_pref(pair, body["label"], "human", PROJ)
    truth = pair["truth"]
    return dict(ok=True, truth=truth, agree=(body["label"] == truth), detail=pair["detail"], prefs=core.prefs_summary(PROJ))


def autolabel(body):
    n, noise = int(body.get("n", 200)), float(body.get("noise", 0))
    if TASKS.get("autolabel", {}).get("running"):
        return {"error": "already running"}
    ref = _model(project()["ref"])

    def work():
        TASKS["autolabel"] = dict(running=True, done=0, total=n)
        rng = np.random.default_rng()
        for i in range(n):
            p = core.make_pair(ref, rng, informative=0.6)
            lab = p["truth"]
            if lab != "same" and rng.random() < noise:
                lab = "B" if lab == "A" else "A"
            core.add_pref(p, lab, "oracle", PROJ)
            TASKS["autolabel"]["done"] = i + 1
        TASKS["autolabel"]["running"] = False
    threading.Thread(target=work, daemon=True).start()
    return {"ok": True}


def clear_prefs(body):
    p = core.prefs_path(PROJ)
    if p.exists():
        p.unlink()
    for f in ("rm.pt", "rm.json"):
        (PROJ / f).unlink(missing_ok=True)
    return {"ok": True}


# ---- reward model -----------------------------------------------------------------------------------------
def rm_train(body):
    if TASKS.get("rm", {}).get("running"):
        return {"error": "already training"}
    recs = core.load_prefs(PROJ)
    if len(recs) < 10:
        return {"error": "need at least 10 judgments first"}
    epochs = int(body.get("epochs", 60))
    ref = _model(project()["ref"])

    def work():
        TASKS["rm"] = dict(running=True, epoch=0, total=epochs, hist=[])
        def prog(ep, tot, row):
            TASKS["rm"].update(epoch=ep, hist=TASKS["rm"]["hist"] + [row])
        try:
            rm, hist, info = train_reward_model(recs, epochs=epochs, progress=prog)
            TASKS["rm"]["stage"] = "checking against the hidden judge on new clips"
            agree = accuracy_vs_judge(rm, ref, n=120)
            save_rm(rm, PROJ / "rm.pt", {"n": len(recs)})
            (PROJ / "rm.json").write_text(json.dumps(dict(n_prefs=len(recs), hist=hist, agreement=agree, trained=time.time(), **info)))
        except Exception as ex:
            TASKS["rm"]["error"] = f"{type(ex).__name__}: {ex}"
        TASKS["rm"]["running"] = False
    threading.Thread(target=work, daemon=True).start()
    return {"ok": True}


def rm_probe(body):
    if not (PROJ / "rm.pt").exists():
        return {"error": "train the reward model first"}
    rm, _ = load_rm(PROJ / "rm.pt")
    ref = _model(project()["ref"])
    pick = None
    for _ in range(12):
        spec = core.new_spec(_rng); spec["scenario"] = True
        acts = core.rollout(ref, spec, 0.05, _rng)
        r = core.replay(spec, acts)
        if pick is None or (r["crashed"] and len(acts) > 40):
            pick = (spec, acts)
        if r["crashed"] and len(acts) > 40:
            break
    spec, acts = pick
    r = core.replay(spec, acts, render=True, every=3, rm=rm)
    idx = [i for i in range(len(acts)) if i % 3 == 0 or i == len(acts) - 1]
    return dict(frames=r["frames"], r=[r["r"][i] for i in idx], crashed=r["crashed"], steps=r["steps"], passes=r["passes"])


# ---- RL stage ----------------------------------------------------------------------------------------------
def rl_start(body):
    if not (PROJ / "rm.pt").exists():
        return {"error": "train the reward model first (step 4)"}
    name = re.sub(r"[^A-Za-z0-9_-]", "", body.get("name") or "") or "rlhf_" + time.strftime("%H%M%S")
    cmd = [sys.executable, "-m", "scripts.train_rlhf", "--ref", project()["ref"], "--rm", str(PROJ / "rm.pt"), "--name", name,
           "--frames", str(int(body.get("frames", 1_500_000))), "--beta", str(float(body.get("beta", 0.05))),
           "--scale", str(float(body.get("scale", 0.1))), "--lr", str(float(body.get("lr", 1e-4))),
           "--eval-every", str(int(max(125_000, int(body.get("frames", 1_500_000)) // 12)))]
    (ROOT / "runs").mkdir(exist_ok=True)
    subprocess.Popen(cmd, cwd=ROOT, stdout=open(ROOT / "runs" / f"{name}.log", "w"), stderr=subprocess.STDOUT)
    return {"ok": True, "name": name}


def exp_start(body):
    cid = body["id"]
    if cid not in BY_ID:
        return {"error": "unknown experiment"}
    if any(cid in c for c in _ps("rlhf_experiment")):
        return {"error": "already running"}
    (ROOT / "runs").mkdir(exist_ok=True)
    subprocess.Popen([sys.executable, "-m", "scripts.rlhf_experiment", cid], cwd=ROOT,
                     stdout=open(ROOT / "runs" / f"exp_{cid}.log", "w"), stderr=subprocess.STDOUT)
    return {"ok": True}
