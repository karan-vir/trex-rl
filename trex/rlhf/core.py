"""Building blocks of the RLHF pipeline, on the dino.

    segment      a short clip of play (T decisions ~ 2.5 s) from a given course: spec = (seed, scenario, speed)
    pair         two clips on the SAME course, played differently; a judge says which is better
    judge        a human clicking in the Studio, or the "hidden judge" (a script that reads the game's true state)
    preference   (pair, verdict) saved to runs/rlhf/<project>/prefs.jsonl; the reward model is trained on these

LLM analogy: a course is a prompt, a clip is a response, a pair is two sampled responses for that prompt.
The game's own score plays the role of the unknowable "true quality" that humans only approximate.
"""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

import numpy as np

from trex.real_env import RealTRexEnv

T_SEG = 150                                   # decisions per clip (150 / 60 = 2.5 s of play)
PROJECT_DIR = Path("runs/rlhf/default")


# ---- courses and clips -------------------------------------------------------------------------------
def new_spec(rng) -> dict:
    return {"seed": int(rng.integers(10**9)), "scenario": bool(rng.random() < 0.6), "speed": round(float(rng.uniform(6.5, 11.0)), 2)}


def make_env(spec: dict) -> RealTRexEnv:
    return RealTRexEnv(max_frames=10**9, randomize=False, frame_ms=1000 / 120, jitter_ms=0.3, latency=0,
                       start_speed=spec["speed"], skip_clear_time=True, scenario_prob=1.0 if spec["scenario"] else 0.0)


def rollout(model, spec: dict, eps: float, rng, T: int = T_SEG) -> list[int]:
    """Play one clip with the model sampling its actions (not argmax) plus random 'bursts' for variety."""
    env = make_env(spec)
    obs, _ = env.reset(seed=spec["seed"])
    acts, burst, forced = [], 0, 0
    for _ in range(T):
        if burst > 0:
            a, burst = forced, burst - 1
        elif rng.random() < eps:
            forced, burst = int(rng.integers(3)), int(rng.integers(3, 9))
            a = forced
        else:
            a = int(model.predict(obs[None], deterministic=False)[0][0])
        acts.append(a)
        obs, _, te, tu, _ = env.step(a)
        if te:
            break
    return acts


class _Shim:                                   # what draw_game needs from a session
    def __init__(self, engine):
        self.engine, self.done, self.crash_text = engine, False, ""


def replay(spec: dict, actions: list[int], render: bool = False, every: int = 3, rm=None):
    """Re-play a clip deterministically. Returns what happened, the observations, and optionally frames."""
    env = make_env(spec)
    obs, _ = env.reset(seed=spec["seed"])
    X, crashed = [], False
    frames = []
    if render:
        import pygame
        from trex.lab import draw
        pygame.font.init()
        f, surf = draw.fonts(), pygame.Surface((600, 150))
        shim = _Shim(env.engine)
    for i, a in enumerate(actions):
        X.append(np.concatenate([obs, np.eye(3, dtype=np.float32)[a]]))
        obs, _, te, tu, _ = env.step(int(a))
        crashed = te
        if render and (i % every == 0 or te or i == len(actions) - 1):
            shim.done = te
            shim.crash_text = "CRASH" if te else ""
            draw.draw_game(surf, shim, 0, 0, 1, False, "", f)
            buf = io.BytesIO()
            pygame.image.save(surf, buf, "x.png")
            frames.append(base64.b64encode(buf.getvalue()).decode())
    X = np.array(X, dtype=np.float32)
    out = dict(X=X, crashed=bool(crashed), steps=len(actions), passes=len(env.passed), score=env.engine.score, frames=frames)
    if rm is not None:
        from trex.rlhf.reward_model import step_rewards
        out["r"] = step_rewards(rm, X)
    return out


# ---- the hidden judge ---------------------------------------------------------------------------------
def judge(a: dict, b: dict) -> str:
    """Prefer the clip that survived; if both crashed, the one that lasted clearly longer; if both survived, more obstacles cleared."""
    if a["crashed"] != b["crashed"]:
        return "B" if a["crashed"] else "A"
    if a["crashed"]:
        if abs(a["steps"] - b["steps"]) < 10:
            return "same"
        return "A" if a["steps"] > b["steps"] else "B"
    if a["passes"] != b["passes"]:
        return "A" if a["passes"] > b["passes"] else "B"
    return "same"


def make_pair(model, rng, informative: float = 0.7) -> dict:
    """Sample 4 clips on one course, then show two. Mostly the two that differ most (those teach the most)."""
    spec = new_spec(rng)
    cands = []
    for eps in rng.permutation([0.0, 0.02, 0.05, 0.12]):
        acts = rollout(model, spec, float(eps), rng)
        cands.append((acts, replay(spec, acts)))
    order = sorted(range(4), key=lambda i: (cands[i][1]["crashed"], -cands[i][1]["steps"], -cands[i][1]["passes"]))
    if rng.random() < informative and judge(cands[order[0]][1], cands[order[-1]][1]) != "same":
        i, j = order[0], order[-1]
    else:
        i, j = [int(x) for x in rng.choice(4, 2, replace=False)]
    if rng.random() < 0.5:
        i, j = j, i
    (aa, ra), (ab, rb) = cands[i], cands[j]
    return {"spec": spec, "A": [int(x) for x in aa], "B": [int(x) for x in ab], "truth": judge(ra, rb),
            "detail": {"A": {k: ra[k] for k in ("crashed", "steps", "passes")}, "B": {k: rb[k] for k in ("crashed", "steps", "passes")}}}


# ---- the preference dataset ---------------------------------------------------------------------------
def prefs_path(project: Path = PROJECT_DIR) -> Path:
    return project / "prefs.jsonl"


def add_pref(pair: dict, verdict: str, source: str, project: Path = PROJECT_DIR) -> None:
    project.mkdir(parents=True, exist_ok=True)
    rec = {"spec": pair["spec"], "A": pair["A"], "B": pair["B"], "label": verdict, "source": source, "truth": pair.get("truth"), "t": time.time()}
    with open(prefs_path(project), "a") as f:
        f.write(json.dumps(rec) + "\n")


def load_prefs(project: Path = PROJECT_DIR) -> list[dict]:
    p = prefs_path(project)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def prefs_summary(project: Path = PROJECT_DIR) -> dict:
    recs = load_prefs(project)
    human = [r for r in recs if r["source"] == "human"]
    agree = [r["label"] == r["truth"] for r in human if r.get("truth")]
    return {"total": len(recs), "human": len(human), "oracle": len(recs) - len(human),
            "a": sum(r["label"] == "A" for r in recs), "b": sum(r["label"] == "B" for r in recs), "same": sum(r["label"] == "same" for r in recs),
            "human_agreement": (sum(agree) / len(agree)) if agree else None}
