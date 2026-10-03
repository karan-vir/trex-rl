"""Evaluate one model (or the hand-written rule) and write the results as JSON for the Studio.

    python scripts/eval_json.py --model runs/ppo_d/best.zip --out runs/_evals/x.json --episodes 50
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from trex.agents.v9_on_engine import V9Agent
from trex.engine import RealEngine
from trex.real_env import RealTRexEnv


def killer(ob):
    return ("pterodactyl" if ob.kind == 2 else "cactus") + f" x{ob.size}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to a PPO .zip, or 'rule'")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--cap", type=int, default=20_000)
    p.add_argument("--hz", type=float, default=120)
    p.add_argument("--jitter", type=float, default=0.3)
    p.add_argument("--latency", type=int, default=0)
    p.add_argument("--scenario", type=int, default=0)
    p.add_argument("--seed0", type=int, default=50_000)
    a = p.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(vars(a), started=time.time(), done=False, scores=[], killers=[], progress=0.0)
    save = lambda: out.write_text(json.dumps(meta))
    save()
    frame_ms, n = 1000 / a.hz, a.episodes
    scores, kills = [None] * n, [None] * n

    if a.model == "rule":
        skip = max(1, round(1000 / 60 / frame_ms))
        for i in range(n):
            e = RealEngine(seed=a.seed0 + i, frame_ms=frame_ms, jitter_ms=a.jitter); ag = V9Agent(); e.frame()
            kills[i] = "survived to cap"
            for _ in range(a.cap):
                for _ in range(skip):
                    ag.act(e, e.last_dt)
                    if not e.frame():
                        kills[i] = killer(e.obstacles[0]); break
                if kills[i] != "survived to cap":
                    break
            scores[i] = e.score
            meta["progress"] = (i + 1) / n; save()
    else:
        from stable_baselines3 import PPO
        model = PPO.load(a.model, device="cpu")
        envs = [RealTRexEnv(max_frames=a.cap, randomize=False, frame_ms=frame_ms, jitter_ms=a.jitter, latency=a.latency,
                            scenario_prob=1.0 if a.scenario else 0.0) for _ in range(n)]
        obs = np.stack([envs[i].reset(seed=a.seed0 + i)[0] for i in range(n)])
        alive = np.ones(n, dtype=bool); t = 0
        while alive.any():
            actions, _ = model.predict(obs, deterministic=True)
            for i in np.flatnonzero(alive):
                o, r, te, tu, info = envs[i].step(int(actions[i]))
                obs[i] = o
                if te or tu:
                    alive[i] = False; scores[i] = info["score"]
                    kills[i] = killer(envs[i].engine.obstacles[0]) if te else "survived to cap"
            t += 1
            if t % 200 == 0:
                meta["progress"] = round(1 - alive.mean(), 3); save()
    meta.update(done=True, progress=1.0, scores=[int(x) for x in scores], killers=kills, finished=time.time())
    save()


if __name__ == "__main__":
    main()
