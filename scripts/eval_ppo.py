"""Measure how CONSISTENT a trained policy is, on the faithful engine, under realistic conditions.

    python scripts/eval_ppo.py runs/ppo_b/best --episodes 100 --cap 120000

Reports the whole score distribution (not just the best run): median, mean, 10th/25th percentile,
how many episodes reached the cap (survived), and what killed the rest. The same seeds can be run
with the hand-written v9 rule for a like-for-like comparison (--v9).
"""

import argparse
from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from trex.agents.v9_on_engine import V9Agent
from trex.engine import RealEngine
from trex.real_env import RealTRexEnv


def run_policy(model, n, cap, frame_ms, jitter, latency, seed0, skip_clear):
    envs = [RealTRexEnv(max_frames=cap, randomize=False, frame_ms=frame_ms, jitter_ms=jitter, latency=latency,
                        skip_clear_time=skip_clear) for _ in range(n)]
    obs = np.stack([envs[i].reset(seed=seed0 + i)[0] for i in range(n)])
    scores, killers = [None] * n, [None] * n
    alive = np.ones(n, dtype=bool)
    while alive.any():
        actions, _ = model.predict(obs, deterministic=True)
        for i in np.flatnonzero(alive):
            o, r, te, tu, info = envs[i].step(int(actions[i]))
            obs[i] = o
            if te or tu:
                alive[i] = False
                scores[i] = info["score"]
                if te:
                    ob = envs[i].engine.obstacles[0]
                    killers[i] = ("pterodactyl" if ob.kind == 2 else "cactus") + f" x{ob.size}"
                else:
                    killers[i] = "survived to cap"
    return scores, killers


def run_v9(n, cap, frame_ms, jitter, seed0):
    scores, killers = [], []
    for i in range(n):
        e = RealEngine(seed=seed0 + i, frame_ms=frame_ms, jitter_ms=jitter)
        a = V9Agent(); e.frame()
        # same number of display frames as the policy gets decisions x frames per decision
        frames = cap * max(1, round(1000 / 60 / frame_ms))
        out = "survived to cap"
        for _ in range(frames):
            a.act(e, e.last_dt)
            if not e.frame():
                ob = e.obstacles[0]
                out = ("pterodactyl" if ob.kind == 2 else "cactus") + f" x{ob.size}"
                break
        scores.append(e.score); killers.append(out)
    return scores, killers


def report(name, scores, killers):
    s = np.array(scores)
    cap_hits = sum(k == "survived to cap" for k in killers)
    print(f"{name:<14} n={len(s):>3} | median {np.median(s):>7.0f}  mean {s.mean():>7.0f} | p10 {np.percentile(s, 10):>6.0f}  p25 {np.percentile(s, 25):>6.0f}  "
          f"p75 {np.percentile(s, 75):>7.0f} | worst {s.min():>5}  best {s.max():>6} | reached cap {cap_hits}/{len(s)} | "
          f">=5k {int((s >= 5000).sum())}  >=10k {int((s >= 10000).sum())}")
    print(f"{'':<14} deaths: {dict(Counter(k for k in killers if k != 'survived to cap'))}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--cap", type=int, default=60_000, help="decisions per episode (60 per second of play)")
    p.add_argument("--frame-ms", type=float, default=1000 / 120)
    p.add_argument("--jitter", type=float, default=0.3)
    p.add_argument("--latency", type=int, default=0)
    p.add_argument("--seed0", type=int, default=50_000)
    p.add_argument("--v9", action="store_true", help="also run the hand-written v9 rule on the same seeds")
    args = p.parse_args()

    model = PPO.load(args.model, device="cpu")
    print(f"engine: {1000/args.frame_ms:.0f} Hz, jitter {args.jitter} ms, latency {args.latency}, cap {args.cap} decisions "
          f"(~{args.cap/60:.0f} s of play), fixed seeds {args.seed0}..{args.seed0+args.episodes-1}\n")
    sc, k = run_policy(model, args.episodes, args.cap, args.frame_ms, args.jitter, args.latency, args.seed0, False)
    report("PPO", sc, k)
    if args.v9:
        sc9, k9 = run_v9(args.episodes, args.cap, args.frame_ms, args.jitter, args.seed0)
        report("v9 rule", sc9, k9)


if __name__ == "__main__":
    main()
