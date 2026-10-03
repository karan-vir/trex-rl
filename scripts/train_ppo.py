"""Train a PPO agent on the faithful engine (trex/real_env.py).

    python scripts/train_ppo.py --frames 20000000 --name ppo_a

Every --eval-every frames it plays fixed-seed episodes under REALISTIC conditions (120 Hz, normal
start, no randomisation) with deterministic actions, logs the score distribution, and keeps the best
checkpoint (by median score, ties broken by mean). Consistency is the goal, so the median and the low
percentile matter more than the best single run.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from trex.real_env import RealTRexEnv


def make_env(max_frames, seed, scenario_prob=0.0):
    def _f():
        env = RealTRexEnv(max_frames=max_frames, randomize=True, scenario_prob=scenario_prob)
        env.reset(seed=seed)
        return env
    return _f


def evaluate(model, n_episodes=24, max_frames=20_000, seed0=10_000):
    """Fixed seeds, realistic timing, deterministic actions. Returns the list of final scores."""
    venv = DummyVecEnv([lambda i=i: RealTRexEnv(max_frames=max_frames, randomize=False) for i in range(n_episodes)])
    obs = np.stack([venv.envs[i].reset(seed=seed0 + i)[0] for i in range(n_episodes)])
    scores = [None] * n_episodes
    alive = np.ones(n_episodes, dtype=bool)
    # step the sub-envs by hand so each episode ends once (no auto-reset)
    while alive.any():
        actions, _ = model.predict(obs, deterministic=True)
        for i in range(n_episodes):
            if not alive[i]:
                continue
            o, r, te, tu, info = venv.envs[i].step(int(actions[i]))
            obs[i] = o
            if te or tu:
                alive[i] = False
                scores[i] = info["score"]
    return scores


class EvalCallback(BaseCallback):
    def __init__(self, out: Path, eval_every: int, n_episodes: int, max_frames: int):
        super().__init__()
        self.out, self.eval_every, self.n_episodes, self.max_frames = out, eval_every, n_episodes, max_frames
        self.next_eval = eval_every
        self.best = (-1, -1)
        self.t0 = time.time()
        with open(out / "eval.csv", "w", newline="") as f:
            csv.writer(f).writerow(["frames", "minutes", "median", "mean", "p10", "p90", "max", "frac_cap"])

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.eval_every
            sc = np.array(evaluate(self.model, self.n_episodes, self.max_frames))
            med, mean = float(np.median(sc)), float(sc.mean())
            row = [self.num_timesteps, round((time.time() - self.t0) / 60, 1), med, round(mean), float(np.percentile(sc, 10)),
                   float(np.percentile(sc, 90)), int(sc.max()), float((sc >= sc.max()).mean()) if False else 0]
            with open(self.out / "eval.csv", "a", newline="") as f:
                csv.writer(f).writerow(row)
            print(f"[eval] {self.num_timesteps/1e6:6.1f}M frames  {row[1]:5.1f} min  median {med:7.0f}  mean {mean:7.0f}  "
                  f"p10 {row[4]:6.0f}  p90 {row[5]:7.0f}  max {int(sc.max()):6d}", flush=True)
            self.model.save(self.out / "latest")
            if (mean, med) > self.best:
                self.best = (mean, med)
                self.model.save(self.out / "best")
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=20_000_000)
    p.add_argument("--name", default="ppo_a")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--max-frames", type=int, default=20_000, help="episode length cap during training")
    p.add_argument("--eval-every", type=int, default=1_000_000)
    p.add_argument("--eval-episodes", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenario-prob", type=float, default=0.0, help="share of training episodes that start with close random obstacles")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--resume", default=None, help="path of a saved model to continue from")
    args = p.parse_args()

    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    venv = SubprocVecEnv([make_env(args.max_frames, args.seed * 1000 + i, args.scenario_prob) for i in range(args.n_envs)])
    if args.resume:
        model = PPO.load(args.resume, env=venv, device="cpu")
    else:
        model = PPO(
            "MlpPolicy", venv, device="cpu", seed=args.seed, verbose=0,
            n_steps=512, batch_size=1024, n_epochs=6, gamma=0.995, gae_lambda=0.95,
            learning_rate=lambda f: args.lr * max(f, 0.1), clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), activation_fn=__import__("torch").nn.Tanh),
        )
    cb = EvalCallback(out, args.eval_every, args.eval_episodes, args.max_frames)
    t0 = time.time()
    model.learn(total_timesteps=args.frames, callback=cb, progress_bar=False)
    model.save(out / "final")
    print(f"done: {args.frames/1e6:.0f}M frames in {(time.time()-t0)/60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
