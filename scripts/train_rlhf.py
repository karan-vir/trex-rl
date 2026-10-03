"""RL fine-tuning against a LEARNED reward model (the "PPO stage" of RLHF), on the dino.

    python scripts/train_rlhf.py --ref runs/ppo_d/ckpt_007M --rm runs/rlhf/default/rm.pt --name rlhf_a --beta 0.05

The game's own reward is thrown away. Each step the agent gets

    reward = scale * (reward_model(obs, action) + shift)  -  beta * (log pi(a|s) - log pi_ref(a|s))

The second term is the KL penalty: a leash that keeps the policy close to the reference model, exactly as
in LLM RLHF. The game's true score is only used to GRADE the agent (eval.csv), never to train it.
"""

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper

from scripts.train_ppo import ControlCallback, EvalCallback, evaluate, make_env
from trex.rlhf.reward_model import load_rm


class RMVecEnv(VecEnvWrapper):
    """Replaces the game's reward with the learned reward minus a KL penalty to the reference policy."""

    def __init__(self, venv, rm, ref_policy, scale, shift, beta):
        super().__init__(venv)
        self.rm, self.ref, self.scale, self.shift, self.beta = rm, ref_policy, scale, shift, beta
        self.model = None
        self.last_obs = None
        self.acts = None
        self.reset_stats()

    def reset_stats(self):
        self.n, self.sum_r, self.sum_kl = 0, 0.0, 0.0

    def reset(self):
        self.last_obs = self.venv.reset()
        return self.last_obs

    def step_async(self, actions):
        self.acts = np.asarray(actions)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, _true_rewards, dones, infos = self.venv.step_wait()
        with torch.no_grad():
            o = torch.as_tensor(self.last_obs, dtype=torch.float32)
            a = torch.as_tensor(self.acts, dtype=torch.long)
            x = torch.cat([o, torch.nn.functional.one_hot(a, 3).float()], dim=1)
            r = self.rm.normalised(x).numpy()
            kl = np.zeros(len(a), dtype=np.float32)
            if self.model is not None:
                _, lp_cur, _ = self.model.policy.evaluate_actions(o, a)
                _, lp_ref, _ = self.ref.evaluate_actions(o, a)
                kl = (lp_cur - lp_ref).numpy()
        rewards = (self.scale * (r + self.shift) - self.beta * kl).astype(np.float32)
        self.n += len(r); self.sum_r += float(r.sum()); self.sum_kl += float(kl.sum())
        self.last_obs = obs
        return obs, rewards, dones, infos


class LogCallback(BaseCallback):
    """Writes proxy reward (what the agent is paid) and KL (how far it has drifted) after every rollout."""

    def __init__(self, out: Path, wrapper: RMVecEnv):
        super().__init__()
        self.out, self.w = out, wrapper
        with open(out / "rlhf.csv", "w", newline="") as f:
            csv.writer(f).writerow(["frames", "proxy_reward", "kl"])

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.w.n:
            with open(self.out / "rlhf.csv", "a", newline="") as f:
                csv.writer(f).writerow([self.num_timesteps, round(self.w.sum_r / self.w.n, 4), round(self.w.sum_kl / self.w.n, 5)])
            self.w.reset_stats()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="reference model (the 'SFT model'): where RL starts and what the KL leash is tied to")
    p.add_argument("--rm", required=True, help="reward model file")
    p.add_argument("--name", default="rlhf_a")
    p.add_argument("--frames", type=int, default=1_500_000)
    p.add_argument("--beta", type=float, default=0.05, help="KL penalty weight")
    p.add_argument("--scale", type=float, default=0.1, help="multiplier on the (standardised) learned reward")
    p.add_argument("--shift", type=float, default=0.0, help="added to the standardised learned reward every step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ent-coef", type=float, default=0.003)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--scenario-prob", type=float, default=0.4)
    p.add_argument("--max-frames", type=int, default=20_000)
    p.add_argument("--eval-every", type=int, default=250_000)
    p.add_argument("--eval-episodes", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    rm, _meta = load_rm(args.rm)
    venv = SubprocVecEnv([make_env(args.max_frames, args.seed * 1000 + i, args.scenario_prob) for i in range(args.n_envs)])
    model = PPO.load(args.ref, env=venv, device="cpu", custom_objects={
        "learning_rate": args.lr, "lr_schedule": (lambda _: args.lr), "ent_coef": args.ent_coef, "clip_range": (lambda _: args.clip),
        "gamma": args.gamma, "gae_lambda": args.gae_lambda, "n_epochs": args.epochs, "n_steps": args.n_steps, "batch_size": args.batch_size})
    ref_policy = copy.deepcopy(model.policy).eval()
    for q in ref_policy.parameters():
        q.requires_grad_(False)
    wrapper = RMVecEnv(venv, rm, ref_policy, args.scale, args.shift, args.beta)
    wrapper.model = model
    model.set_env(wrapper)

    (out / "params.json").write_text(json.dumps({**vars(args), "kind": "rlhf", "frames": args.frames, "started": time.time()}))
    (out / "control.json").unlink(missing_ok=True)
    ev = EvalCallback(out, args.eval_every, args.eval_episodes, args.max_frames)
    sc = np.array(evaluate(model, args.eval_episodes, args.max_frames))      # point 0 on the curve: the reference model itself
    with open(out / "eval.csv", "a", newline="") as f:
        csv.writer(f).writerow([0, 0.0, float(np.median(sc)), round(sc.mean()), float(np.percentile(sc, 10)), float(np.percentile(sc, 90)), int(sc.max()), 0])
    print(f"[eval]   0.0M frames  reference: median {np.median(sc):.0f} mean {sc.mean():.0f} p10 {np.percentile(sc, 10):.0f}", flush=True)
    t0 = time.time()
    model.learn(total_timesteps=args.frames, callback=[ev, LogCallback(out, wrapper), ControlCallback(out, venv)], progress_bar=False)
    model.save(out / "final")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
