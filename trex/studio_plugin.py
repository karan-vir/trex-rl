"""T-Rex as a problem plug-in for RLHF Studio (https://github.com/karan-vir/rlhf-studio, private).

    PYTHONPATH=/path/to/t-rex-runner rlhf-studio --problem trex.studio_plugin:TRexProblem

Needs the studio's Python environment (rlhf_studio, stable-baselines3) plus this repo on PYTHONPATH.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

from rlhf_studio.problem import Problem, Reference
from trex.real_env import RealTRexEnv
from trex.lab.session import OBS_NAMES

ROOT = Path(__file__).resolve().parents[1]


class TRexProblem(Problem):
    name = "trex"
    title = "Chrome T-Rex Runner"
    blurb = ("A faithful port of the Chrome dino game engine. The policy sees 22 numbers (height, speed, the next two obstacles, "
             "held keys) and presses nothing / jump / duck about 60 times a second.")
    action_names = ["no key", "jump (held)", "duck (held)"]
    obs_names = OBS_NAMES
    gold_label = "game score"
    gold_cap = 6200.0
    clip_len = 150
    max_steps = 20_000
    rl_steps = 1_500_000
    parallel = True
    judge_help = ("Better = the dino survives and clears the obstacles without crashing. Both crash: the one that lasts longer is better. "
                  "Both survive: they are the same unless one clears more obstacles.")

    def make_env(self, spec, render=False):
        return RealTRexEnv(max_frames=10**9, randomize=False, frame_ms=1000 / 120, jitter_ms=0.3, latency=0,
                           start_speed=spec["speed"], skip_clear_time=spec["skip"], scenario_prob=1.0 if spec["scenario"] else 0.0)

    def sample_spec(self, rng, mode="train"):
        seed = int(rng.integers(10**9))
        if mode == "eval":                                   # a normal game from the standard start
            return {"seed": seed, "speed": 6.0, "skip": False, "scenario": False}
        if mode == "clip":                                   # decisions matter right away: obstacles at once, often dangerous
            return {"seed": seed, "speed": round(float(rng.uniform(6.5, 11.0)), 2), "skip": True, "scenario": bool(rng.random() < 0.6)}
        return {"seed": seed, "speed": round(float(rng.uniform(6.0, 11.0)), 2), "skip": bool(rng.random() < 0.5), "scenario": bool(rng.random() < 0.4)}

    def outcome(self, env, terminated, steps):
        return {"failed": bool(terminated), "steps": int(steps), "passes": len(env.passed), "score": int(env.engine.score)}

    def judge(self, a, b):
        if a["failed"] != b["failed"]:
            return "B" if a["failed"] else "A"
        if a["failed"]:
            return "same" if abs(a["steps"] - b["steps"]) < 10 else ("A" if a["steps"] > b["steps"] else "B")
        if a["passes"] != b["passes"] and max(a["passes"], b["passes"]) <= 40:
            return "A" if a["passes"] > b["passes"] else "B"
        return "same"

    def gold(self, outcome):
        return float(outcome["score"])

    def render(self, env):
        import pygame
        from trex.lab import draw
        if not hasattr(self, "_surf"):
            pygame.font.init()
            self._surf, self._f = pygame.Surface((600, 150)), draw.fonts()

        class Shim:
            pass
        sh = Shim()
        sh.engine, sh.done, sh.crash_text = env.engine, env.engine.crashed, "CRASH" if env.engine.crashed else ""
        draw.draw_game(self._surf, sh, 0, 0, 1, False, "", self._f)
        return np.ascontiguousarray(pygame.surfarray.array3d(self._surf).transpose(1, 0, 2))

    def references(self):
        out = []
        for name, desc in (("ckpt_007M", "flaky: strong median but weak worst-case games (recommended)"), ("ckpt_001M", "an early checkpoint"), ("best", "the strongest of the original runs")):
            p = ROOT / "runs" / "ppo_d" / name
            if p.with_suffix(".zip").exists():
                out.append(Reference(f"ppo_d/{name}", str(p), desc))
        return out
