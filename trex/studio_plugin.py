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
    obs_help = [
        "Height of the dino above the ground, divided by 90 px. 0 on the ground, about 1 at the top of a full jump.",
        "Vertical velocity of the dino, divided by 12. Negative while rising, positive while falling, 0 on the ground.",
        "1 while the game considers the dino to be in a jump. Quirk: a landing exactly on the ground leaves this at 1 for about a second (a 'stuck landing').",
        "1 while the dino is ducking (lowered body, smaller hit box).",
        "1 if the duck key was pressed in mid-air to fall faster (fast fall / speed drop).",
        "1 if the animation state is 'ducking' even though the dino is airborne. It drives the stuck-landing quirk, so the policy needs it.",
        "Game speed, rescaled so 0 is the starting speed (6) and 1 the maximum (13). It rises slowly through the game.",
        "Obstacle speed the policy measures itself from how far the nearest obstacle moved since the last step, smoothed, divided by 13. It differs from the game speed because the game's clock is in whole milliseconds.",
        "Time since the previous display frame divided by 16.7 ms. About 0.5 at 120 Hz and 1.0 at 60 Hz, so the policy can adapt to the screen's refresh rate and jitter.",
        "Horizontal gap between the dino's front edge and the nearest obstacle ahead, divided by the 600 px screen width. 1.5 means no obstacle ahead.",
        "Width of the nearest obstacle ahead divided by 75 px (a group of cacti is wider). 0 if none.",
        "Height of the nearest obstacle ahead divided by 50 px. 0 if none.",
        "Gap between the ground and the bottom of the nearest obstacle, divided by 60 px. 0 for cacti; for pterodactyls it tells whether to duck under or jump over. 0 if none.",
        "1 if the nearest obstacle is a pterodactyl, 0 if it is a cactus.",
        "Same as obstacle 1 distance, for the second obstacle ahead (to plan the jump after the next). 1.5 means there is none.",
        "Width of the second obstacle ahead divided by 75 px. 0 if none.",
        "Height of the second obstacle ahead divided by 50 px. 0 if none.",
        "Ground-to-bottom gap of the second obstacle divided by 60 px. 0 if none or a cactus.",
        "1 if the second obstacle is a pterodactyl.",
        "1 if no key is currently held (the key state the game actually received, after any input delay).",
        "1 if the jump key is currently held down.",
        "1 if the duck key is currently held down.",
    ]
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
