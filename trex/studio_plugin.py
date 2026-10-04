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
    obs_note = ("The policy never sees pixels, only these 22 numbers. The game canvas is 600 px wide by 150 px tall; the dino stands on a ground line "
                "at y = 140 with its top at y = 93. Most numbers are divided by a constant so they land roughly between -1 and 1, which helps the network "
                "train. Each definition says where its constant comes from: a rule of the game, a fact measured from the engine, or a design choice of this "
                "environment (it could be changed; only the scale would change).")
    obs_help = [
        "Dino height above the ground in px, divided by 90. Design choice: 90 is the apex of a full jump, measured in the engine (about 88 to 90 px at "
        "game speeds 6 and 13), so 1.0 means the top of a jump. It is NOT the height of the visible view: the canvas is 150 px tall.",
        "Vertical velocity of the dino in px per game frame (1/60 s), divided by 12. Negative while rising, positive while falling, 0 on the ground. "
        "A jump launches at -(10 + speed/10), i.e. -10.6 at the start speed and -11.3 at the top speed (game rule); measured values stay within about "
        "-10.7 to +10.6. The 12 is a design choice slightly above that, keeping the number inside -1 to 1.",
        "1 while the game considers the dino to be in a jump, else 0. Quirk of the original game: if the dino lands exactly on the ground line "
        "(rather than overshooting it), the game keeps it in jump physics for about a second (a 'stuck landing').",
        "1 while the dino is ducking (lowered body, smaller hit box), else 0.",
        "1 if the duck key was pressed in mid-air, which makes the dino fall faster (the game's 'speed drop'), else 0.",
        "1 when the game's internal animation state is 'ducking'. Normally this equals the 'ducking' flag. It differs in the stuck-landing case: the dino "
        "lands exactly on the ground while ducking, the game keeps jump physics ('jumping' = 1) but switches the animation state to ducking. "
        "The policy needs this to recognise that situation.",
        "Game speed rescaled as (speed - 6) / (13 - 6): 0 at the starting speed, 1 at the maximum. Both 6 and 13 are rules of the game (px per frame); "
        "speed rises by 0.001 per frame.",
        "Obstacle speed that the policy measures itself, from how far the nearest obstacle moved since the previous step (px per 1/60 s), smoothed, "
        "divided by 13 (the game's maximum speed). It differs slightly from the game speed because the game's clock counts whole milliseconds.",
        "Time since the previous display frame divided by 16.67 ms, which is one frame at the game's native 60 FPS (a game rule). So about 0.5 on a "
        "120 Hz screen, 1.0 at 60 Hz, and it wobbles with timing jitter. Lets the policy adapt to the screen's refresh rate.",
        "Horizontal distance from the dino's front edge to the nearest obstacle ahead, divided by 600 (the canvas width). The front edge is at x = 94 "
        "= 50 (where the dino stands) + 44 (its width), both game constants. Obstacles appear at the right edge, so real values never exceed about "
        "0.82. Value 1.5 is a design sentinel meaning 'no obstacle ahead'. Slightly negative values mean the obstacle is overlapping the dino.",
        "Width of the nearest obstacle ahead in px, divided by 75. 75 is the widest obstacle in the game (three large cacti, 3 x 25 px), so the range "
        "is 0 to 1. A small cactus is 17 px, a pterodactyl 46 px. 0 if none.",
        "Height of the nearest obstacle ahead in px, divided by 50. 50 is the tallest obstacle (a large cactus); a small cactus is 35 px, a "
        "pterodactyl 40 px. 0 if none.",
        "Clearance between the ground line (y = 140) and the bottom of the nearest obstacle, in px, divided by 60. From the engine's obstacle table: "
        "0 for a large cactus and for a low pterodactyl, 10 for a small cactus, 25 and 50 for a middle and a high pterodactyl (high ones can be walked "
        "under). The 60 is a design choice just above the largest clearance, 50 px. 0 if none.",
        "1 if the nearest obstacle is a pterodactyl (flying), 0 if it is a cactus.",
        "Same as 'obstacle 1 distance', for the second obstacle ahead, so the policy can plan the jump after the next one. 1.5 means there is none.",
        "Width of the second obstacle ahead in px, divided by 75 (see obstacle 1 width). 0 if none.",
        "Height of the second obstacle ahead in px, divided by 50 (see obstacle 1 height). 0 if none.",
        "Ground clearance of the second obstacle in px, divided by 60 (see obstacle 1 flying height). 0 if none.",
        "1 if the second obstacle is a pterodactyl.",
        "1 if no key is currently held, according to the key state the game actually received (after any input delay). The three 'holding' numbers "
        "are a one-hot encoding of the key state: exactly one of them is 1.",
        "1 if the jump key (space) is currently held down.",
        "1 if the duck key (down arrow) is currently held down.",
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
