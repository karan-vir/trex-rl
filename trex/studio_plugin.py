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

    primer = """
<h3>1 · The screen and its coordinates</h3>
<p>The game is drawn on a canvas <b>600 px wide and 150 px tall</b>. Positions use screen coordinates: <b>x</b> counts to the right from the left edge, and <b>y</b> counts <b>downward</b> from the top edge, so a larger y means lower on the screen. The dino's feet rest on the ground line at <b>y = 140</b>; its body is 47 px tall, so its top is at y = 93. It stands at x = 50 and is 44 px wide, so its front edge is at x = 94. Obstacles appear at the right edge (x = 600) and move left.</p>
<svg viewBox="0 0 640 190" style="width:100%;max-width:640px;background:#fff;border:1px solid #888;border-radius:8px" font-family="system-ui" font-size="11">
 <rect x="20" y="20" width="600" height="150" fill="#f7f7f7" stroke="#999"/>
 <line x1="20" y1="160" x2="620" y2="160" stroke="#444" stroke-width="2"/>
 <rect x="70" y="113" width="44" height="47" fill="#7cc3b9" stroke="#2b7a70"/>
 <rect x="450" y="125" width="25" height="35" fill="#9ccf9c" stroke="#3d7a3d"/>
 <text x="26" y="34" fill="#555">(0, 0) top-left</text>
 <text x="615" y="34" text-anchor="end" fill="#555">x grows to the right (up to 600)</text>
 <text x="26" y="62" fill="#555">y grows DOWNWARD</text>
 <text x="26" y="76" fill="#555">(150 at the bottom edge)</text>
 <text x="122" y="122" fill="#2b7a70">dino top: y = 93</text>
 <text x="122" y="139" fill="#2b7a70">dino: x from 50 to 94</text>
 <text x="122" y="178" fill="#444" font-weight="bold">ground line: y = 140 (the dino's feet)</text>
 <text x="462" y="120" fill="#3d7a3d">obstacle, moving left</text>
 <line x1="94" y1="100" x2="94" y2="165" stroke="#c0392b" stroke-dasharray="3 3"/>
 <text x="98" y="95" fill="#c0392b" dx="0">front edge x = 94</text>
</svg>
<p class="mini">(The picture adds a 20 px margin around the canvas; the numbers in the labels are the game's own coordinates.)</p>

<h3>2 · Time in the game: frames and ticks</h3>
<p><b>Frame.</b> Your screen redraws 60, 120 or 144 times per second. Each time, the browser asks the game: "update yourself and draw". One such update is a <b>frame</b>. So how many frames happen per second depends on your screen.</p>
<p><b>The problem.</b> If the game moved an obstacle a fixed 6 px every frame, it would run twice as fast on a 120 Hz screen as on a 60 Hz one. To avoid that, on every frame the game measures how much real time has passed since the previous frame and moves things in proportion.</p>
<p><b>Tick.</b> To write its speeds down, the game needs one reference length of time. It picked 1/60 of a second (16.7 ms) and we call that a <b>tick</b>. "Speed 6" means "6 px per tick", i.e. 6 px every 1/60 s, which is 360 px per second. The same yardstick is used for gravity and jump velocity below.</p>
<table><tr><th>screen</th><th class="n">one frame lasts</th><th class="n">that is</th><th class="n">an obstacle at speed 6 moves, per frame</th><th class="n">per second</th></tr>
<tr><td>60 Hz</td><td class="n">16.7 ms</td><td class="n">1 tick</td><td class="n">6 px</td><td class="n">360 px</td></tr>
<tr><td>120 Hz</td><td class="n">8.3 ms</td><td class="n">half a tick</td><td class="n">3 px</td><td class="n">360 px</td></tr></table>
<p class="mini">The game moves objects in whole pixels (it rounds down), so at 120 Hz you actually see 2 px and 3 px alternate, averaging 3.</p>
<p><b>A tick is a unit of time, not a rate.</b> The game does not run 60 times per second; that depends on your screen. "Tick" only means "1/60 s", the yardstick for every number in the game.</p>
<p><b>What the agent does.</b> It makes one decision per tick (1/60 s of game time): that is 1 frame on a 60 Hz screen and 2 frames on a 120 Hz screen. The observation "frame time" (time since the last frame ÷ 16.7 ms) tells the policy which case it is in.</p>
<p><b>One thing that does depend on the screen.</b> The game's speed-up (+0.001 to the speed) is applied once per frame, not once per tick, so on a 120 Hz screen the game gets faster twice as quickly per second.</p>
<details><summary>How I checked all this (I did not assume it)</summary>
<ul>
<li><b>The game's own code</b> (version 29 from chromedino.com, fetched earlier in this project) contains <code>FPS = 60</code>, <code>msPerFrame = 1000 / FPS</code>, a line that measures the time since the previous frame, and the movement rule <code>floor(speed × FPS / 1000 × time since last frame)</code>. This engine uses the same constants.</li>
<li><b>Measured in the real game in the browser:</b> at 120 frames per second a full jump lasted about 68 frames, which is about 34 ticks, as the tick rule predicts.</li>
<li><b>The game's clock</b> showed gaps of 8 and 9 ms between frames, which only happens if the game updates once per screen frame (120 per second), not at a fixed 60.</li>
</ul></details>

<h3>3 · A jump, tick by tick</h3>
<p>Pressing jump gives the dino an upward velocity of <b>10 + speed/10</b> px per tick (10.6 at the starting speed). Because y grows downward, upward is <b>negative</b>: the velocity starts at −10.6. Then, every tick, two things happen: the dino moves by its velocity (y += velocity), and gravity adds 0.6 to the velocity. So it rises more and more slowly, stops at the top, and falls faster and faster. Measured in the engine at the starting speed, 60 ticks per second, jump key held:</p>
<table><tr><th>tick</th><th class="n">1</th><th class="n">5</th><th class="n">10</th><th class="n">16</th><th class="n">18</th><th class="n">20</th><th class="n">25</th><th class="n">30</th><th class="n">34</th></tr>
<tr><td>vertical velocity (px/tick)</td><td class="n">−10.0</td><td class="n">−7.6</td><td class="n">−3.8</td><td class="n">−0.2</td><td class="n">+1.0</td><td class="n">+2.2</td><td class="n">+5.2</td><td class="n">+8.2</td><td class="n">+10.6</td></tr>
<tr><td>height above the ground (px)</td><td class="n">11</td><td class="n">48</td><td class="n">78</td><td class="n">92</td><td class="n">92</td><td class="n">89</td><td class="n">72</td><td class="n">39</td><td class="n">3</td></tr></table>
<p>The whole jump takes 34 to 35 ticks, about 0.58 s, and the top is about 92 px up (the simple formula 10.6² ÷ (2 × 0.6) gives 94). The vertical velocity therefore ranges from about −10.7 to +10.6 px per tick.</p>

<h3>4 · Why the observations are divided by constants (including 12)</h3>
<p>A neural network learns best when its inputs are all of similar size, ideally between about −1 and 1. The raw numbers here are not: a flag is 0 or 1, a velocity reaches ±10.7, and a distance reaches 500 or more. So each number is divided by a typical maximum. The vertical velocity peaks at about 10.7 px per tick, so dividing by <b>12</b>, a little above that, maps it into roughly −0.9 to +0.9. There is nothing special about 12; 11 or 15 would do about as well, only the scale would change. The same reasoning gives height ÷ 90 (the jump top, about 90 px), distance ÷ 600 (the screen width), and so on, as each definition below says.</p>

<h3>5 · The "stuck landing" quirk</h3>
<p>This is a quirk of the original Chrome game, ported faithfully, and it matters because the dino cannot react while stuck.</p>
<p><b>What normally happens.</b> The jump ends when the dino's y goes <i>past</i> the ground (y &gt; 93, meaning slightly below ground level); the game then puts it back at y = 93 and resets to running.</p>
<p><b>What goes wrong.</b> The game advances the jump physics by "elapsed time ÷ the frame time of the dino's current pose". For the jumping pose that frame time is 16.7 ms (one tick), but for the <b>ducking</b> pose it is 125 ms, so physics suddenly runs 7.5 times slower. If the dino touches the ground <b>exactly</b> at y = 93 (not past it) while the duck key is held, the jump has not ended (y is not &gt; 93) but the pose switches to ducking. Now each step moves it only velocity × (1/15) px, about 0.25 px, and positions are rounded to whole pixels, so the move rounds to 0. The dino hangs on the ground, still flagged as <b>jumping</b>, while its velocity creeps up slowly until a move finally rounds to 1 px. Measured at 120 Hz (screen frames of 8.3 ms):</p>
<table><tr><th>screen frame</th><th class="n">12</th><th class="n">13</th><th class="n">14</th><th class="n">15</th><th class="n">16</th><th class="n">…</th><th class="n">100</th></tr>
<tr><td>y</td><td class="n">83</td><td class="n">88</td><td class="n"><b>93</b></td><td class="n">93</td><td class="n">93</td><td class="n">93</td><td class="n">ends</td></tr>
<tr><td>pose</td><td class="n">jumping</td><td class="n">jumping</td><td class="n">ducking</td><td class="n">ducking</td><td class="n">ducking</td><td class="n">ducking</td><td class="n">running</td></tr>
<tr><td>jumping flag</td><td class="n">1</td><td class="n">1</td><td class="n">1</td><td class="n">1</td><td class="n">1</td><td class="n">1</td><td class="n">0</td></tr></table>
<p>The dino sat on the ground, unable to jump again (the game refuses a new jump while 'jumping'), for about 86 screen frames, roughly 0.7 s. Landing just one pixel deeper (y &gt; 93) would have ended the jump cleanly. In the observation this shows up as <b>height 0 while jumping = 1 and duck-anim = 1</b>. In my quick search I found it at 120 Hz; I did not find a case at 60 Hz, so it may be rarer there.</p>
"""
    obs_note = ("The policy never sees pixels, only these 22 numbers. The game canvas is 600 px wide by 150 px tall; the dino stands on a ground line "
                "at y = 140 with its top at y = 93. Most numbers are divided by a constant so they land roughly between -1 and 1, which helps the network "
                "train. Each definition says where its constant comes from: a rule of the game, a fact measured from the engine, or a design choice of this "
                "environment (it could be changed; only the scale would change).")
    obs_help = [
        "Dino height above the ground in px, divided by 90. Design choice: 90 is about the top of a full jump, measured in the engine (88 to 92 px depending on speed and "
        "timing; see Game basics), so 1.0 means the top of a jump. It is NOT the height of the visible view: the canvas is 150 px tall.",
        "Vertical velocity of the dino in px per game frame (1/60 s), divided by 12. Negative while rising, positive while falling, 0 on the ground. "
        "A jump launches at -(10 + speed/10), i.e. -10.6 at the start speed and -11.3 at the top speed (game rule); measured values stay within about "
        "-10.7 to +10.6. The 12 is a design choice slightly above that, keeping the number inside -1 to 1.",
        "1 while the game considers the dino to be in a jump, else 0. Quirk of the original game: if the dino touches the ground exactly (instead of going slightly past it) while the "
        "duck key is held, the jump does not end and it stays stuck for about 0.7 s (see 'The stuck landing' in Game basics).",
        "1 while the dino is ducking (lowered body, smaller hit box), else 0.",
        "1 if the duck key was pressed in mid-air, which makes the dino fall faster (the game's 'speed drop'), else 0.",
        "1 when the game's internal animation state is 'ducking'. Normally this equals the 'ducking' flag. It differs in the stuck-landing case: the dino "
        "lands exactly on the ground while ducking, the game keeps jump physics ('jumping' = 1) but switches the animation state to ducking. "
        "The policy needs this to recognise that situation.",
        "Game speed rescaled as (speed - 6) / (13 - 6): 0 at the starting speed, 1 at the maximum. Both 6 and 13 are rules of the game (px per frame); "
        "speed increases by 0.001 on every pass of the game loop.",
        "Obstacle speed that the policy measures itself, from how far the nearest obstacle moved since the previous step (px per 1/60 s), smoothed, "
        "divided by 13 (the game's maximum speed). It differs slightly from the game speed because the game's clock counts whole milliseconds.",
        "Time since the previous display frame divided by 16.67 ms, which is one tick, the game's unit of time (1/60 s, from its FPS = 60 constant). So about 0.5 on a "
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
    ref_steps = 3_000_000
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
