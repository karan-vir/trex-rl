"""Gymnasium environment on top of the faithful engine (trex/engine.py).

One env step = ONE DISPLAY FRAME (60, 120 or 144 per second, randomised per episode), exactly what a
script running in the browser sees. The agent holds one of three key states; key EVENTS are produced
only on changes, the way a person (or the browser script) would press and release keys:

    action 0: no key held     action 1: JUMP held (space)     action 2: DUCK held (down arrow)

Every number in the observation can be read in the browser from the game's own objects
(Runner.instance_.tRex, .horizon.obstacles, currentSpeed) plus the frame clock, so a trained policy
can be exported to JavaScript and run on the real game unchanged.

Domain randomisation (what is NOT known exactly about the real machine):
  * display rate and the exact frame time (the integer-ms clock makes landings timing-sensitive),
  * timing jitter, and 0-1 frames of input latency,
  * starting speed, so the high-speed regime is trained on from the start.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trex.engine import (
    CACTUS_SMALL, DUCKING, GROUND_Y, MAX_SPEED, MS_PER_FRAME, SPEED0, T_START_X, T_WIDTH, WIDTH,
    RealEngine,
)

OBS_DIM = 22
NOSE_X = T_START_X + T_WIDTH               # 94: the dino's front edge
GROUND_BOTTOM = GROUND_Y + 47              # 140: y of the ground line


class RealTRexEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        max_frames: int = 20_000,
        randomize: bool = True,
        frame_ms: float | None = None,        # fixed display timing (evaluation); None = randomise
        jitter_ms: float | None = None,
        latency: int | None = None,
        start_speed: float | None = None,     # None = randomise when `randomize`, else normal start
        skip_clear_time: bool | None = None,
        alive_reward: float = 0.01,
        death_penalty: float = 1.0,
        pass_bonus: float = 0.3,
        decision_ms: float = 1000.0 / 60.0,     # how often the agent decides (keys stay held in between)
        scenario_prob: float = 0.0,             # share of episodes that start with a close, random obstacle pair
    ):
        super().__init__()
        self.max_frames = max_frames
        self.randomize = randomize
        self.fixed = dict(frame_ms=frame_ms, jitter_ms=jitter_ms, latency=latency,
                          start_speed=start_speed, skip_clear_time=skip_clear_time)
        self.alive_reward = alive_reward
        self.death_penalty = death_penalty
        self.pass_bonus = pass_bonus
        self.decision_ms = decision_ms
        self.scenario_prob = scenario_prob
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(OBS_DIM,), dtype=np.float32)
        self.engine: RealEngine | None = None
        self._obs = np.zeros(OBS_DIM, dtype=np.float32)

    # ------------------------------------------------------------------
    def _draw_config(self):
        rng = self.np_random
        f = self.fixed
        if self.randomize:
            rate = rng.choice([60.0, 120.0, 144.0], p=[0.25, 0.6, 0.15])
            frame_ms = 1000.0 / rate * rng.uniform(0.97, 1.03)
            jitter = rng.uniform(0.0, 0.4)
            latency = int(rng.choice([0, 1], p=[0.7, 0.3]))
            if rng.random() < 0.5:
                start_speed, skip = float(rng.uniform(SPEED0, MAX_SPEED)), True
            else:
                start_speed, skip = SPEED0, False
        else:
            frame_ms, jitter, latency, start_speed, skip = 1000.0 / 120.0, 0.3, 0, SPEED0, False
        if f["frame_ms"] is not None: frame_ms = f["frame_ms"]
        if f["jitter_ms"] is not None: jitter = f["jitter_ms"]
        if f["latency"] is not None: latency = f["latency"]
        if f["start_speed"] is not None: start_speed = f["start_speed"]
        if f["skip_clear_time"] is not None: skip = f["skip_clear_time"]
        return frame_ms, jitter, latency, start_speed, skip

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        frame_ms, jitter, latency, start_speed, skip = self._draw_config()
        self.engine = RealEngine(seed=int(self.np_random.integers(2**31)), frame_ms=frame_ms,
                                 jitter_ms=jitter, start_speed=start_speed, skip_clear_time=skip)
        self.latency = latency
        self.skip = max(1, round(self.decision_ms / frame_ms))     # display frames per decision
        self.passed: set[int] = set()
        self.pending: list[int] = [0] * latency
        self.held = 0                      # key state actually applied to the game
        self.prev_x = None
        self.eff_speed = start_speed       # smoothed observed obstacle speed (px per 1/60 s)
        if self.scenario_prob > 0 and self.np_random.random() < self.scenario_prob:
            self._scenario(skip)
        self.engine.frame()                # first frame so dt and positions exist
        self.t = 0
        self.last_elapsed = float(getattr(self.engine, "last_dt", 8))
        return self._observe(), {}

    # ------------------------------------------------------------------
    def _scenario(self, skip_clear: bool) -> None:
        """Start with rare, dangerous situations on the screen instead of waiting for them to occur.

        Half the time an obstacle is already overhead or beside the dino (a high pterodactyl passing
        above a standing dino is the classic one); then a second obstacle of any kind follows at a
        random distance. Kinds and heights are uniform, not the game's natural mix.
        """
        rng, e = self.np_random, self.engine
        e.running_time = float(3001)                       # obstacles are active immediately
        e.speed = max(e.speed, float(rng.uniform(8.5, 13.0)))
        e.obstacles.clear()
        if rng.random() < 0.5:
            e.inject(2, NOSE_X + float(rng.uniform(-60, 40)), y_index=0)                  # high pterodactyl overhead
            nxt = NOSE_X + float(rng.uniform(150, 500))
        else:
            nxt = NOSE_X + float(rng.uniform(40, 350))
            e.inject(int(rng.integers(0, 3)), nxt, y_index=int(rng.integers(0, 3)), size=int(rng.integers(1, 4)))
            nxt += float(rng.uniform(120, 400))
        last = e.inject(int(rng.integers(0, 3)), nxt, y_index=int(rng.integers(0, 3)), size=int(rng.integers(1, 4)))
        last.gap = int(rng.integers(150, 500))

    def _apply_keys(self, new: int) -> None:
        e = self.engine
        if new != self.held:
            if self.held == 1: e.key_up("jump")
            elif self.held == 2: e.key_up("duck")
            if new == 1: e.key_down("jump")
            elif new == 2: e.key_down("duck")
            self.held = new

    def step(self, action):
        e = self.engine
        self.pending.append(int(action))
        self._apply_keys(self.pending.pop(0))
        alive, elapsed = True, 0
        for _ in range(self.skip):
            alive = e.frame()
            elapsed += getattr(e, "last_dt", 8)
            if not alive:
                break
        self.last_elapsed = float(max(elapsed, 1))
        self.t += 1
        reward = self.alive_reward * elapsed / MS_PER_FRAME
        for ob in e.obstacles:                                   # obstacles the dino has now left behind
            if ob.x + ob.width < T_START_X and id(ob) not in self.passed:
                self.passed.add(id(ob))
                reward += self.pass_bonus
        if len(self.passed) > 40:
            live = {id(o) for o in e.obstacles}
            self.passed &= live
        terminated = not alive
        if terminated:
            reward -= self.death_penalty
        truncated = alive and self.t >= self.max_frames
        info = {}
        if terminated or truncated:
            info = {"score": e.score, "frames": self.t, "speed": e.speed}
        return self._observe(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def _observe(self) -> np.ndarray:
        e, o = self.engine, self._obs
        dt = self.last_elapsed
        ahead = [ob for ob in e.obstacles if ob.x + ob.width > T_START_X][:2]
        # observed obstacle speed from the first obstacle's movement since the last frame
        if ahead and self.prev_x is not None and self.prev_x[0] is ahead[0]:
            moved = self.prev_x[1] - ahead[0].x
            if dt > 0:
                self.eff_speed += 0.1 * (moved * MS_PER_FRAME / dt - self.eff_speed)
        self.prev_x = (ahead[0], ahead[0].x) if ahead else None
        o[0] = e.height / 90.0
        o[1] = e.vel / 12.0
        o[2] = float(e.jumping)
        o[3] = float(e.ducking)
        o[4] = float(e.speed_drop)
        o[5] = float(e.status == DUCKING)
        o[6] = (e.speed - SPEED0) / (MAX_SPEED - SPEED0)
        o[7] = dt / MS_PER_FRAME
        o[8] = self.eff_speed / MAX_SPEED
        for k in range(2):
            base = 9 + 5 * k
            if k < len(ahead):
                ob = ahead[k]
                o[base] = (ob.x - NOSE_X) / WIDTH
                o[base + 1] = ob.width / 75.0
                o[base + 2] = ob.height / 50.0
                o[base + 3] = (GROUND_BOTTOM - (ob.y + ob.height)) / 60.0
                o[base + 4] = 1.0 if ob.kind == 2 else 0.0
            else:
                o[base:base + 5] = (1.5, 0.0, 0.0, 0.0, 0.0)
        o[19:22] = 0.0
        o[19 + self.held] = 1.0
        np.clip(o, -5.0, 5.0, out=o)
        return o.copy()
