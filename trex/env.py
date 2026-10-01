"""Gymnasium wrapper around TRexGame: the agent's eyes (observation), hands (action),
and report card (reward).

    env = TRexEnv()
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

`terminated` = the dino crashed (a real ending of the game).
`truncated`  = we cut the episode short (time limit) while the dino was still alive.
RL algorithms treat the two differently, which is why Gymnasium keeps them apart.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trex.game import (
    DINO_W, DINO_X, DUCK, JUMP, MAX_SPEED, NOOP, START_SPEED, WORLD_WIDTH, TRexGame,
)

ACTION_NAMES = {NOOP: "NOOP", JUMP: "JUMP", DUCK: "DUCK"}
N_OBSTACLES = 2   # how many upcoming obstacles the agent can see

# Feature names, in the order they appear in the observation vector.
OBS_NAMES = ["dino_y", "dino_vy", "speed", "ducking"] + [
    f"obs{i}_{f}" for i in range(N_OBSTACLES) for f in ("dist", "width", "height", "y")
]

# Scale factors turn pixels into numbers that are roughly -1..1. Neural networks
# learn much more easily from small, similar-sized numbers than from raw pixels.
Y_SCALE, VY_SCALE = 90.0, 10.0
SIZE_SCALE, FLY_SCALE = 50.0, 60.0
NO_OBSTACLE_DIST = 1.5   # "nothing there" looks like an obstacle far away, size zero


class TRexEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        survive_reward: float = 1.0,
        death_reward: float = -100.0,
        pass_reward: float = 0.0,
        max_episode_steps: int = 10_000,
    ):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.survive_reward = survive_reward
        self.death_reward = death_reward
        self.pass_reward = pass_reward
        self.max_episode_steps = max_episode_steps

        self.action_space = spaces.Discrete(3)  # NOOP, JUMP, DUCK
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(len(OBS_NAMES),), dtype=np.float32
        )
        self.game = TRexGame()
        self._renderer = None
        self._last_action = NOOP

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)   # sets up self.np_random from the seed
        # Derive the game's own seed from Gymnasium's RNG, so reset(seed=7)
        # always gives the same obstacle course.
        self.game.reset(int(self.np_random.integers(2**31)))
        self._last_action = NOOP
        if self.render_mode == "human":
            self.render()
        return self._observe(), self._info()

    def step(self, action):
        passed_before = self.game.passed
        self._last_action = int(action)
        alive = self.game.step(int(action))

        reward = self.survive_reward if alive else self.death_reward
        reward += self.pass_reward * (self.game.passed - passed_before)

        terminated = not alive
        truncated = alive and self.game.ticks >= self.max_episode_steps
        if self.render_mode == "human":
            self.render()
        return self._observe(), float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------------
    def _observe(self) -> np.ndarray:
        g = self.game
        vec = [
            g.dino_y / Y_SCALE,
            g.dino_vy / VY_SCALE,
            (g.speed - START_SPEED) / (MAX_SPEED - START_SPEED),
            float(g.ducking),
        ]
        upcoming = g.next_obstacles(N_OBSTACLES)
        for i in range(N_OBSTACLES):
            if i < len(upcoming):
                o = upcoming[i]
                vec += [
                    (o.x - (DINO_X + DINO_W)) / WORLD_WIDTH,   # gap to the dino's nose
                    o.w / SIZE_SCALE,
                    o.h / SIZE_SCALE,
                    o.y / FLY_SCALE,
                ]
            else:
                vec += [NO_OBSTACLE_DIST, 0.0, 0.0, 0.0]
        return np.clip(np.array(vec, dtype=np.float32), -3.0, 3.0)

    def _info(self) -> dict:
        g = self.game
        return {"score": int(g.score * 10), "ticks": g.ticks, "passed": g.passed, "speed": g.speed}

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from trex.render import Renderer   # imported lazily: headless training never loads pygame
            self._renderer = Renderer(self.render_mode, fps=self.metadata["render_fps"])
        import pygame
        pygame.event.pump()    # keeps the window responsive
        return self._renderer.draw(self.game, [f"action: {ACTION_NAMES[self._last_action]}"])

    def toggle_fullscreen(self):
        if self._renderer is not None and self.render_mode == "human":
            self._renderer.toggle_fullscreen()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
