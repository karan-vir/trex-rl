"""One fair way to measure every agent: same seeds, same statistics.

An agent is anything with `act(obs) -> action`. Evaluating on FIXED seeds means two
agents face exactly the same obstacle courses, so score differences come from the
agent, not from luck of the draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trex.env import TRexEnv


@dataclass
class Result:
    scores: list[int]
    ticks: list[int]
    survived: int          # episodes that hit the time limit alive
    deaths: list[str]      # what killed the dino, per crashed episode

    def summary(self) -> str:
        n = len(self.scores)
        return (
            f"{n} episodes | score mean {np.mean(self.scores):.0f}  "
            f"median {np.median(self.scores):.0f}  min {min(self.scores)}  max {max(self.scores)} | "
            f"survived to time limit: {self.survived}/{n}"
        )


def evaluate(agent, seeds, max_episode_steps: int = 10_000, **env_kwargs) -> Result:
    env = TRexEnv(max_episode_steps=max_episode_steps, **env_kwargs)
    scores, ticks, deaths, survived = [], [], [], 0
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(agent.act(obs))
            done = terminated or truncated
        scores.append(info["score"])
        ticks.append(info["ticks"])
        if truncated:
            survived += 1
        else:
            crashed = [o for o in env.game.obstacles if o.x < 120 and o.x + o.w > 40]
            deaths.append(crashed[0].kind if crashed else "unknown")
    env.close()
    return Result(scores, ticks, survived, deaths)
