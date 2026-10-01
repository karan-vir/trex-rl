"""Presses random buttons. The floor every other agent should beat."""

import numpy as np


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs) -> int:
        return int(self.rng.integers(3))
