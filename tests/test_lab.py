"""The training lab runs headless: every agent kind steps, renders, and the controls change state."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from trex.lab.app import Lab
from trex.lab.session import Config, Session


def test_session_kinds_step_and_finish():
    for kind in ("rule", "random", "human"):
        s = Session(kind, Config(seed=1))
        for _ in range(3000):
            s.step(0)
            if s.done:
                break
        assert s.decisions > 0 and s.score >= 0


def test_session_is_deterministic():
    runs = []
    for _ in range(2):
        s = Session("rule", Config(seed=4))
        for _ in range(600):
            s.step()
        runs.append((s.score, s.done))
    assert runs[0] == runs[1]


def test_lab_controls_and_render():
    lab = Lab(seed=2)
    for k in (pygame.K_2, pygame.K_4, pygame.K_1, pygame.K_c, pygame.K_v, pygame.K_h, pygame.K_j, pygame.K_l, pygame.K_s, pygame.K_n):
        assert lab.key(k) is True
        for _ in range(20):
            lab.update(1 / 60)
            lab.render()
    assert lab.cfg.latency == 1 and lab.cfg.scenario is True
    assert lab.key(pygame.K_ESCAPE) is False
