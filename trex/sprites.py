"""Pixel-art sprites, drawn from code (no image files to manage).

Each sprite is a small grid of palette letters ('.' = transparent) that gets
scaled up 2x. Built from rectangles so it is easy to tweak: change a number,
re-run, see the result. Sprites are cached after the first build.
"""

from __future__ import annotations

from functools import lru_cache

import pygame

SCALE = 2

PALETTE = {
    # dino
    "G": (70, 170, 160),    # body
    "B": (205, 238, 228),   # belly
    "S": (240, 150, 60),    # back spikes
    "P": (245, 150, 165),   # blush
    "D": (35, 85, 85),      # feet
    "M": (40, 40, 40),      # mouth
    "R": (225, 60, 70),     # tongue
    # shared
    "K": (30, 30, 30),
    "W": (255, 255, 255),
    # cactus
    "C": (95, 165, 70),
    "c": (65, 125, 50),
    "L": (215, 235, 160),
    # pterodactyl
    "Y": (150, 95, 160),    # body
    "y": (110, 65, 125),    # wings
    "O": (240, 170, 50),    # beak
}


def _grid(w: int, h: int):
    return [["."] * w for _ in range(h)]


def _rect(g, x0, y0, x1, y1, c):
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            g[y][x] = c


def _px(g, x, y, c):
    g[y][x] = c


def _surface(g) -> pygame.Surface:
    h, w = len(g), len(g[0])
    surf = pygame.Surface((w * SCALE, h * SCALE), pygame.SRCALPHA)
    for y, row in enumerate(g):
        for x, ch in enumerate(row):
            if ch != ".":
                surf.fill(PALETTE[ch], (x * SCALE, y * SCALE, SCALE, SCALE))
    return surf


# --- Dino ----------------------------------------------------------------
def _eye(g, x, y, eyes: str):
    """Draw an eye with its top-left at (x, y). Expressions: normal/blink/scared/dead."""
    if eyes == "blink":
        _rect(g, x, y + 1, x + 1, y + 1, "K")
    elif eyes == "scared":
        _rect(g, x - 1, y - 1, x + 1, y + 1, "W")
        _px(g, x, y, "K")
    elif eyes == "dead":
        for dx, dy in ((-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)):
            _px(g, x + dx, y + dy, "K")
    else:
        _rect(g, x, y, x + 1, y + 1, "W")
        _px(g, x + 1, y + 1, "K")


@lru_cache(maxsize=None)
def dino(pose: str, eyes: str) -> pygame.Surface:
    """pose: run_a | run_b | jump | duck_a | duck_b.  eyes: normal|blink|scared|dead."""
    if pose.startswith("duck"):
        return _dino_duck(pose, eyes)
    g = _grid(22, 24)
    _rect(g, 3, 10, 7, 13, "G"); _rect(g, 1, 11, 3, 12, "G"); _px(g, 0, 12, "G")   # tail
    _rect(g, 6, 9, 17, 17, "G")                                                   # body
    _rect(g, 12, 12, 16, 17, "B")                                                 # belly
    _rect(g, 13, 0, 21, 7, "G"); _rect(g, 12, 8, 17, 8, "G")                      # head, neck
    _rect(g, 18, 6, 21, 7 if eyes == "scared" else 6, "M")                        # mouth
    _rect(g, 16, 5, 17, 5, "P")                                                   # blush
    _rect(g, 17, 12, 19, 12, "G"); _px(g, 19, 13, "G")                            # tiny arm
    for x, y in ((7, 8), (10, 8), (4, 9)):                                        # spikes
        _px(g, x, y, "S")
    if eyes == "dead":
        _rect(g, 20, 7, 20, 8, "R")                                               # tongue
    _eye(g, 16, 2, eyes)
    left_up, right_up = pose == "run_b", pose == "run_a"
    for x, up in ((7, left_up), (12, right_up)):
        if up:
            _rect(g, x, 18, x + 2, 19, "G"); _rect(g, x, 20, x + 3, 20, "D")
        else:
            _rect(g, x, 18, x + 2, 22, "G"); _rect(g, x, 23, x + 3, 23, "D")
    return _surface(g)


def _dino_duck(pose: str, eyes: str) -> pygame.Surface:
    g = _grid(30, 12)
    _rect(g, 0, 5, 5, 8, "G"); _px(g, 0, 6, "G")                                  # tail
    _rect(g, 5, 3, 22, 9, "G")                                                    # long body
    _rect(g, 10, 7, 20, 9, "B")                                                   # belly
    _rect(g, 21, 0, 29, 6, "G")                                                   # head
    _rect(g, 25, 5, 29, 5, "M")                                                   # mouth
    _rect(g, 24, 4, 25, 4, "P")
    for x in (8, 11, 14, 17):
        _px(g, x, 2, "S")
    _eye(g, 25, 1, eyes if eyes != "scared" else "normal")
    left_up = pose == "duck_b"
    for x, up in ((8, left_up), (16, not left_up)):
        if up:
            _rect(g, x, 10, x + 3, 10, "D")
        else:
            _rect(g, x, 10, x + 2, 10, "G"); _rect(g, x, 11, x + 3, 11, "D")
    return _surface(g)


# --- Obstacles -----------------------------------------------------------
@lru_cache(maxsize=None)
def cactus_small() -> pygame.Surface:
    """Small and cheerful."""
    g = _grid(9, 18)
    _rect(g, 3, 1, 5, 17, "C"); _px(g, 4, 0, "C"); _rect(g, 5, 1, 5, 17, "c")
    _rect(g, 1, 10, 3, 11, "C"); _rect(g, 1, 7, 2, 11, "C")      # left arm
    _rect(g, 5, 8, 7, 9, "C"); _rect(g, 7, 5, 8, 9, "C")         # right arm
    for x, y in ((1, 8), (8, 6), (4, 13), (4, 15)):
        _px(g, x, y, "L")
    _px(g, 3, 3, "K"); _px(g, 5, 3, "K")                          # eyes
    _px(g, 3, 5, "K"); _px(g, 5, 5, "K"); _px(g, 4, 6, "K")       # smile
    return _surface(g)


@lru_cache(maxsize=None)
def cactus_large() -> pygame.Surface:
    """Big and grumpy."""
    g = _grid(13, 25)
    _rect(g, 5, 1, 8, 24, "C"); _rect(g, 6, 0, 7, 0, "C"); _rect(g, 8, 1, 8, 24, "c")
    _rect(g, 1, 15, 4, 16, "C"); _rect(g, 1, 9, 2, 16, "C")      # left arm
    _rect(g, 9, 12, 11, 13, "C"); _rect(g, 11, 6, 12, 13, "C")   # right arm
    for x, y in ((2, 11), (12, 8), (6, 18), (7, 21), (6, 4)):
        _px(g, x, y, "L")
    _px(g, 5, 7, "K"); _px(g, 8, 7, "K")                          # eyes
    _px(g, 5, 5, "K"); _px(g, 6, 6, "K"); _px(g, 8, 5, "K"); _px(g, 7, 6, "K")  # angry brows
    _px(g, 6, 10, "K"); _px(g, 7, 10, "K"); _px(g, 5, 11, "K"); _px(g, 8, 11, "K")  # frown
    return _surface(g)


@lru_cache(maxsize=None)
def ptero(flap_up: bool) -> pygame.Surface:
    """Pterodactyl with a one-frame wing flap."""
    g = _grid(23, 15)
    _rect(g, 0, 8, 6, 9, "Y"); _rect(g, 6, 6, 17, 10, "Y")       # tail, body
    _rect(g, 16, 5, 19, 8, "Y"); _rect(g, 20, 6, 22, 7, "O")     # head, beak
    _rect(g, 2, 7, 5, 7, "Y")                                     # crest
    _px(g, 17, 6, "W"); _px(g, 18, 6, "K")                        # eye
    if flap_up:
        for i, (a, b) in enumerate(((8, 14), (9, 13), (10, 12), (10, 11))):
            _rect(g, a, 5 - i, b, 5 - i, "y")
    else:
        for i, (a, b) in enumerate(((8, 14), (9, 13), (10, 12), (10, 11))):
            _rect(g, a, 11 + i, b, 11 + i, "y")
    return _surface(g)
