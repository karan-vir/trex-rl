"""Draws a TRexGame with Pygame. It only READS the game; it never changes it.

Because of that, the same renderer works for you playing, an agent playing, or
a replay. Pass `info` lines (e.g. "action: JUMP") to show extra text on screen.

All animation (running legs, wing flaps, blinking) is driven by `game.ticks`,
not the wall clock, so a replay looks exactly like the original run.

The game is always drawn at its fixed logical size (600x200) and Pygame scales
it to the window, so fullscreen never changes the physics.
"""

from __future__ import annotations

import random

import numpy as np
import pygame

from trex import sprites
from trex.game import (
    CACTUS_LARGE, CACTUS_SMALL, DINO_X, PTERO, WORLD_WIDTH, TRexGame,
)

HEIGHT = 200
GROUND_Y = 160          # screen y of the ground line
BG, FG = (247, 247, 247), (60, 60, 60)
SCARED_DISTANCE = 130   # the dino panics when an obstacle is this close

_rng = random.Random(1)  # fixed decoration layout
PEBBLES = [(_rng.randrange(WORLD_WIDTH), _rng.randrange(6, 30), _rng.choice((2, 3, 5)))
           for _ in range(45)]
CLOUDS = [(40, 30), (230, 55), (420, 25), (560, 60)]


class Renderer:
    def __init__(self, mode: str = "human", fps: int = 60):
        self.mode, self.fps = mode, fps
        pygame.init()
        if mode == "human":
            # SCALED: draw at 600x200, let Pygame stretch it to the window/fullscreen.
            self.screen = pygame.display.set_mode(
                (WORLD_WIDTH, HEIGHT), pygame.SCALED | pygame.RESIZABLE
            )
            pygame.display.set_caption("T-Rex RL")
        else:
            self.screen = pygame.display.set_mode((WORLD_WIDTH, HEIGHT), pygame.HIDDEN)
        self.font = pygame.font.SysFont("menlo,monospace", 14)
        self.clock = pygame.time.Clock()

    # ------------------------------------------------------------------
    def toggle_fullscreen(self):
        pygame.display.toggle_fullscreen()

    @property
    def is_fullscreen(self) -> bool:
        return pygame.display.is_fullscreen()

    # ------------------------------------------------------------------
    def draw(self, game: TRexGame, info: list[str] | None = None):
        s = self.screen
        s.fill(BG)
        self._draw_background(s, game)
        for o in game.obstacles:
            self._draw_obstacle(s, game, o)
        self._draw_dino(s, game)
        self._draw_hud(s, game, info)

        if self.mode == "human":
            pygame.display.flip()
            self.clock.tick(self.fps)
        else:
            return self.frame()

    def frame(self) -> np.ndarray:
        """The current screen as an (H, W, 3) array - handy for videos/GIFs."""
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def close(self):
        pygame.quit()

    # ------------------------------------------------------------------
    def _draw_background(self, s, game):
        # clouds drift slowly (parallax: they move slower than the ground)
        span = WORLD_WIDTH + 100
        for cx, cy in CLOUDS:
            x = (cx - game.distance * 0.15) % span - 50
            for dx, dy, w in ((0, 6, 44), (10, 0, 24), (24, 3, 28)):
                pygame.draw.rect(s, (225, 232, 238), (x + dx, cy + dy, w, 10), border_radius=5)
        pygame.draw.line(s, FG, (0, GROUND_Y), (WORLD_WIDTH, GROUND_Y), 2)
        for px, dy, w in PEBBLES:
            x = (px - game.distance) % WORLD_WIDTH
            pygame.draw.line(s, (175, 175, 175), (x, GROUND_Y + dy), (x + w, GROUND_Y + dy), 2)

    def _draw_obstacle(self, s, game, o):
        if o.kind == CACTUS_SMALL:
            img = sprites.cactus_small()
        elif o.kind == CACTUS_LARGE:
            img = sprites.cactus_large()
        else:
            img = sprites.ptero((game.ticks // 8) % 2 == 0)
        s.blit(img, (o.x, GROUND_Y - o.y - img.get_height()))

    def _draw_dino(self, s, game):
        if game.done:
            eyes = "dead"
        elif any(0 < o.x - DINO_X < SCARED_DISTANCE for o in game.next_obstacles(1)):
            eyes = "scared"
        elif game.ticks % 200 < 6:
            eyes = "blink"
        else:
            eyes = "normal"

        step = (game.ticks // 5) % 2 == 0
        if not game.on_ground:
            pose = "jump"
        elif game.ducking:
            pose = "duck_a" if step else "duck_b"
        else:
            pose = "run_a" if step else "run_b"
        if game.done:
            pose = "duck_a" if game.ducking else "jump" if not game.on_ground else "run_a"

        img = sprites.dino(pose, eyes)
        s.blit(img, (DINO_X, GROUND_Y - game.dino_y - img.get_height()))

    def _draw_hud(self, s, game, info):
        lines = [f"score {int(game.score * 10):05d}   speed {game.speed:.1f}"] + (info or [])
        for i, line in enumerate(lines):
            s.blit(self.font.render(line, True, FG), (10, 8 + 18 * i))
        if game.done:
            msg = self.font.render("GAME OVER  -  press R", True, (200, 50, 50))
            s.blit(msg, msg.get_rect(center=(WORLD_WIDTH // 2, 80)))
