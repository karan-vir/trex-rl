"""Draws a TRexGame with Pygame. It only READS the game; it never changes it.

Because of that, the same renderer works for you playing, an agent playing, or
a replay. Pass `info` lines (e.g. "action: JUMP") to show extra text on screen.
"""

from __future__ import annotations

import numpy as np
import pygame

from trex.game import (
    CACTUS_LARGE, CACTUS_SMALL, DINO_X, PTERO, WORLD_WIDTH, TRexGame,
)

HEIGHT = 200
GROUND_Y = 160          # screen y of the ground line
BG, FG = (247, 247, 247), (60, 60, 60)
COLORS = {CACTUS_SMALL: (40, 130, 60), CACTUS_LARGE: (30, 100, 50), PTERO: (190, 90, 40)}


class Renderer:
    def __init__(self, mode: str = "human", fps: int = 60):
        self.mode, self.fps = mode, fps
        pygame.init()
        flags = 0 if mode == "human" else pygame.HIDDEN
        self.screen = pygame.display.set_mode((WORLD_WIDTH, HEIGHT), flags)
        self.font = pygame.font.SysFont("menlo,monospace", 14)
        self.clock = pygame.time.Clock()

    def draw(self, game: TRexGame, info: list[str] | None = None):
        s = self.screen
        s.fill(BG)
        pygame.draw.line(s, FG, (0, GROUND_Y), (WORLD_WIDTH, GROUND_Y), 2)

        for o in game.obstacles:
            rect = pygame.Rect(o.x, GROUND_Y - o.y - o.h, o.w, o.h)
            pygame.draw.rect(s, COLORS[o.kind], rect)

        w, h = game.dino_size
        dino = pygame.Rect(DINO_X, GROUND_Y - game.dino_y - h, w, h)
        pygame.draw.rect(s, (200, 50, 50) if game.done else FG, dino)

        lines = [f"score {int(game.score * 10):05d}   speed {game.speed:.1f}"] + (info or [])
        for i, line in enumerate(lines):
            s.blit(self.font.render(line, True, FG), (10, 8 + 18 * i))
        if game.done:
            msg = self.font.render("GAME OVER  -  press R", True, (200, 50, 50))
            s.blit(msg, msg.get_rect(center=(WORLD_WIDTH // 2, 80)))

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
