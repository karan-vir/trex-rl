"""Play the game yourself.

SPACE/UP = jump, DOWN = duck, R = restart, F or F11 = fullscreen, ESC = quit
(ESC leaves fullscreen first).
"""

import pygame

from trex.game import DUCK, JUMP, NOOP, TRexGame
from trex.render import Renderer


def main():
    game, renderer = TRexGame(), Renderer("human")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if renderer.is_fullscreen:
                        renderer.toggle_fullscreen()
                    else:
                        running = False
                elif event.key in (pygame.K_f, pygame.K_F11):
                    renderer.toggle_fullscreen()
                elif event.key == pygame.K_r:
                    game.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            action = JUMP
        elif keys[pygame.K_DOWN]:
            action = DUCK
        else:
            action = NOOP

        game.step(action)
        renderer.draw(game, ["F: fullscreen"])
    renderer.close()


if __name__ == "__main__":
    main()
