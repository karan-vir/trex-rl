"""Play the game yourself.  SPACE/UP = jump, DOWN = duck, R = restart, ESC = quit."""

import pygame

from trex.game import DUCK, JUMP, NOOP, TRexGame
from trex.render import Renderer


def main():
    game, renderer = TRexGame(), Renderer("human")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                game.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            action = JUMP
        elif keys[pygame.K_DOWN]:
            action = DUCK
        else:
            action = NOOP

        game.step(action)
        renderer.draw(game)
    renderer.close()


if __name__ == "__main__":
    main()
