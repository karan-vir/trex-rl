"""Watch an agent play in the game window.

    python scripts/watch.py rule
    python scripts/watch.py random --seed 3

F = fullscreen, ESC = quit.
"""

import argparse

import pygame

from trex.agents.random_agent import RandomAgent
from trex.agents.rule_based import RuleBasedAgent
from trex.env import TRexEnv

AGENTS = {"random": RandomAgent, "rule": RuleBasedAgent}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("agent", choices=AGENTS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=5_000, help="stop an episode after this many ticks")
    args = p.parse_args()

    env, agent = TRexEnv(render_mode="human", max_episode_steps=args.max_steps), AGENTS[args.agent]()
    seed, running = args.seed, True
    while running:
        obs, info = env.reset(seed=seed)
        done = False
        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_f, pygame.K_F11):
                    env.toggle_fullscreen()
            obs, _, terminated, truncated, info = env.step(agent.act(obs))
            done = terminated or truncated
        print(f"seed {seed}: score {info['score']}, {info['ticks']} ticks, dodged {info['passed']}")
        seed += 1
    env.close()


if __name__ == "__main__":
    main()
