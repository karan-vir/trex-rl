"""The dumbest possible agent: press random buttons. Our baseline to beat.

    python scripts/random_agent.py              # 100 episodes, no window, prints stats
    python scripts/random_agent.py --watch      # watch it flail (Ctrl-C or close to stop)
"""

import argparse

import numpy as np

from trex.env import TRexEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--watch", action="store_true", help="render in a window")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = TRexEnv(render_mode="human" if args.watch else None)
    scores, lengths = [], []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
        scores.append(info["score"])
        lengths.append(info["ticks"])
        if args.watch:
            print(f"episode {ep}: score {info['score']}, survived {info['ticks']} ticks")
    env.close()

    print(f"\nrandom agent over {len(scores)} episodes")
    print(f"  score : mean {np.mean(scores):.0f}  median {np.median(scores):.0f}  max {max(scores)}")
    print(f"  ticks : mean {np.mean(lengths):.0f}  max {max(lengths)}")


if __name__ == "__main__":
    main()
