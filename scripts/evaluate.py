"""Score an agent on fixed seeds (the fair comparison tool).

    python scripts/evaluate.py rule
    python scripts/evaluate.py random --episodes 200
"""

import argparse
from collections import Counter

from trex.agents.random_agent import RandomAgent
from trex.agents.rule_based import RuleBasedAgent
from trex.evaluate import evaluate

AGENTS = {"random": RandomAgent, "rule": RuleBasedAgent}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("agent", choices=AGENTS)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--first-seed", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=10_000)
    args = p.parse_args()

    seeds = range(args.first_seed, args.first_seed + args.episodes)
    result = evaluate(AGENTS[args.agent](), seeds, max_episode_steps=args.max_steps)
    print(f"{args.agent}: {result.summary()}")
    if result.deaths:
        print("  killed by:", dict(Counter(result.deaths)))


if __name__ == "__main__":
    main()
