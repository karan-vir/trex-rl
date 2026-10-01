# trex-rl

A learning project: build a T-Rex Runner simulation from scratch, expose it as a
[Gymnasium](https://gymnasium.farama.org/) environment, and train agents to play it.

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Game sim + human play mode (Pygame) | done |
| 2 | Tests: determinism, collisions, solvability | done |
| 3 | Gymnasium env + random agent | todo |
| 4 | Rule-based baseline agent | todo |
| 5 | PPO (Stable-Baselines3) | todo |
| 6 | Neuroevolution (hand-written NN + GA) | todo |
| 7 | Compare agents on fixed seeds | todo |
| 8 | Stretch: pixel observations / LLM agent | todo |

## Design rule

The simulation (`trex/game.py`) is deterministic, seedable, and knows nothing about
rendering or Gymnasium. The env and the viewer are thin layers on top.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Notes on what each milestone taught lives in [`notes/`](notes/).
