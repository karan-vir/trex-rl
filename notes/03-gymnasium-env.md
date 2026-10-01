# Milestone 3: the Gymnasium environment

An environment is the contract between the game and any learning agent:

    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

The agent never touches the game directly. It only sees `obs`, chooses an `action`,
and receives a `reward`. That is the whole interface of reinforcement learning.

## What the agent sees (12 numbers, in `OBS_NAMES`)

dino height, dino vertical speed, game speed, is-ducking, then for each of the next
2 obstacles: distance to the dino's nose, width, height, flying height.
Everything is scaled to roughly -1..1 because neural networks learn much more easily
from small, similar-sized numbers than from raw pixels. "No obstacle" is shown as
"far away and zero-sized" (distance 1.5) so the vector never has a hole.

## What it can do

`Discrete(3)`: 0 = nothing, 1 = jump, 2 = duck.

## Reward (adjustable in the constructor)

+1 for every tick survived, -100 for dying, optional bonus per obstacle dodged.
This is the first thing we get to experiment with in milestone 5. The reward is the
ONLY thing telling the agent what we want, so a badly chosen reward trains
the wrong behaviour very efficiently.

## terminated vs truncated

- `terminated`: the dino crashed. A real end of the game.
- `truncated`: we stopped it at 10,000 ticks while it was still alive.
The difference matters for learning: a crash means "future reward is zero", but
a time cutoff doesn't mean the future was worthless.

## Other things worth remembering

- `reset(seed=7)` always gives the same obstacle course. The game's own seed is drawn
  from Gymnasium's random generator, so evaluation on fixed seeds is reproducible.
- `check_env` is Gymnasium's official checker. It passes, which means PPO and other
  libraries will accept the env without surprises.
- `gym.make("TRex-v0")` works after `import trex` (registered in `trex/__init__.py`).
- Rendering is lazy and optional: pygame is only loaded if you ask for a render mode,
  so training runs headless. Measured speed: ~290,000 steps/sec, about 5,000x a human.

## Baseline: the random agent

`python scripts/random_agent.py` over 200 episodes: mean score ~206, max 526.
Nearly every episode ends around tick 120, which is when the first cactus arrives.
Random button-mashing almost never clears obstacle one. So ~200 is the floor, and
any agent that learns anything should beat it easily.
