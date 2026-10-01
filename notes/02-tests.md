# Milestone 2: tests

Run them with `pytest` (about 35 seconds, nearly all of it the solvability search).

## What is tested

- **Repeatability** - same seed + same actions gives an identical game; different
  seeds give different obstacle courses.
- **Physics** - jump apex ~83 px, airtime ~33 ticks, no double jump, duck-in-the-air
  falls faster, ducking only counts on the ground.
- **Collisions** - hits end the game, near-misses inside the shrunken hitbox don't,
  jumping clears a cactus, ducking clears the mid-height pterodactyl, a high one
  passes over a standing dino, and a dead game stays frozen.
- **Spawning and difficulty** - no pterodactyls early, all three heights later,
  gap between obstacles >= a jump, speed ramps then caps, score only increases.
- **Solvability** - for every seed tested, some sequence of actions survives the
  whole run (including at maximum speed).

## Ideas worth remembering

- **Refactor so you can test.** The dino's physics became a pure function,
  `dino_physics(y, vy, action) -> (y, vy, ducking)`. Pure functions (no hidden state)
  are easy to test and to reuse. A planning agent could use it later too.
  I recorded game traces before the refactor and checked they matched after.
- **Obstacles never react to the dino.** For a seed, the course is fixed, so you can
  record it once and ask "is there ANY path through?" without playing.
- **Reachability search.** Keep the set of every dino state that is still alive at
  each tick, try all 3 actions from each. If the set ever empties, the game is unfair.
  The exact set grows to ~5,000 states, so we bucket similar states (2 px, 1 px/tick).
  Each kept state is a real reachable one, so a pass is a genuine proof.
- **Test your test.** Shrinking `MIN_GAP_TICKS` from 40 to 10 made seeds unwinnable and
  the test caught it, so the test can actually fail. 20 still passes, so 40 has margin.
- **Why this matters for RL:** if a game can be unwinnable, a bad score might be the
  game's fault, not the agent's. Now we know a perfect player exists.
