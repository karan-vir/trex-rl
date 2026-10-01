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
- **Reachability search: "is this game fair?"** A single bot crashing proves nothing,
  because the bot might be bad or the game might be unfair. To be sure, track every
  position the dino could still be in without having crashed:
  1. A *state* is the dino's situation at one instant: (height, vertical speed).
  2. Start with one state: standing on the ground.
  3. Each tick, try all 3 actions (nothing / jump / duck) from every state we have.
  4. Throw away any state that would hit an obstacle. Keep the survivors. Repeat.
  If the set of survivors ever becomes empty, every possible player has died, so the
  game is unfair. If it never empties, a perfect player exists. This only works
  because obstacles never react to the dino: the course is fixed by the seed, so we
  can record it once and ask "is there ANY path through?"
- **Bucketing: keeping the search fast.** Different ways of jumping (jump now or a
  few ticks later, press duck early or late) leave the dino at about 4,800 different
  (height, speed) pairs at once, which is slow to check every tick. Many are nearly
  identical, e.g. (40.2, 3.1), (40.9, 3.4), (41.3, 2.8). So we group states within
  2 px of height and 1 px/tick of speed and keep ONE real state per group, like
  keeping one coin from each pile. That cuts it to ~400 states.
- **Why bucketing is safe for a pass, but not for a fail.** We keep actual states that
  really happened, never made-up averages. So every state in the set is genuinely
  reachable with real button presses. If one survives all the way, a real winning
  sequence exists: a pass is a true proof. But discarding similar states might throw
  away the one that fits through a very tight squeeze, so a fail is only a warning.
  If a seed ever fails, re-check it with exact search (`BUCKET = (0.001, 0.001)`)
  before blaming the game.
- **Test your test.** Shrinking `MIN_GAP_TICKS` from 40 to 10 made seeds unwinnable and
  the test caught it, so the test can actually fail. 20 still passes, so 40 has margin.
- **Why this matters for RL:** if a game can be unwinnable, a bad score might be the
  game's fault, not the agent's. Now we know a perfect player exists.
