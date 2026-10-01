# Milestone 4: rule-based agent and the evaluation tool

## The agent (`trex/agents/rule_based.py`)

A hand-written player. It sees only the same 12 numbers a learning agent sees.

- Cactus or low pterodactyl: jump when it will arrive within `lead_ticks` ticks.
- Mid-height pterodactyl: duck while it is within `lead_ticks`.
- High pterodactyl, or nothing ahead: do nothing.
- Never jump again while already in the air.

The key step is turning distance (px) into time (ticks) by dividing by speed.

## Result

On 100 fixed seeds, 30,000 ticks each: **100 / 100 survive**. Random: 0 / 100.

Score depends only on time survived, so every perfect run gets the identical score
(26,375 for 10,000 ticks). That number is the ceiling for a 10,000-tick episode.

## Ideas worth remembering

- **Tuning a parameter with data.** I guessed `lead_ticks = 14`, then swept it:
  6 to 18 survive every seed, 4 fails (too late) and 22+ fails (lands on the cactus).
  There is a working window and a cliff on both sides. I then picked 12, the middle,
  so small errors in the exact timing don't matter.
  Learning agents face the same thing: they have to find that window by trial and error.
- **Honest correction about "time vs pixels".** I expected a fixed pixel distance to
  fail, since the right jump distance changes with speed. It doesn't fail outright:
  80 to 100 px survives every seed, 60 and 120 fail. But that window is much narrower
  than the time-based one (about +-20 px, versus +-6 ticks, which is +-36 px at speed
  6 and +-78 px at speed 13). So converting to time is more forgiving, not strictly required.
- **A saturated benchmark is a problem.** A baseline that is already perfect leaves
  a learner nothing to beat; the best it can do is match it. So at default difficulty
  we can only judge learners on how fast they learn and how reliably, not on score.
- **The difficulty dial.** Shrinking the minimum gap between obstacles (`MIN_GAP_TICKS`):

  | gap (ticks) | perfect player exists? | rule agent survives |
  |---|---|---|
  | 40 (current) | yes | 40/40 |
  | 30 | yes | 40/40 |
  | 26 | yes | 37/40 |
  | 22 | yes | 0/40 |
  | 20 | yes | 0/40 |
  | 16 | yes | 0/40 |

  (solvability checked on 6 seeds x 3,000 ticks with the milestone 2 search.)
  At gap 22 and below the hand-written rule fails every time while winning is still
  possible, so a learner could genuinely beat it there. Likely trick it would need:
  press duck in mid-air to fall faster and land in time for the next jump.

## Tools added

- `trex/evaluate.py` and `scripts/evaluate.py`: score any agent on fixed seeds.
- `scripts/watch.py`: watch an agent play in the window (F = fullscreen).

    python scripts/evaluate.py rule
    python scripts/watch.py rule

## Testing the rule on the REAL Chrome dino game

`scripts/chrome_dino_agent.js` is a line-for-line port of the rule, run inside
chrome://dino. It presses keys like a person (keydown/keyup events) and only reads
the game's state to decide.

### How the real game's numbers map onto our 12-number view

| What the agent needs | Our sim | Chrome's game |
|---|---|---|
| speed (px per tick) | `game.speed` | `Runner.instance_.currentSpeed` |
| distance to obstacle | `o.x - (DINO_X + 44)` | `o.xPos - (tRex.xPos + 44)` |
| obstacle height off ground | `o.y` (0 / 30 / 60) | ground line minus the obstacle's underside: 0 / 25 / 50 |
| on the ground? | `dino_y <= 0` | `!tRex.jumping` |

The three pterodactyl heights in the real game fall into the same three categories
(jump / duck / ignore), which is why the thresholds 20 and 45 still work.

### What is verified, and what is not

Verified (by tests in the browser pane, not in Chrome itself):
- JS and Python make identical decisions on 570 hand-picked situations (0 mismatches).
- The observation mapping gives the expected numbers for a mid-height pterodactyl.
- The script runs end to end against a mock of `Runner`: starts the game, plays,
  survives, records a crash with its cause, and restarts. It holds the jump key until
  landing (no early releases during play).

NOT verified: anything about the real game. The mock is my own stand-in, so passing it
says the plumbing works, not that the dino survives in Chrome. The internal names
(`tRex.groundYPos`, `horizon.obstacles`, ...) come from my memory of Chrome's source and
may differ in the current version; if the script throws, that is the first place to look.

### Predictions (written down BEFORE running, to compare against the real result)

1. Single cacti: fine, as in our sim.
2. **Groups of 2-3 cacti** (up to 75 px wide, we only had single ones): the window of
   good jump timing shrinks, so it may fail at low speed on triple large cacti.
3. Jumps in Chrome get higher at higher speeds (`-10 - speed/10`), unlike our fixed
   jump, which should make things easier, not harder.
4. Pterodactyls only appear after speed reaches ~8.5, so they show up later.
5. Overall: expect it to work for a while, then die to something our sim never showed it.
   That gap, between the simulator we trained on and the real thing, is called the
   sim-to-real gap. Closing it is a real research problem.

### To run it

1. Open chrome://dino, DevTools (Cmd+Option+J), Console tab.
2. Paste the script, then `dinoAgent.start({ episodes: 3, maxScore: 1500 })`.
3. Keep the tab in front. Results appear in the console and in `dinoAgent.results`.

## Result: the rule on chromedino.com (built-in browser, 10 agent-controlled episodes)

chromedino.com runs the classic Chromium dino code (same 600x150 world, same speeds and
gravity). Episodes were capped at score 400 or 800.

| # | outcome | score | speed | what happened |
|---|---|---|---|---|
| A1 | crash | 116 | 7.4 | landed on the tail of a 3x large cactus group (dist -90) |
| B1 | survived | 800 | 12.8 | |
| B2 | crash | 105 | 7.3 | landed on the tail of a 3x large group (dist -80, height 47) |
| B3 | crash | 159 | 7.8 | landed on the tail of a 3x large group (dist -94, height 46) |
| C1 | survived | 400 | 10.0 | |
| C2 | crash | 252 | 8.7 | 3x small group, dino at height 86 (cause NOT identified) |
| C3 | crash | 227 | 8.5 | 3x large group, dino on the ground inside it (jumped too late) |
| C4 | crash | 171 | 8.0 | landed on the tail of a 3x large group (dist -92, height 47) |
| C5 | crash | 266 | 8.9 | mid-height pterodactyl, dino airborne at height 43 |
| C6 | crash | 120 | 7.4 | landed on the tail of a 3x large group (dist -84, height 46) |

**2 of 10 survived; 8 crashed at scores 105 to 266.** In our simulator the same rule
survived 100 of 100.

### What the data says

- **6 of the 8 crashes involve a group of 3 large cacti (75 px wide).** In 5 of them the
  dino came down onto the far end of the group while still over it. For C4 and C6 I
  recorded every obstacle on screen at the crash: only that cactus group was there, and
  the dino was descending through 54, 51, 47 px (large cactus: 50 px).
- **Why:** the rule jumps a fixed 12 ticks before the obstacle's front edge arrives.
  A jump only stays above a 50 px cactus for about ticks 5 to 24 after takeoff. A single
  cactus fits easily; a 75 px group needs the dino to stay above it for ~15 more ticks, so
  the jump has to start LATER. With the arc about 19 ticks long and the obstacle needing
  (75 + 36) / 8 = ~14 ticks of cover, jump lead must be roughly 6 to 10, not 12.
  (This is my own calculation from the traces, not yet tested as a fix.)
- **Our simulator only had single cacti**, so the fixed lead looked perfect there.
  This is the sim-to-real gap, and it matches prediction 2 from above.
- C2, C3 and C5 are not explained; they are one case each.

### Predictions, scored

1. Single cacti fine: plausible (both survivors passed many), not isolated.
2. Groups break it: **confirmed**.
3. Higher real jumps help: wrong or untested. The real game caps jump height
   (apex ~88 px here) and the arc is shorter than I assumed.
4. Pterodactyls appear later: consistent (the only pterodactyl crash was at speed 8.9).
5. Works for a while then dies: confirmed.

### Bugs found in my own script along the way (fixed in the repo)

- This site never sets `activated` back to true after a restart, and leaves `paused` true
  while the game runs. My "is it playing?" check used both, so the agent never took
  control and the dino ran into the first obstacle; those runs were logged as crashes.
  Fix: use `isRunning() && !crashed`, and don't record a crash if the agent never played a frame.
- Lesson: when a result looks wrong (0 frames, a score that doesn't match), check the
  measuring tool before blaming the agent.

### Other things noticed

- The game's speed rises per frame, not per second. This display's animation loop runs at
  ~110 to 120 Hz, so speed 12.8 was reached in about a minute instead of about two.
- The page tries to POST scores and analytics. I blocked all non-GET requests on the page
  before running (106 blocked), so no scores were submitted to the public leaderboard.

### Next

Add cactus groups (1 to 3) to our simulator, make the rule width-aware (jump later for
wider obstacles), check it in the sim, then rerun on chromedino.com. This also matters for
the learning agents: anything trained only on single cacti will inherit the same blind spot.
