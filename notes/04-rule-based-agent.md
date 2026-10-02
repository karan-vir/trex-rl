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

(Done, see the next section.)

## Fixing it: what the real cause turned out to be

My first diagnosis (wide groups need a later jump) was only half right. The full story:

1. **The jump itself is fine.** I measured it on chromedino.com: apex 87 px at tick ~14.6,
   in the air ~34 ticks, above a 50 px cactus from tick 5.6 to 27.6. The jump is capped
   (about 85 px at any speed), so speed barely changes it.
2. **Obstacles move slower than the reported speed.** The browser runs at 121 fps; the game
   moves objects by `floor(speed x step)` whole pixels per frame, so they lose a fraction
   every frame. Measured over 2,484 frames: actual / reported = **0.871 at speed 6-7,
   0.858 at speed 7-8**. The agent trusted the reported speed, so it believed obstacles
   arrive ~14% sooner than they do, jumped ~4 ticks early, and came down on the far
   end of wide groups.
3. **This is a property of the screen, not the game design.** On a 60 Hz display the same
   rounding loses less. Our numbers are for a 120 Hz pane.

### Reproducing it in the simulator

`TRexGame(chrome_like=True, motion_scale=0.87)` moves obstacles at 0.87x while still
reporting the full speed. 120 seeds each, chrome-like generator:

| agent | survived | median crash score | killed by |
|---|---|---|---|
| fixed lead 12, obstacles at full speed | 113/120 | 1270 | few large cacti |
| **fixed lead 12, obstacles at 0.87x** | **0/120** | **225** | **120 large cacti** |
| width-aware, trusts reported speed | 120/120 | | |
| width-aware, uses the real speed | 120/120 | | |

Row 2 matches the real result for the old rule (median crash score ~170, large groups).
So the diagnosis is confirmed independently, not just argued.

### The fix in the Chrome script

`scripts/chrome_dino_agent.js` now (a) jumps later for wider obstacles, with the arc center
taken from the measured jump, and (b) **measures** obstacle speed from frame to frame
instead of trusting `currentSpeed`.

**Real result on chromedino.com: 6 of 6 episodes survived to the score-400 cap** (speed ~10).
The old rule: 1 of 6 in the same setup, 2 of 10 overall. Caveats: the cap is low, the
sample is 6, and the real game gets harder (speed 13) after score ~600 which this run did not reach.

### Lessons

- The simulator can be wrong in ways you can only find by comparing against the real thing.
  Here a single hidden number (0.87) separated "perfect" from "fails every time".
- When the cause is uncertain, measure it directly (the arc, the speed) instead of guessing.
- An agent that measures what it needs is more robust than one given a fixed assumption.
  A learning agent trained in the real environment would learn this offset for itself.
- Bug found in my own script again: on a freshly loaded page the game loop already ticks,
  so "loop is running" is not "dino is playing". Now also requires distanceRan > 0.

### The leaderboard

chromedino.com posts every finished game to `/inc/set.php` (name from a cookie, score,
duration, obstacles passed, jump counts), automatically and unconditionally. Setting a
nickname registers it on their server (`/inc/nick.php`, with "already taken" handling). All my
runs blocked outgoing POSTs, so nothing has been submitted.

### Leaderboard gate (added to scripts/chrome_dino_agent.js)

The built-in browser pane cannot show `window.prompt()`, and the site's Nickname button
uses it, so a nickname cannot be claimed there (no network request is ever made). So the
real run happens in the person's own Chrome, and the script protects the leaderboard:

- Score posts to `/inc/set.php` are **blocked by default** (XHR, fetch and sendBeacon).
- `dinoAgent.start({ submitAt: 4300, episodes: 5 })` plays until a run reaches 4300, stops,
  and shows a `confirm()` dialog with the exact score and the nickname. OK lets that one post
  through; Cancel discards it. A crash below the target is never posted.
- The site only submits a score that beats the session's best, and ignores very low scores.

Tested in the pane with a stub underneath the gate that swallowed every POST, so nothing real
was sent: Cancel -> blocked; OK -> exactly one post passes (body `name=...&score=...&t=...&o=...`);
default run -> a new best is blocked.

The score post carries duration, obstacles passed and jump counts, so the site can sanity-check
that a score matches real play. A genuine run of the bot produces consistent numbers.

## First run in the user's own Chrome (5 attempts, target 4300, no cap)

| # | score | speed | killer | what the log says |
|---|---|---|---|---|
| 1 | 1108 | 13 | mid pterodactyl | `action DUCK`, still airborne at dinoY 0 (landing), ptero 21 px behind the nose |
| 2 | 288 | 9.06 | mid pterodactyl | `DUCK` in the air at dinoY 38, ptero 19 px behind the nose |
| 3 | 288 | 9.06 | (same as 2) | identical in every field except frames: 12842 vs 4138 (about the sum of attempts 1+2) |
| 4 | **2992** | 13 | 3x large cactus | `NOOP` at dinoY 40, dist -23: came down on the group |
| 5 | 96 | 7.17 | 3x large cactus | `NOOP` at dinoY 19 (still rising), **measured speed 0** |

Best 2992. No attempt reached 4300, so no confirmation dialog appeared and nothing was
submitted. (The score of 2992 was attempt 4, not the first.)

### What is clearly wrong (my script)

- **The speed estimator was corrupted.** It kept sampling while the game was frozen (crash
  screen, hidden tab), so obstacle movement read as 0. Attempt 5 shows `measuredSpeed 0`;
  with a speed of 0 the clamp falls back to half the real speed, the agent thinks obstacles
  are twice as far in time, and jumps too late (dinoY 19 and rising when it hit).
- **Attempt 3 is probably a duplicate** of 2: a freeze or hidden tab that the script logged
  as a second crash. Pausing happens when a tab is hidden: the game stops and the script's
  frame loop stops with it, and on return my code could not tell a pause from a freeze.

### What is NOT explained

Attempts 1-3 (3 of 5) were pterodactyl hits while the dino was still in the air or landing
after jumping for something else. My simulator, even with the pterodactyl speed offset
(+/-0.8) and the measured slowdown, survives 118/120 long runs and never dies to a
pterodactyl, so the simulator is still missing something. Candidates, none verified:
the real game's faster speed ramp on a 120 Hz screen, different collision boxes, or the
gap between a jump and the next obstacle. I will not guess; the script now records evidence.

### Changes made (not yet verified in a real run)

- Speed is only sampled while the game is running, averaged over a window, and taken from the
  nearest obstacle's own movement first (pterodactyls move at speed +/- 0.8).
- A frame gap over 0.5 s (hidden tab) resets the estimator and the freeze timer.
- Freeze detection: distance not changing for 2.5 s while "playing" is logged as `stuck`.
- **Flight recorder:** each crash/stuck result carries `trace`, the last ~0.75 s of frames
  (dino height, air/duck state, action, first two obstacles with distance and type, speed).
  Print crash 0 with `dinoAgent.trace(0)`.

## First leaderboard entry, and why the script no longer stops at the target

Attempt 6 of the v3 run reached the target of 4300, the script stopped there and showed the
confirmation dialog, and the person pressed OK. **konqueror, 4300, 5th place of the day**
(verified on chromedino.com: 5 highest of the day were Anonym 24230, cesar 13004, joshua D
5548, win 5322, konqueror 4300; the site's own lookup `get_best&name=konqueror` returned 4300).

That also showed a design mistake of mine: stopping at the target froze the score at exactly
the threshold, ending a run that might have gone on. Fixed in v4 (`v4-play-on`):

- The agent keeps playing past `submitAt`. The confirmation happens when the site actually posts,
  at game over, using the score inside the request (`score=...`).
- Below the target: dropped silently, no dialog. At or above: dialog with the final score.
  After an answer it stops (`stopAfterSubmit: false` keeps going). Optional ceiling `stopAt`.
- Tested in the pane with a stub under the gate (nothing real sent): target 60 / ceiling 100
  -> asked about 100 not 60; Cancel -> blocked; OK -> one post of 150; target 5000 / end 200 ->
  silent block, no dialog.

### v3 run: the other five attempts

| # | score | killer |
|---|---|---|
| 1 | 223 | 3x cactus group, but dinoY 88 at the last recorded frame (see below) |
| 2 | 801 | mid pterodactyl, `DUCK`, airborne at 39 |
| 3 | 611 | 3x cactus group, `NOOP`, dinoY 11 |
| 4 | 480 | mid pterodactyl, `DUCK`, airborne at 24 |
| 5 | 1898 | mid pterodactyl, `DUCK`, landing (0) |
| 6 | **4300** | submitted |

Over the 15 attempts so far (old script and v3), mid-height pterodactyls killed about 10.

**trace(0), read honestly.** The jump was timed correctly: take-off about 75 px before a 75 px
wide group, an arc to 88 px, and the trace ends with the dino at 88 px above the group. A 50 px
cactus cannot hit a dino at 88 px, so the crash happened after the last recorded frame. Frames
are missing, which points to the page stalling (the loop did not run for a while) and the game
landing the dino on the cactus once it resumed. Cause not identified.

Traces of the pterodactyl crashes have not been looked at yet; that is the next evidence to get.

## v5: no fixed target, auto-submit above 5th place (`submit: 'top5'`)

`dinoAgent.start({ episodes: 30, submit: 'top5' })` plays 30 episodes straight. At game over,
if the score is above the CURRENT 5th place of the day (read fresh from the page's
`.high-scores` list before every run), it is posted automatically, with no dialog, under the
nickname set with the site's own Nickname button. Otherwise it is never posted. It refuses to
start without a nickname so nothing is posted as Anonym, and it prints a line for every
decision (`SUBMITTING n ...` / `not submitting n ...`).

How the site decides to post (read from its game.js): it only posts a score that beats the
personal best of the nickname (`score > personalBest`), after a read-only GET to
`/inc/check.php?score=N` whose text it shows in an `alert()`. So for a name whose best is 4300,
only scores above 4300 are ever posted. The script logs those site alerts to the console instead
of letting a modal popup freeze the run.

Tested in the pane with a stub under the gate (nothing real sent) and a fake leaderboard:
parser read the real 5th place (4300); no-nickname guard refused; 5th=80 -> a run ending at 100
posted automatically with no dialog and the run continued; 5th=5000 -> not posted.

Bug fixed on the way: decisions from an earlier start() leaked into the next run's result
(`decisionIdx` now starts at the current length).

## Result of the v5 auto-submit run

Verified on chromedino.com afterwards (read-only): **konqueror 6470, 3rd of the day**, above
joshua D 5548 and win 5322, below cesar 13004 and Anonym 24230 (`get_best` for konqueror: 6470).
It took about 3 to 4 attempts. It does not enter the all-time top 5 (5th place there is 13004).

Progress of the rule-based agent on the real game:

| version | change | typical result |
|---|---|---|
| v1 | fixed jump lead | 2 of 10 episodes survive past score 400; dies by ~170 |
| v2 | width-aware jump + measured speed | 6 of 6 survive to 400; median ~1470 over 10 attempts, best 3435 |
| v5 | no target, auto-submit above 5th place | 6470 within 3-4 attempts |

Remaining weakness: mid-height pterodactyls hit while the dino is still airborne (about 10 of 15
logged crashes). Not yet explained or fixed; the flight recorder exists to find out.

## v6: choose the place to aim for (`submit: 'top1'` ... `'top5'`)

`dinoAgent.start({ episodes: 100, submit: 'top2' })` posts a finished run only if its score is
above the CURRENT score of that place of the day (re-read before every run). Verified in the
pane against the real board (ranks 1-5 read as 24230, 13004, 6470, 5548, 5322), a bad rank is
rejected, and a fake board with 2nd place at 80 auto-posted a run that ended at 100.

### What top 2 asks for (arithmetic, not a promise)

2nd place today is 13004, so the run must end above 13004. Score is distance x 0.025, so that is
about 520,000 px. At the top speed (13 px per tick at 60 ticks/s) that is ~40,000 ticks, about
11 minutes without one mistake, passing on the order of 1,000 obstacles at short gaps. The best
real run so far is 6470 and the median is ~1,500, so doubling the best needs a much lower
per-obstacle failure rate than the rule has now. Expect many attempts, and the pterodactyl fix
is what would move the odds most.
