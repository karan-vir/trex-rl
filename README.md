# trex-rl

A learning project: build a T-Rex Runner simulation from scratch, expose it as a
[Gymnasium](https://gymnasium.farama.org/) environment, and put agents on it. It grew into testing a
hand-written agent on a real, public copy of the Chrome dino game
([chromedino.com](https://chromedino.com)) and reading what went wrong from flight-recorder traces.

**Status:** a rule-based agent reached **2nd place of all time (18,792)** on chromedino.com's leaderboard.
The learning agents (PPO, neuroevolution) are not done.

## Results on the real game

Everything below is from the **rule-based agent** (no learning) playing chromedino.com in a real browser,
Oct 2-3, 2026, on one Mac display running at 120 frames/s.

### Leaderboard progression (player name `konqueror`)

| Score | Where it landed | Script |
|---|---|---|
| 4,300 | 5th of the day | v3 |
| 6,470 | 3rd of the day | v5 |
| 9,559 | 1st of the day | v8 |
| **18,792** | **2nd of all time**, 1st of the day (behind Anonymo's 20,156) | **v9** |

### Per-version results (score = the game's displayed score; each episode is one run until it crashed)

| Script | What it did | Episodes | Median | Mean | Best | >= 5,000 | >= 10,000 |
|---|---|---|---|---|---|---|---|
| v6 and earlier | re-sent the duck key every frame (a bug) | 98 * | 892 | 1,232 | 6,593 | 2 | 0 |
| v7 | same bug, now with saved flight recorder | 10 | 804 | 1,026 | 2,361 | 0 | 0 |
| v8 | duck pressed **once** per fall | 12 | 2,755 | 3,642 | 9,559 | 4 | 0 |
| **v9** | v8 + landing handling | **40** | **2,431** | **3,622** | **18,792** | 8 | 3 |
| v10 | + fast fall after clearing an obstacle | 40 (run stopped early) | 1,298 | 2,814 | 13,567 | 8 | 4 |

\* v6 numbers are transcribed from console screenshots; v7 to v10 are from saved files, which are summarised
per episode in [`results/real_game_episodes.csv`](results/real_game_episodes.csv) (raw files with full
traces are local only, in the gitignored `runs/`).

In total roughly **225 real-game episodes** were played (102 saved with traces, 98 transcribed, and about 25
earlier ones with older versions).

**v9 is the best script so far.** It is kept in [`scripts/archive/chrome_dino_agent_v9.js`](scripts/archive/chrome_dino_agent_v9.js)
(also commit `86f404a`). The file `scripts/chrome_dino_agent.js` currently holds v10, which regressed.
v10 fixed the pterodactyl deaths (1 of 40, from 15 of 40) but it died on the first obstacle of 11 games
because the fast fall fired too early; that is understood from the traces but not fixed.

## What made the difference

1. **A bug in my own script, not the strategy.** The agent re-sent the duck key on every frame. In the
   game, every new press resets the fall speed, so ducking in mid-air made the dino fall *slower* than doing
   nothing (1.5 vs 2.7 px/frame, measured). Pressing once raised the median from about 800 to about 2,700.
2. **The simulator was too easy until it matched the real game.** Obstacle groups, width-scaled gaps, the
   real jump arc (2.8 px mean error), and a measured 4-14% slowdown of obstacles versus the reported speed
   (a whole-pixel rounding effect on a high-refresh display) were each found by measuring the real game.
3. **Trusting a number the game reports was wrong.** The script now measures obstacle speed itself.
4. **Reading flight-recorder traces beat guessing.** Several explanations I proposed first were wrong and
   were corrected by the next set of traces (all recorded in [`notes/04-rule-based-agent.md`](notes/04-rule-based-agent.md)).

## Limitations

**What this is not**
- It is **not a learning agent.** The rules were written and tuned by hand. The project's original goal
  (PPO and neuroevolution, milestones 5-8) is untouched; the environment and baseline for them exist.
- A rule-based agent reading the game's internal state is a different problem from learning from pixels.

**How much the numbers can be trusted**
- **High variance, modest samples.** Scores in one version range from about 40 to 18,792. Each version has
  only 10 to 40 saved episodes. The headline 18,792 is the best of roughly 100 recent attempts, so treat the
  **median (about 2,400) as the typical result**. Across v8 to v10, about 1 attempt in 5 got above 5,000
  (20 of 92) and 1 in 13 above 10,000 (7 of 92).
- **v8 and v9 are not statistically separable** (medians 2,755 and 2,431 with 12 and 40 episodes). v9 leads
  on its best score and on sample size, not on a clear difference in typical results. v10's tail is no worse
  than v9's (4 vs 3 episodes above 10,000); its regression is the early-game deaths.
- **One machine, one display.** All results are from a 120 Hz screen. The obstacle slowdown the script
  corrects for depends on the refresh rate, and results on a 60 Hz display, another browser, or another
  computer have not been tested.
- **The simulator is milder than the real game.** It decides once per tick (the real agent acts about 120
  times a second) and has no slow-landing state. Simulator results helped explain *why* things failed but
  did not predict real scores.
- Several causes are **inferred from traces, not confirmed** against the game's code (for example the
  "stuck at ground level after an exact landing" state, which was reproduced in the browser but whose
  mechanism is my explanation).

**Fragility and ethics**
- It depends on **one site's internals** (`Runner.instance_`, field names, its score-posting code) and on the
  classic Chromium dino code, not `chrome://dino` or current Chromium. A site change would break it.
- **Leaderboards move.** The "of the day" board resets daily (a 6,593 run was missed because the script was
  still reading the previous day's board), and the all-time board changed during the project (a 24,230 entry disappeared).
- **Entries were bot-played but posted under a human-looking name** (`konqueror`) with no label. The script
  blocks all score posts by default and only posts a score above a chosen rank, but a public board is shared
  with human players, so consider that before using it.
- The record (20,156) needs about 11 minutes of play at top speed without one mistake, so it is out of
  reach at the current failure rate.

**Still failing**
- Cacti caused 25 of v9's 40 deaths, mostly groups of three large cacti. Most of them look like the dino
  landing too late to jump for the *next* obstacle (inferred from traces). The attempted fix (v10)
  introduced a worse early-game bug.

## Repository

| Path | What |
|---|---|
| `trex/game.py` | the game: deterministic, seedable, no rendering. `chrome_like=True` mimics the real game's obstacle rules, jump and fast fall |
| `trex/env.py` | Gymnasium environment (12-number observation, 3 actions) |
| `trex/render.py`, `trex/sprites.py` | Pygame viewer with pixel-art sprites, fullscreen |
| `trex/agents/` | random and rule-based agents |
| `trex/evaluate.py` | fair comparison on fixed seeds |
| `scripts/play.py`, `watch.py`, `evaluate.py` | play yourself, watch an agent, score an agent |
| `scripts/chrome_dino_agent.js` | the agent for the real game (runs in the browser console; see its header) |
| `scripts/archive/chrome_dino_agent_v9.js` | v9, the best version so far |
| `tests/` | 71 tests (physics, determinism, solvability, Gymnasium API, chrome-like mode) |
| `results/real_game_episodes.csv` | one row per saved real-game episode (v7-v10) |
| `notes/` | what each stage taught, including the wrong turns |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                      # about a minute
python scripts/play.py      # play it yourself: space/up jump, down duck, F fullscreen
python scripts/evaluate.py rule
```

To try the agent on the real game: open the dino game page, paste `scripts/archive/chrome_dino_agent_v9.js`
into the DevTools console, and run `dinoAgent.start({ episodes: 3, maxScore: 1500 })`. It blocks score
submissions unless you pass `submit: 'top1'` .. `'top5'`.

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Game sim + human play mode (Pygame) | done |
| 2 | Tests: determinism, collisions, solvability | done |
| 3 | Gymnasium env + random agent | done |
| 4 | Rule-based baseline agent (+ real-game test on chromedino.com) | done |
| 5 | PPO (Stable-Baselines3) | todo |
| 6 | Neuroevolution (hand-written NN + GA) | todo |
| 7 | Compare agents on fixed seeds | todo |
| 8 | Stretch: pixel observations / LLM agent | todo |
