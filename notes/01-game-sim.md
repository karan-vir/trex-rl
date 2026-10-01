# Milestone 1: game sim + renderer

## Ideas worth remembering

- **Fixed timestep.** `step()` always advances exactly one tick. The game never
  looks at a real clock, so speed is decided by whoever calls `step()`:
  60/sec for a human, thousands/sec for training.
- **Logic and drawing are separate.** `game.py` has no graphics. `render.py` only
  reads the game state. That is why an AI can train with no window and we can
  still watch it later using the same renderer.
- **Determinism.** `TRexGame(seed)` uses its own `random.Random(seed)`, so the same
  seed plus the same actions gives the same game every time.
- **Fairness.** Gaps between obstacles are at least 40 ticks of travel. A jump lasts
  about 33 ticks, so every obstacle can be cleared (milestone 2 tests this).
- **Shrunken hitboxes.** Boxes are 4px smaller than the drawn shapes, so close
  calls feel fair.

## Try it

    python scripts/play.py      # SPACE/UP jump, DOWN duck, R restart, ESC quit
