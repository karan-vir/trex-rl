/*
 * Rule-based agent for the REAL Chrome dino game (chrome://dino).
 * A port of trex/agents/rule_based.py - keep the two in sync.
 *
 * HOW TO RUN
 *   1. In Chrome open chrome://dino   (or go offline and open any page).
 *   2. Open DevTools (Cmd+Option+J), go to the Console tab.
 *   3. Paste this whole file and press Enter.
 *   4. Run:   dinoAgent.start({ episodes: 3, maxScore: 1500 })
 *      Keep the tab visible and in front: Chrome pauses hidden tabs.
 *   5. Stop any time with dinoAgent.stop().  Results: dinoAgent.results
 *
 * It presses keys the same way a person would (dispatches keydown/keyup events);
 * it only READS the game's state to decide, using the same facts our Python agent sees.
 */
(() => {
  const LEAD_TICKS = 12;                       // same default as the Python agent
  const KEY = { JUMP: 32, DUCK: 40, RESTART: 13 };

  // ---- The decision rule (pure function; identical logic to rule_based.py) ----------
  // obs: { hasObstacle, speed (px/tick), dist (px from dino's nose), flyY (px off the
  //        ground), onGround }
  function decide(obs, lead = LEAD_TICKS) {
    if (!obs.hasObstacle) return 'NOOP';
    const arrivesIn = obs.dist / obs.speed;               // distance -> time
    if (obs.flyY >= 45) return 'NOOP';                    // high pterodactyl: flies over us
    if (obs.flyY >= 20) return arrivesIn < lead ? 'DUCK' : 'NOOP';   // mid: duck under
    if (obs.onGround && arrivesIn >= 0 && arrivesIn < lead) return 'JUMP';  // cactus / low
    return 'NOOP';
  }

  // ---- Read the game's state and translate it into the same terms as our sim --------
  function observe(r) {
    const t = r.tRex;
    const noseX = t.xPos + t.config.WIDTH;
    const groundBottom = t.groundYPos + t.config.HEIGHT;   // y of the ground line
    const o = r.horizon.obstacles.find((ob) => ob.xPos + ob.width > t.xPos);
    const base = { speed: r.currentSpeed, onGround: !t.jumping, dinoY: groundBottom - t.config.HEIGHT - t.yPos };
    if (!o) return { ...base, hasObstacle: false };
    return {
      ...base,
      hasObstacle: true,
      dist: o.xPos - noseX,
      flyY: groundBottom - (o.yPos + o.typeConfig.height),  // height of its underside
      type: o.typeConfig.type,
      size: o.size,                                         // cacti in the cluster
      width: o.width,
    };
  }

  // ---- Keyboard ------------------------------------------------------------------
  function key(type, code) {
    const e = new Event(type, { bubbles: true, cancelable: true });
    e.keyCode = code;
    e.which = code;
    document.dispatchEvent(e);
  }

  const S = { running: false, results: [], jumpDown: false, jumpFrame: 0, duckDown: false };

  function apply(action, r, frame) {
    const t = r.tRex;
    // Jump: hold the key until we land (letting go early cuts a jump short in Chrome).
    if (action === 'JUMP' && !S.jumpDown) {
      key('keydown', KEY.JUMP);
      S.jumpDown = true;
      S.jumpFrame = frame;
    } else if (S.jumpDown && !t.jumping && frame - S.jumpFrame >= 2) {
      key('keyup', KEY.JUMP);
      S.jumpDown = false;
    }
    // Duck: on the ground it crouches; in the air it makes you fall faster.
    if (action === 'DUCK') {
      if (t.jumping || !t.ducking) key('keydown', KEY.DUCK);
      S.duckDown = true;
    } else if (S.duckDown) {
      key('keyup', KEY.DUCK);
      S.duckDown = false;
    }
  }

  function releaseKeys() {
    if (S.jumpDown) key('keyup', KEY.JUMP);
    if (S.duckDown) key('keyup', KEY.DUCK);
    S.jumpDown = S.duckDown = false;
  }

  // chrome://dino has r.playing. Older copies (e.g. chromedino.com) don't. There `activated`
  // is not reset after a restart and `paused` stays true while the game runs, so neither
  // can be trusted; ask whether the game loop is actually running and not crashed.
  const isPlaying = (r) =>
    r.playing !== undefined ? r.playing : !!(r.isRunning && r.isRunning() && !r.crashed);

  const score = (r) => Math.ceil(r.distanceMeter.getActualDistance(Math.ceil(r.distanceRan)));

  // ---- Episode loop --------------------------------------------------------------
  function start({ episodes = 3, maxScore = 1500, lead = LEAD_TICKS } = {}) {
    const Runner = window.Runner;
    if (!Runner || !Runner.instance_) throw new Error('No Runner.instance_ here: open chrome://dino');
    S.running = true;
    S.results = [];
    let frame = 0, lastObs = null, lastAction = 'NOOP', restartAt = 0, lastKick = 0;
    console.log(`[dinoAgent] ${episodes} episodes, stop at score ${maxScore}, lead ${lead} ticks`);

    function finish(outcome) {
      const r = Runner.instance_;
      const res = {
        episode: S.results.length + 1, outcome, score: score(r), frames: frame,
        speed: +r.currentSpeed.toFixed(2), distance: Math.round(r.distanceRan),
      };
      if (outcome === 'crash' && lastObs) {
        Object.assign(res, {
          killer: lastObs.type, cluster: lastObs.size, killerWidth: lastObs.width,
          distAtLastFrame: Math.round(lastObs.dist), flyY: Math.round(lastObs.flyY),
          action: lastAction, wasOnGround: lastObs.onGround, dinoY: Math.round(lastObs.dinoY),
        });
      }
      S.results.push(res);
      console.log('[dinoAgent]', JSON.stringify(res));
      releaseKeys();
      frame = 0; lastObs = null; lastAction = 'NOOP';
    }

    function tick() {
      if (!S.running) return;
      const r = Runner.instance_;
      const now = performance.now();

      if (r.crashed) {
        if (!restartAt) {                                  // just crashed
          if (frame > 0) finish('crash');                  // ignore crashes we weren't playing
          if (S.results.length >= episodes) return done();
          restartAt = now + 1300;
        } else if (now >= restartAt) {
          key('keydown', KEY.RESTART); key('keyup', KEY.RESTART);   // Enter restarts
          restartAt = 0;
        }
      } else if (!isPlaying(r)) {
        if (now - lastKick > 1000) {                       // press jump to start the game
          key('keydown', KEY.JUMP); key('keyup', KEY.JUMP); lastKick = now;
        }
      } else {
        frame++;
        lastObs = observe(r);
        lastAction = decide(lastObs, lead);
        apply(lastAction, r, frame);
        if (score(r) >= maxScore) {                        // survived long enough
          finish('survived');
          if (S.results.length >= episodes) { r.gameOver(); return done(); }
          r.gameOver(); restartAt = now + 1300;
        }
      }
      requestAnimationFrame(tick);
    }

    function done() {
      S.running = false;
      const crashed = S.results.filter((x) => x.outcome === 'crash').length;
      console.log(`[dinoAgent] done: ${S.results.length - crashed} survived, ${crashed} crashed`);
      console.table(S.results);
    }
    requestAnimationFrame(tick);
  }

  function stop() { S.running = false; releaseKeys(); console.log('[dinoAgent] stopped'); }

  window.dinoAgent = { start, stop, decide, observe, get results() { return S.results; } };
  console.log('[dinoAgent] ready. Run: dinoAgent.start({ episodes: 3, maxScore: 1500 })');
})();
