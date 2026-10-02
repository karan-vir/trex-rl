/*
 * Rule-based agent for the REAL Chrome dino game (chrome://dino, chromedino.com).
 * A port of trex/agents/rule_based.py - keep the two in sync.
 *
 * HOW TO RUN   (RELOAD the page first if an older copy was pasted: stale copies stack up
 *               and an old copy's score gate would silently block a real submission)
 *   1. Open the game page, open DevTools (Cmd+Option+J), go to the Console tab.
 *   2. Paste this whole file and press Enter.
 *   3. Run:   dinoAgent.start({ episodes: 3, maxScore: 1500 })
 *      Keep the tab visible and in front: browsers throttle hidden tabs.
 *   4. Stop any time with dinoAgent.stop().  Results: dinoAgent.results
 *
 * LEADERBOARD (chromedino.com posts every finished game to /inc/set.php automatically)
 *   This script BLOCKS those score submissions by default. Two ways to allow them on purpose:
 *
  *   1. AUTO, no questions asked:   dinoAgent.start({ episodes: 30, submit: 'top2' })
 *        'top1' .. 'top5' = the place you are aiming for. Plays all the episodes straight.
 *        Whenever a run ends with a score ABOVE the current score of that place (re-read from
 *        the site before every run), it is posted under the nickname you set with the site's own
 *        Nickname button. Anything at or below it is never posted. Every decision is announced
 *        in the console. It refuses to start without a nickname, so nothing is ever posted as
 *        "Anonym".
 *   2. ASK FIRST at a fixed score:  dinoAgent.start({ submitAt: 5600, episodes: 30 })
 *        Plays on; when a run ends at or above 5600, a dialog shows the exact score and asks.
 *
 *   Optional in both: stopAt: 30000 ends a run by itself at that score.
 *
 * NOTHING IS LOST ON A REFRESH: every finished episode (with its flight-recorder trace) is saved
 * to the browser's localStorage. After a reload:  dinoAgent.saved()   (the list)
 * To hand the data over as a file:                dinoAgent.download() (saves dino_results_*.json)
 * To wipe it:                                     dinoAgent.clearSaved()
 *
 * It presses keys the same way a person would (dispatches keydown/keyup events);
 * it only READS the game's state to decide.
 *
 * TWO LESSONS BUILT IN (see notes/04-rule-based-agent.md)
 *   - Wide obstacles (groups of 2-3 cacti) need a LATER jump (widthAware).
 *   - The game's reported speed is not the real speed. On a 120 Hz screen obstacles move
 *     ~14% slower than `currentSpeed` says (whole-pixel steps per frame), so we MEASURE
 *     how fast obstacles actually move instead of trusting the number.
 */
(() => {
  const VERSION = 'v10-fast-fall';
  const DEFAULTS = { lead: 12, widthAware: true, arcCenter: 16.5, fastFall: true, minLeadLarge: 6.5, minLeadSmall: 5.0 };
  const GRAVITY = 0.6, RELEASE_HEIGHT = 10;   // let go of duck this close to the ground (clean landing)
  const DINO_W = 44, HITBOX_SHRINK = 4;
  const KEY = { JUMP: 32, DUCK: 40, RESTART: 13 };

  // Ticks until touchdown from height y (px) with upward velocity vy; fast = fast fall (velocity 1 down, displacement x3).
  function ticksToLand(y, vy, fast) {
    let t = 0;
    if (fast) vy = -1;
    while (y > 0 && t < 80) { y += vy * (fast ? 3 : 1); vy -= GRAVITY; t++; }
    return t;
  }

  // ---- The decision rule (pure function; same logic as rule_based.py) ---------------
  // obs: { hasObstacle, speed (px per 1/60 s), dist (px from dino's nose), width (px),
  //        flyY (px off the ground), onGround }
  function decide(obs, opts = {}) {
    const { lead, widthAware, arcCenter, fastFall, minLeadLarge, minLeadSmall } = { ...DEFAULTS, ...opts };
    if (!obs.hasObstacle) return 'NOOP';
    const arrivesIn = obs.dist / obs.speed;                 // distance -> time
    // Airborne and already past the previous obstacle? If a normal landing would leave less than
    // the minimum workable warning for this one, but a fast fall would leave enough, fall fast.
    // (Real traces: most deaths were landing with the next group only ~25 px away.)
    if (fastFall && !obs.onGround && obs.speedDrop && obs.dist > 0 && obs.flyY < 45) {
      return obs.flyY >= 20 ? 'DUCK' : 'FALL';             // a fast fall is under way: keep it up until touchdown
    }
    if (fastFall && !obs.onGround && obs.dist > 0 && obs.vy !== undefined && obs.vy <= 0 && obs.flyY < 45) {
      const need = obs.flyY >= 20 ? 1.0 : (obs.height >= 45 ? minLeadLarge : minLeadSmall);
      const slow = ticksToLand(obs.dinoY, obs.vy, false), quick = ticksToLand(obs.dinoY, obs.vy, true);
      if (arrivesIn - slow < need && need <= arrivesIn - quick) return obs.flyY >= 20 ? 'DUCK' : 'FALL';
    }
    if (obs.flyY >= 45) return 'NOOP';                      // high pterodactyl: flies over us
    if (obs.flyY >= 20) return arrivesIn < lead ? 'DUCK' : 'NOOP';   // mid: duck under
    // cactus or low pterodactyl: jump. A wide obstacle needs the high part of the arc to
    // cover it for longer, so jump later: aim the arc's center at the overlap's center.
    const l = widthAware
      ? arcCenter - (obs.width + DINO_W - 2 * HITBOX_SHRINK) / (2 * obs.speed)
      : lead;
    if (obs.onGround && arrivesIn >= 0 && arrivesIn < l) return 'JUMP';
    return 'NOOP';
  }

  // ---- Measure the real world speed ---------------------------------------------------
  // Track every obstacle's x between frames; px moved per 1/60 s is the true speed. Only
  // sample while the game is actually running (a frozen crash screen would read as speed 0),
  // average over a window, and prefer an obstacle's OWN speed (pterodactyls fly at +/-0.8).
  const WINDOW = 60, OWN_MIN = 8;
  const motion = { all: [], seen: new WeakMap(), own: new WeakMap(), v: null };
  const rate = (arr) => {
    let dx = 0, dt = 0;
    for (const s of arr) { dx += s[0]; dt += s[1]; }
    return dt > 0 ? (dx / dt) * (1000 / 60) : null;
  };
  function resetMotion() { motion.all.length = 0; motion.v = null; }
  function updateMotion(r, now, playing) {
    for (const o of r.horizon.obstacles) {
      const p = motion.seen.get(o);
      if (playing && p && now - p.t > 1 && now - p.t < 100) {
        const sample = [p.x - o.xPos, now - p.t];
        motion.all.push(sample);
        if (motion.all.length > WINDOW) motion.all.shift();
        motion.v = rate(motion.all);
        const own = motion.own.get(o) || [];
        own.push(sample);
        if (own.length > 20) own.shift();
        motion.own.set(o, own);
      }
      motion.seen.set(o, { x: o.xPos, t: now });
    }
  }
  function effectiveSpeed(r, o) {
    const nominal = r.currentSpeed;
    const own = o && motion.own.get(o);
    let v = own && own.length >= OWN_MIN ? rate(own) : motion.all.length >= 10 ? motion.v : null;
    if (v === null) v = nominal * 0.9;                       // before enough measurements
    return Math.min(nominal * 1.05, Math.max(nominal * 0.5, v));
  }

  // ---- Read the game's state and translate it into the same terms as our sim --------
  function observe(r) {
    const t = r.tRex;
    const noseX = t.xPos + t.config.WIDTH;
    const groundBottom = t.groundYPos + t.config.HEIGHT;   // y of the ground line
    const ahead = r.horizon.obstacles.filter((ob) => ob.xPos + ob.width > t.xPos);
    const o = ahead[0], o2 = ahead[1];
    const base = {
      speed: effectiveSpeed(r, o), nominalSpeed: r.currentSpeed, onGround: !t.jumping,
      ducking: !!t.ducking, speedDrop: !!t.speedDrop, vy: -t.jumpVelocity, dinoY: groundBottom - t.config.HEIGHT - t.yPos,
    };
    const describe = (ob) => ob && {
      dist: ob.xPos - noseX,
      flyY: groundBottom - (ob.yPos + ob.typeConfig.height),   // height of its underside
      type: ob.typeConfig.type, size: ob.size, width: ob.width, height: ob.typeConfig.height,
    };
    if (!o) return { ...base, hasObstacle: false };
    return { ...base, hasObstacle: true, ...describe(o), next: describe(o2) || null };
  }

  // ---- Keyboard ------------------------------------------------------------------
  function key(type, code) {
    const e = new Event(type, { bubbles: true, cancelable: true });
    e.keyCode = code;
    e.which = code;
    document.dispatchEvent(e);
  }

  const S = { running: false, results: [], jumpDown: false, jumpFrame: 0, duckDown: false, duckPressedInJump: false, releasedNearGround: false };

  function apply(action, r, frame) {
    const t = r.tRex;
    // JUMP. Hold the key until we land (letting go early cuts a jump short). If a new jump is
    // wanted on the very frame we land, release the old press AND press again in the same
    // frame: the old code released on that frame and lost the new jump (seen in the traces).
    const jumpFinished = S.jumpDown && !t.jumping && frame - S.jumpFrame >= 2;
    if (jumpFinished) { key('keyup', KEY.JUMP); S.jumpDown = false; }
    if (action === 'JUMP' && !S.jumpDown && !t.jumping) {
      key('keydown', KEY.JUMP);
      S.jumpDown = true;
      S.jumpFrame = frame;
    }
    // DUCK / FALL. On the ground DUCK crouches (FALL does nothing there). In the air both start the
    // fast fall. Rules learned from the real traces:
    //  - press ONCE per jump: every new press resets the fall speed, so re-pressing every frame (v1-v7)
    //    made the dino fall at a constant ~3 px per tick and land on pterodactyls;
    //  - if a fast fall touches down on the ground exactly, the game turns it into a duck and its
    //    physics then runs ~15x slower, leaving the dino unable to jump for about a second (v8/v9:
    //    13-27 frames stuck, 21 of 21 never recovered before the crash). So let go of the key just
    //    before touchdown (RELEASE_HEIGHT px) to land normally, and press DUCK again on the ground;
    //  - never let go of the key while stuck at ground level: a ducking dino is safe from a mid
    //    pterodactyl, a standing one is not (v9 stood the dino up and lost).
    const airborneWant = action === 'DUCK' || action === 'FALL';
    if (!t.jumping) { S.duckPressedInJump = false; S.releasedNearGround = false; }
    if (airborneWant && t.jumping) {
      if (!S.duckPressedInJump) { key('keydown', KEY.DUCK); S.duckPressedInJump = true; S.duckDown = true; }
      else if (S.duckDown && t.speedDrop && !S.releasedNearGround && (t.groundYPos - t.yPos) <= RELEASE_HEIGHT) {
        key('keyup', KEY.DUCK); S.duckDown = false; S.releasedNearGround = true;     // land cleanly
      }
    } else if (action === 'DUCK') {                          // on the ground
      if (!t.ducking) key('keydown', KEY.DUCK);
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
    S.duckPressedInJump = false; S.releasedNearGround = false;
  }

  // chrome://dino has r.playing. Older copies (chromedino.com) don't. There `activated` is not
  // reset after a restart and `paused` stays true while the game runs, so neither can be
  // trusted. On a freshly loaded page the loop already ticks before the dino has started, so
  // "loop running" is not enough either: also require that the dino has actually run
  // (distanceRan > 0), which is false on the start screen and true a moment after any start.
  const isPlaying = (r) =>
    r.playing !== undefined
      ? r.playing
      : !!(r.isRunning && r.isRunning() && !r.crashed && r.distanceRan > 0);

  const score = (r) => Math.ceil(r.distanceMeter.getActualDistance(Math.ceil(r.distanceRan)));

  // ---- Score gate: nothing reaches the leaderboard unless you say yes -----------------
  const gate = { installed: false, policy: null, blocked: [], sent: [], decisions: [], threshold: null, rank: 5 };
  const isScoreUrl = (u) => /\/inc\/set\.php/.test(String(u));
  function installGate() {
    if (gate.installed) return;
    gate.installed = true;
    const open = XMLHttpRequest.prototype.open, send = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.open = function (m, u, ...rest) {
      this.__dinoUrl = u; this.__dinoMethod = m;
      return open.call(this, m, u, ...rest);
    };
    XMLHttpRequest.prototype.send = function (body) {
      if (isScoreUrl(this.__dinoUrl) && String(this.__dinoMethod).toUpperCase() !== 'GET') {
        const rec = { url: String(this.__dinoUrl), body: String(body).slice(0, 300) };
        const posted = Number(new URLSearchParams(String(body)).get('score'));
        // The policy (set by start()) decides per post, from the score actually being posted.
        const ask = gate.policy ? gate.policy(posted) : { asked: false, allowed: false };
        gate.decisions.push({ score: posted, ...ask });
        if (!ask.allowed) { gate.blocked.push(rec); return; }
        gate.sent.push(rec);
      }
      return send.call(this, body);
    };
    const fetch0 = window.fetch;
    window.fetch = function (u, o) {
      if (isScoreUrl(u) && String((o && o.method) || 'GET').toUpperCase() !== 'GET') {
        gate.blocked.push({ url: String(u) });
        return Promise.resolve(new Response('{}'));
      }
      return fetch0.apply(this, arguments);
    };
    const beacon0 = navigator.sendBeacon && navigator.sendBeacon.bind(navigator);
    navigator.sendBeacon = function (u, d) {
      if (isScoreUrl(u)) { gate.blocked.push({ url: String(u) }); return true; }
      return beacon0 ? beacon0(u, d) : false;
    };
  }
  const currentName = () => {
    const m = document.cookie.match(/(^| )name=([^;]+)/);
    return (window.user_name || (m && decodeURIComponent(m[2])) || '(no nickname set: it would post as Anonym)');
  };

  // ---- Episode loop --------------------------------------------------------------
  // ---- Leaderboard: the score of 5th place of the day, read fresh from the site -----------
  // The list is <div class="high-scores"><h2>5 highest scores of the day:</h2><ul><li>
  // <span class="user">..</span><span class="result">NNNN</span></li>... (best first).
  async function readPlace(n) {
    const html = await (await fetch('/', { cache: 'no-store' })).text();
    const doc = new DOMParser().parseFromString(html, 'text/html');
    const h = [...doc.querySelectorAll('h2')].find((x) => /highest scores of the day/i.test(x.textContent));
    if (!h) throw new Error('leaderboard not found on the page');
    const scores = [...h.parentElement.querySelectorAll('.result')]
      .map((n) => Number(n.textContent.trim())).filter((n) => Number.isFinite(n));
    if (scores.length < n) return 0;                       // board not full that far: any score gets in
    return scores.slice(0, 5).sort((a, b) => b - a)[n - 1];
  }
  const readFifthPlace = () => readPlace(5);
  async function refreshThreshold() {
    try {
      const t = await readPlace(gate.rank);
      if (t !== gate.threshold) console.log(`[dinoAgent] place ${gate.rank} of the day is now ${t}`);
      gate.threshold = t;
    } catch (e) {
      gate.threshold = null;                               // unknown: block everything
      console.warn('[dinoAgent] could not read the leaderboard; posts stay blocked:', e.message);
    }
  }
  const hasNickname = () => !!(window.user_name || /(^| )name=[^;]+/.test(document.cookie));

  // ---- Persistence: results survive a page reload --------------------------------------
  const LOG_KEY = 'dinoAgentLog';
  const MAX_SAVED = 400;
  const readLog = () => { try { return JSON.parse(localStorage.getItem(LOG_KEY) || '[]'); } catch (e) { return []; } };
  function saveResult(res) {
    try {
      const log = readLog();
      log.push({ at: new Date().toISOString(), version: VERSION, ...res });
      while (log.length > MAX_SAVED) log.shift();
      localStorage.setItem(LOG_KEY, JSON.stringify(log));
    } catch (e) { console.warn('[dinoAgent] could not save the result (storage full?):', e.message); }
  }
  function downloadSaved() {
    const log = readLog();
    const blob = new Blob([JSON.stringify(log)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `dino_results_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    document.body.appendChild(a); a.click(); a.remove();
    console.log(`[dinoAgent] saved ${log.length} episodes to ${a.download} (check your Downloads folder)`);
    return log.length;
  }

  function start(options = {}) {
    if (S.running) throw new Error('A run is already in progress. Call dinoAgent.stop() first (two runs at once would corrupt each other).');
    const { episodes = 3, submit = null, submitAt = null, stopAt = null, ...rest0 } = options;
    const { stopAfterSubmit = submit === null, ...rest } = rest0;
    const { maxScore = stopAt !== null ? stopAt : (submit || submitAt !== null) ? Infinity : 1500, ...opts } = rest;
    const rankMatch = submit === null ? null : /^top([1-5])$/.exec(String(submit));
    if (submit !== null && !rankMatch) throw new Error("submit must be 'top1' ... 'top5' (or leave it out)");
    if (rankMatch) gate.rank = Number(rankMatch[1]);
    if (submit !== null && !hasNickname()) {
      throw new Error('No nickname set. Click the Nickname button on the page and set one first, so nothing is posted as Anonym.');
    }
    installGate();                                         // always: block score posts unless allowed
    // The site alert()s whatever its score check returns. A modal popup would freeze the run
    // until someone clicks it, so log it to the console instead (restored when the run ends).
    if (!S.alert0) {
      S.alert0 = window.alert;
      window.alert = (msg) => console.log('[dinoAgent] (site alert, not shown)', String(msg).slice(0, 300));
    }
    if (submit !== null) {
      gate.policy = (posted) => {
        const beats = gate.threshold !== null && posted > gate.threshold;
        if (beats) console.log(`[dinoAgent] SUBMITTING ${posted} as "${currentName()}" (place ${gate.rank} of the day was ${gate.threshold})`);
        else console.log(`[dinoAgent] not submitting ${posted} (place ${gate.rank} of the day: ${gate.threshold})`);
        return { asked: false, allowed: beats, threshold: gate.threshold };
      };
      refreshThreshold();
    } else {
      gate.policy = submitAt === null ? null : (posted) => {
        if (!(posted >= submitAt)) return { asked: false, allowed: false };   // below target: never post
        const ok = window.confirm(
          `The run ended with score ${posted}.\n\nSubmit it to the chromedino.com leaderboard as "${currentName()}"?\n\nOK = submit   Cancel = discard`);
        return { asked: true, allowed: ok };
      };
    }
    const Runner = window.Runner;
    if (!Runner || !Runner.instance_) throw new Error('No Runner.instance_ here: open the dino game');
    S.running = true;
    S.results = [];
    let frame = 0, lastObs = null, lastAction = 'NOOP', restartAt = 0, lastKick = 0;
    let lastDist = -1, lastProgress = performance.now(), prevTick = performance.now();
    let decisionIdx = gate.decisions.length;               // only count decisions made from now on
    const ring = [];                                       // flight recorder: last ~0.75 s of play
    const rec = (o, a) => ring.push([
      frame, Math.round(o.dinoY), o.onGround ? 0 : 1, o.ducking ? 1 : 0, a[0],
      o.hasObstacle ? Math.round(o.dist) : null, o.hasObstacle ? o.type.slice(0, 5) + o.size : null,
      o.hasObstacle ? Math.round(o.flyY) : null, o.next ? Math.round(o.next.dist) : null,
      o.next ? o.next.type.slice(0, 5) + o.next.size : null, +o.speed.toFixed(1), o.speedDrop ? 1 : 0,
    ]) && ring.length > 100 && ring.shift();
    console.log(`[dinoAgent] ${episodes} episodes, ` +
      (submit !== null ? `auto-submit anything above place ${gate.rank} of the day, as "${currentName()}"` : submitAt === null ? `stop at score ${maxScore}; score posts BLOCKED` : `plays on; at game over, scores >= ${submitAt} are offered to you; below that nothing is posted`),
      { ...DEFAULTS, ...opts });

    function finish(outcome) {
      const r = Runner.instance_;
      const res = {
        episode: S.results.length + 1, outcome, score: score(r), frames: frame,
        speed: +r.currentSpeed.toFixed(2), measuredSpeed: motion.v === null ? null : +motion.v.toFixed(2),
        distance: Math.round(r.distanceRan),
      };
      if (outcome === 'crash' && lastObs) {
        Object.assign(res, {
          killer: lastObs.type, group: lastObs.size, killerWidth: lastObs.width,
          distAtLastFrame: Math.round(lastObs.dist), flyY: Math.round(lastObs.flyY),
          action: lastAction, wasOnGround: lastObs.onGround, dinoY: Math.round(lastObs.dinoY),
        });
      }
      const posted = gate.decisions.slice(decisionIdx).pop();
      if (posted) res.posted = posted;                     // what the site tried to post, and the answer
      decisionIdx = gate.decisions.length;
      if (outcome === 'crash' || outcome === 'stuck') res.trace = ring.slice();   // full resolution
      ring.length = 0;
      S.results.push(res);
      saveResult(res);
      console.log('[dinoAgent]', JSON.stringify({ ...res, trace: undefined }));
      releaseKeys();
      frame = 0; lastObs = null; lastAction = 'NOOP';
    }

    function tick() {
      if (!S.running) return;
      const r = Runner.instance_;
      const now = performance.now();
      // A long gap between frames means the tab was hidden or the machine stalled. That is
      // not the game freezing, and any speed samples across it are meaningless: start fresh.
      if (now - prevTick > 500) { lastProgress = now; resetMotion(); }
      prevTick = now;
      updateMotion(r, now, isPlaying(r));

      if (isPlaying(r)) {                                  // is the game actually advancing?
        if (r.distanceRan !== lastDist) { lastDist = r.distanceRan; lastProgress = now; }
        else if (now - lastProgress > 2500) {
          console.warn('[dinoAgent] game seems frozen (distance not changing); logging and nudging');
          finish('stuck'); lastProgress = now;
          key('keydown', KEY.RESTART); key('keyup', KEY.RESTART);
        }
      }

      if (r.crashed) {
        if (!restartAt) {                                  // just crashed
          if (frame > 0) finish('crash');                  // ignore crashes we weren't playing
          const last = S.results[S.results.length - 1];
          if (stopAfterSubmit && last && last.posted && last.posted.asked) return done();   // you answered
          if (S.results.length >= episodes) return done();
          restartAt = now + 1300;
          if (submit !== null) refreshThreshold();       // board may have changed
        } else if (now >= restartAt) {
          key('keydown', KEY.RESTART); key('keyup', KEY.RESTART);   // Enter restarts
          restartAt = 0; resetMotion(); lastDist = -1; lastProgress = now;
        }
      } else if (!isPlaying(r)) {
        if (now - lastKick > 1000) {                       // press jump to start the game
          key('keydown', KEY.JUMP); key('keyup', KEY.JUMP); lastKick = now;
        }
      } else {
        frame++;
        lastObs = observe(r);
        lastAction = decide(lastObs, opts);
        rec(lastObs, lastAction);
        apply(lastAction, r, frame);
        if (score(r) >= maxScore) {                        // reached the ceiling: end the run
          releaseKeys();
          r.gameOver();                                    // the site posts here (gate decides)
          finish('survived');
          const last = S.results[S.results.length - 1];
          if ((stopAfterSubmit && last.posted && last.posted.asked) || S.results.length >= episodes) return done();
          restartAt = now + 1300;
          if (submit !== null) refreshThreshold();
        }
      }
      requestAnimationFrame(tick);
    }

    function done() {
      S.running = false;
      if (S.alert0) { window.alert = S.alert0; S.alert0 = null; }
      const crashed = S.results.filter((x) => x.outcome === 'crash').length;
      console.log(`[dinoAgent] done: ${S.results.length - crashed} survived, ${crashed} crashed`);
      console.table(S.results);
    }
    requestAnimationFrame(tick);
  }

  function stop() {
    S.running = false; releaseKeys();
    if (S.alert0) { window.alert = S.alert0; S.alert0 = null; }
    console.log('[dinoAgent] stopped');
  }

  window.dinoAgent = {
    version: VERSION, _apply: apply,
    start, stop, decide, observe, motion, gate, readPlace, readFifthPlace,
    saved: readLog, download: downloadSaved,
    clearSaved() { localStorage.removeItem(LOG_KEY); console.log('[dinoAgent] saved results cleared'); },
    // print the flight recorder of crash number i:  dinoAgent.trace(0)
    trace(i) { const t = (S.results[i] || {}).trace; if (t) console.table(t.map((x) => ({ frame: x[0], dinoY: x[1], air: x[2], duck: x[3], action: x[4], dist0: x[5], obs0: x[6], flyY0: x[7], dist1: x[8], obs1: x[9], speed: x[10] }))); return t; },
    get results() { return S.results; },
    get running() { return S.running; },
  };
  console.log(`[dinoAgent ${VERSION}] ready (${readLog().length} episodes saved from earlier). Run: dinoAgent.start({ episodes: 3, maxScore: 1500 })`);
})();
