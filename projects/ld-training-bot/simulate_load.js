// Replays a working day of bot traffic against the mock LMS, with and without the
// session cache, and prints how many API calls and rate-limit errors each produced.
const { MockLms } = require("./src/mockLms");
const { SessionCache } = require("./src/sessionCache");
const { TrainingBot } = require("./src/bot");

const USERS = 400;          // employees opening the bot in one day
const MSGS_PER_USER = 3;    // browse, maybe certifications, maybe register
const DAY_MS = 9 * 60 * 60 * 1000;

class NoCache {
  constructor(lms) { this.lms = lms; }
  async upcoming() { return this.lms.listSessions(new SessionCache(this.lms, { now: this.lms.now }).window()); }
  applySeatChange() {}
}

async function run(useCache) {
  let t = Date.UTC(2026, 0, 5, 8);
  const now = () => t;
  const lms = new MockLms({ now, limitPerMinute: 20 });
  const cache = useCache ? new SessionCache(lms, { now }) : new NoCache(lms);
  const bot = new TrainingBot({ lms, cache });
  let seed = 11;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  // Arrivals cluster around 9:00 and 14:00, like real reminder emails.
  const events = [];
  for (let u = 0; u < USERS; u++) {
    const peak = rand() < 0.5 ? 1 : 6;
    const start = (peak + (rand() - 0.5) * 0.6) * 3600_000;
    for (let m = 0; m < MSGS_PER_USER; m++) events.push({ at: start + m * 20_000, u, m });
  }
  events.sort((a, b) => a.at - b.at);
  let busy = 0, booked = 0, firstSessions = null;
  for (const e of events) {
    t = Date.UTC(2026, 0, 5, 8) + e.at;
    const text = e.m === 0 ? "show training" : e.m === 1 ? "show certifications" : null;
    let r;
    if (text) r = await bot.handle(`u${e.u}`, text);
    else {
      const sessions = await cache.upcoming().catch(() => []);
      const pick = sessions[Math.floor(rand() * sessions.length)];
      r = await bot.handle(`u${e.u}`, `register ${pick ? pick.id : "S-0001"}`);
      if (/booked on/.test(r.text)) booked++;
    }
    if (/busy/.test(r.text)) busy++;
  }
  return { messages: events.length, apiCalls: lms.callCount, busyReplies: busy, booked, stats: cache.stats };
}

(async () => {
  const without = await run(false);
  const withCache = await run(true);
  const out = { without_cache: without, with_cache: withCache,
    api_call_reduction_pct: Math.round(100 * (1 - withCache.apiCalls / without.apiCalls)) };
  console.log(JSON.stringify(out, null, 2));
  require("fs").mkdirSync(__dirname + "/results", { recursive: true });
  require("fs").writeFileSync(__dirname + "/results/load_test.json", JSON.stringify(out, null, 2));
})();
