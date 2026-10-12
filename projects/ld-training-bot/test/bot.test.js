const test = require("node:test");
const assert = require("node:assert");
const { MockLms, DAY } = require("../src/mockLms");
const { SessionCache } = require("../src/sessionCache");
const { TrainingBot, detectIntent } = require("../src/bot");

function setup(start = Date.UTC(2026, 0, 5)) {
  let t = start;
  const clock = { now: () => t, advance: (ms) => (t += ms) };
  const lms = new MockLms({ now: clock.now });
  const cache = new SessionCache(lms, { now: clock.now });
  return { clock, lms, cache, bot: new TrainingBot({ lms, cache }) };
}

test("intents", () => {
  assert.equal(detectIntent("register S-0003"), "enroll");
  assert.equal(detectIntent("please cancel S-0003"), "unenroll");
  assert.equal(detectIntent("unenroll me from S-0001"), "unenroll");
  assert.equal(detectIntent("my bookings"), "mine");
  assert.equal(detectIntent("show certifications"), "certifications");
  assert.equal(detectIntent("what training is on next week?"), "browse");
});

test("calendar always covers a rolling 30 days, even from a stale cache", async () => {
  const { clock, cache } = setup();
  await cache.upcoming();
  clock.advance(10 * DAY); // cache is now stale and holds sessions that have passed
  const sessions = await cache.upcoming();
  const now = clock.now();
  assert.ok(sessions.length > 0);
  for (const s of sessions) {
    const t = Date.parse(s.startsAt);
    assert.ok(t >= now && t < now + 30 * DAY, `${s.id} outside the window`);
  }
  await new Promise((r) => setImmediate(r)); // let the background refresh land
  const refreshed = await cache.upcoming();
  const last = Math.max(...refreshed.map((s) => Date.parse(s.startsAt)));
  assert.ok(last > now + 20 * DAY, "refresh pulled in newly visible sessions");
});

test("register, list and cancel a booking", async () => {
  const { bot, cache } = setup();
  const browse = await bot.handle("u1", "show training");
  const open = browse.sessions.find((s) => s.booked < s.seats);
  const before = open.booked;
  const r = await bot.handle("u1", `register ${open.id}`);
  assert.match(r.text, /You're booked/);
  assert.equal((await cache.upcoming()).find((s) => s.id === open.id).booked, before + 1);
  assert.match((await bot.handle("u1", "my bookings")).text, new RegExp(open.id));
  assert.match((await bot.handle("u1", `register ${open.id}`)).text, /already booked/);
  assert.match((await bot.handle("u1", `cancel ${open.id}`)).text, /released/);
  assert.match((await bot.handle("u1", "my bookings")).text, /no bookings/);
});

test("cache keeps answering when the LMS rate-limits", async () => {
  const { bot, lms } = setup();
  await bot.handle("u1", "show training");
  lms.limitPerMinute = 0; // every API call now fails with 429
  const r = await bot.handle("u2", "show training");
  assert.match(r.text, /Live training/);
  const e = await bot.handle("u2", "register S-0001");
  assert.match(e.text, /busy/);
});
