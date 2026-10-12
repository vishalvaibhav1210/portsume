// Cache of the rolling 30-day training calendar.
//
// Every user who opens the bot wants the same calendar, so it is fetched once and
// shared. Entries are served fresh for `ttlMs`, then served stale while a single
// background refresh runs. If the LMS rate-limits or fails, the last good copy
// keeps the bot answering instead of erroring.

const { DAY } = require("./mockLms");

const WINDOW_DAYS = 30;

class SessionCache {
  constructor(lms, { now = () => Date.now(), ttlMs = 5 * 60_000 } = {}) {
    this.lms = lms;
    this.now = now;
    this.ttlMs = ttlMs;
    this.entry = null; // { fetchedAt, sessions }
    this.refreshing = null;
    this.stats = { hits: 0, staleServes: 0, misses: 0, refreshErrors: 0 };
  }

  window() {
    const from = new Date(this.now());
    const to = new Date(from.getTime() + WINDOW_DAYS * DAY);
    return { from: from.toISOString(), to: to.toISOString() };
  }

  async #refresh() {
    if (!this.refreshing) {
      this.refreshing = this.lms
        .listSessions(this.window())
        .then((sessions) => {
          this.entry = { fetchedAt: this.now(), sessions };
          return sessions;
        })
        .catch((err) => {
          this.stats.refreshErrors++;
          if (!this.entry) throw err;
          return this.entry.sessions;
        })
        .finally(() => {
          this.refreshing = null;
        });
    }
    return this.refreshing;
  }

  // Always trims to the rolling window, so a cached copy never shows past sessions.
  #visible(sessions) {
    const { from, to } = this.window();
    return sessions.filter((s) => s.startsAt >= from && s.startsAt < to);
  }

  async upcoming() {
    if (!this.entry) {
      this.stats.misses++;
      return this.#visible(await this.#refresh());
    }
    const age = this.now() - this.entry.fetchedAt;
    if (age < this.ttlMs) {
      this.stats.hits++;
    } else {
      this.stats.staleServes++;
      this.#refresh(); // background; the user gets the stale copy immediately
    }
    return this.#visible(this.entry.sessions);
  }

  // Keep seat counts right after a booking without a full refetch.
  applySeatChange(sessionId, delta) {
    const s = this.entry?.sessions.find((x) => x.id === sessionId);
    if (s) s.booked += delta;
  }
}

module.exports = { SessionCache, WINDOW_DAYS };
