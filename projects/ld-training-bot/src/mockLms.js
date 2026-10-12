// In-memory stand-in for a LearnUpon-style LMS REST API.
// It enforces a rate limit like the real API does, which is the reason the bot
// needs its own cache. All course names and people are invented.

const DAY = 24 * 60 * 60 * 1000;

const CATALOGUE = [
  { id: "C-101", title: "Security Awareness Essentials", type: "course", hours: 2 },
  { id: "C-205", title: "Conflict De-escalation", type: "course", hours: 3 },
  { id: "C-310", title: "First Aid at Work", type: "certification", hours: 6 },
  { id: "C-412", title: "Fire Marshal", type: "certification", hours: 4 },
  { id: "C-520", title: "Leading Remote Teams", type: "course", hours: 2 },
  { id: "C-615", title: "Data Protection (GDPR) Refresher", type: "course", hours: 1 },
  { id: "C-730", title: "Health & Safety Supervisor", type: "certification", hours: 8 },
];

class RateLimitError extends Error {
  constructor(retryAfterMs) {
    super("429 Too Many Requests");
    this.status = 429;
    this.retryAfterMs = retryAfterMs;
  }
}

class MockLms {
  constructor({ now = () => Date.now(), limitPerMinute = 20, seed = 7 } = {}) {
    this.now = now;
    this.limitPerMinute = limitPerMinute;
    this.calls = [];
    this.callCount = 0;
    this.enrollments = new Map(); // userId -> Set(sessionId)
    this.sessions = this.#buildSessions(seed);
  }

  #buildSessions(seed) {
    // Deterministic live sessions spread over the next 75 days.
    let s = seed;
    const rand = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    const start = new Date(this.now());
    start.setUTCHours(0, 0, 0, 0);
    const sessions = [];
    let n = 1;
    for (let d = 1; d <= 75; d++) {
      for (const course of CATALOGUE) {
        if (rand() < 0.07) {
          const at = new Date(start.getTime() + d * DAY);
          at.setUTCHours([4, 6, 9, 11][Math.floor(rand() * 4)], 30);
          sessions.push({
            id: `S-${String(n++).padStart(4, "0")}`,
            courseId: course.id,
            title: course.title,
            type: course.type,
            startsAt: at.toISOString(),
            seats: 12 + Math.floor(rand() * 13),
            booked: Math.floor(rand() * 10),
            mode: rand() < 0.6 ? "Virtual" : "Classroom",
          });
        }
      }
    }
    return sessions;
  }

  #hit() {
    const t = this.now();
    this.calls = this.calls.filter((c) => t - c < 60_000);
    if (this.calls.length >= this.limitPerMinute) {
      throw new RateLimitError(60_000 - (t - this.calls[0]));
    }
    this.calls.push(t);
    this.callCount++;
  }

  async listSessions({ from, to }) {
    this.#hit();
    return this.sessions
      .filter((x) => x.startsAt >= from && x.startsAt < to)
      .map((x) => ({ ...x }));
  }

  async enroll(userId, sessionId) {
    this.#hit();
    const session = this.sessions.find((x) => x.id === sessionId);
    if (!session) throw Object.assign(new Error("Session not found"), { status: 404 });
    const mine = this.enrollments.get(userId) ?? new Set();
    if (mine.has(sessionId)) return { status: "already_enrolled", session: { ...session } };
    if (session.booked >= session.seats) return { status: "full", session: { ...session } };
    session.booked++;
    mine.add(sessionId);
    this.enrollments.set(userId, mine);
    return { status: "enrolled", session: { ...session } };
  }

  async unenroll(userId, sessionId) {
    this.#hit();
    const mine = this.enrollments.get(userId);
    const session = this.sessions.find((x) => x.id === sessionId);
    if (!mine || !mine.has(sessionId) || !session) return { status: "not_enrolled" };
    mine.delete(sessionId);
    session.booked--;
    return { status: "unenrolled", session: { ...session } };
  }

  async myEnrollments(userId) {
    this.#hit();
    const mine = this.enrollments.get(userId) ?? new Set();
    return this.sessions.filter((x) => mine.has(x.id)).map((x) => ({ ...x }));
  }
}

module.exports = { MockLms, RateLimitError, CATALOGUE, DAY };
