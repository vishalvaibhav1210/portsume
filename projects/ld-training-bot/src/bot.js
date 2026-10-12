// Conversational layer: turns a message into an intent and replies with text plus
// suggested actions (the buttons a Teams / Bot Framework card would show).

const INTENTS = [
  { name: "unenroll", re: /\b(unenrol+|un-enrol+|cancel|withdraw|drop)\b/i },
  { name: "enroll", re: /\b(enrol+|register|book|sign me up|join)\b/i },
  { name: "mine", re: /\b(my (trainings?|courses?|bookings?|enrol+ments?)|what am i)\b/i },
  { name: "certifications", re: /\bcert(ification)?s?\b/i },
  { name: "browse", re: /\b(show|list|what|upcoming|available|calendar|trainings?|courses?|sessions?|next)\b/i },
  { name: "help", re: /\b(help|hi|hello|hey|start)\b/i },
];

function detectIntent(text) {
  for (const i of INTENTS) if (i.re.test(text)) return i.name;
  return "unknown";
}

function sessionIdIn(text) {
  const m = text.match(/\bS-\d{4}\b/i);
  return m ? m[0].toUpperCase() : null;
}

const fmt = (iso) =>
  new Date(iso).toLocaleString("en-GB", { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "UTC" }) + " UTC";

function line(s) {
  const left = s.seats - s.booked;
  const seats = left <= 0 ? "full" : `${left} seat${left === 1 ? "" : "s"} left`;
  return `${s.id}  ${fmt(s.startsAt)}  ${s.title} (${s.type}, ${s.mode}) · ${seats}`;
}

class TrainingBot {
  constructor({ lms, cache }) {
    this.lms = lms;
    this.cache = cache;
  }

  async handle(userId, text) {
    const intent = detectIntent(text);
    const id = sessionIdIn(text);
    try {
      switch (intent) {
        case "enroll":
          return await (id ? this.#enroll(userId, id) : this.#browse("Which session? Reply with its code, e.g. “register S-0003”."));
        case "unenroll":
          return await (id ? this.#unenroll(userId, id) : this.#mine(userId, "Which booking should I cancel? Reply with its code."));
        case "mine":
          return await this.#mine(userId);
        case "certifications":
          return await this.#browse("Certification sessions in the next 30 days:", (s) => s.type === "certification");
        case "browse":
          return await this.#browse();
        case "help":
          return {
            text: "Hi! I can show live training for the next 30 days, book you in, or cancel a booking.",
            actions: ["Show upcoming training", "Show certifications", "My bookings"],
          };
        default:
          return { text: "Sorry, I didn't catch that. Try “show training”, “register S-0003” or “my bookings”.", actions: ["Show upcoming training"] };
      }
    } catch (err) {
      if (err.status === 429) {
        return { text: "The training system is busy right now. Please try that again in a minute.", actions: [] };
      }
      throw err;
    }
  }

  async #browse(header = "Live training in the next 30 days:", filter = () => true) {
    const sessions = (await this.cache.upcoming()).filter(filter).sort((a, b) => a.startsAt.localeCompare(b.startsAt));
    if (!sessions.length) return { text: "Nothing is scheduled in the next 30 days.", actions: [] };
    return {
      text: [header, ...sessions.slice(0, 8).map(line), sessions.length > 8 ? `…and ${sessions.length - 8} more.` : ""].filter(Boolean).join("\n"),
      actions: sessions.slice(0, 3).filter((s) => s.booked < s.seats).map((s) => `Register ${s.id}`),
      sessions,
    };
  }

  async #enroll(userId, id) {
    const res = await this.lms.enroll(userId, id);
    if (res.status === "enrolled") {
      this.cache.applySeatChange(id, +1);
      return { text: `You're booked on ${res.session.title}, ${fmt(res.session.startsAt)}. A calendar invite is on its way.`, actions: [`Cancel ${id}`, "My bookings"] };
    }
    if (res.status === "already_enrolled") return { text: `You're already booked on ${id}.`, actions: ["My bookings"] };
    return { text: `${res.session.title} on ${fmt(res.session.startsAt)} is full. Want to see other dates?`, actions: ["Show upcoming training"] };
  }

  async #unenroll(userId, id) {
    const res = await this.lms.unenroll(userId, id);
    if (res.status === "unenrolled") {
      this.cache.applySeatChange(id, -1);
      return { text: `Done. Your place on ${res.session.title} is released.`, actions: ["Show upcoming training"] };
    }
    return { text: `You don't have a booking for ${id}.`, actions: ["My bookings"] };
  }

  async #mine(userId, header = "Your bookings:") {
    const mine = await this.lms.myEnrollments(userId);
    if (!mine.length) return { text: "You have no bookings yet.", actions: ["Show upcoming training"] };
    return { text: [header, ...mine.map(line)].join("\n"), actions: mine.map((s) => `Cancel ${s.id}`) };
  }
}

module.exports = { TrainingBot, detectIntent };
