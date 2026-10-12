# L&D training bot: rolling 30-day calendar, enrol and unenrol

**Background:** at Securitas I built a chatbot that connected our LMS (LearnUpon) to a cloud bot in the employee workflow. Staff could see every live training session for the next 30 days, at any time, and register, enrol in or leave a course or certification without going into the LMS. The original ran on Node.js with Microsoft Bot Framework and is internal. **This folder is a dependency-free re-creation** with a mocked LMS and invented courses, so the design can be read and run.

**The design problem:** everyone asks the bot the same question ("what's on?"), and the LMS API is rate-limited. Calling the API once per message falls over the moment a reminder email goes out.

**How it works**
- [`src/mockLms.js`](src/mockLms.js) is a LearnUpon-style API with sessions, seats and enrolments, plus a per-minute rate limit that returns 429.
- [`src/sessionCache.js`](src/sessionCache.js) is a shared cache of the **rolling 30-day calendar**. It serves fresh copies for 5 minutes, then serves stale copies while one background refresh runs. It falls back to the last good copy if the API fails. Every response is re-trimmed to *now → now + 30 days*, so a cached copy never shows sessions that have already happened. Seat counts are patched locally after each booking.
- [`src/bot.js`](src/bot.js) handles the intents (browse, certifications only, register, cancel, my bookings) and replies with suggested actions, which become buttons in Teams.

**Results**

[`simulate_load.js`](simulate_load.js) replays a working day: 400 employees and 1,200 messages, clustered around two reminder emails.

| | API calls | "System busy" replies |
|---|---|---|
| Direct API per message | 1,192 | 317 (26% of messages) |
| **With the session cache** | **416** (−65%) | **0** |

The tests ([`test/bot.test.js`](test/bot.test.js)) cover intent detection, the rolling window after the cache goes stale, the full register, list and cancel flow, and serving the calendar while the LMS is rate-limiting.

```bash
npm test          # 4 tests, no dependencies (Node 18+)
npm run load      # the load comparison above
npm run demo      # chat with the bot in the terminal
```

**Skills shown:** API integration, caching under rate limits, conversational design, Node.js, testing, and a product built for non-technical users.
