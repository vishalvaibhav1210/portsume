// Interactive console demo: node demo.js
const readline = require("node:readline");
const { MockLms } = require("./src/mockLms");
const { SessionCache } = require("./src/sessionCache");
const { TrainingBot } = require("./src/bot");

const lms = new MockLms();
const bot = new TrainingBot({ lms, cache: new SessionCache(lms) });
const rl = readline.createInterface({ input: process.stdin, output: process.stdout, prompt: "you> " });
console.log('Training bot demo. Try "show training", "show certifications", "register S-0003", "my bookings", "cancel S-0003".');
rl.prompt();
rl.on("line", async (text) => {
  const r = await bot.handle("demo-user", text);
  console.log(`bot> ${r.text}`);
  if (r.actions?.length) console.log(`     [${r.actions.join("] [")}]`);
  rl.prompt();
});
