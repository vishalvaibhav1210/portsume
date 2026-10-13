// Renders resume/cv.html to docs/cv.pdf with headless Chromium.
// Run from the repo root: node resume/render_cv.js
const path = require("path");
const { chromium } = require("playwright");

(async () => {
  const browser = await chromium.launch(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {});
  const page = await browser.newPage();
  await page.goto("file://" + path.resolve(__dirname, "cv.html"), { waitUntil: "networkidle" });
  await page.evaluate(() => document.fonts.ready);
  await page.pdf({ path: path.resolve(__dirname, "../docs/cv.pdf"), format: "A4", printBackground: true, preferCSSPageSize: true });
  await browser.close();
})();
