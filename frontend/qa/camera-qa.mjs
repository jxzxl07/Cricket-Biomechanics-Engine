/**
 * Browser QA for the in-browser recording flow.
 *
 * Chrome is launched with a synthetic camera device, so getUserMedia, the
 * countdown, the six-second MediaRecorder capture, the local preview and the
 * analysis of the recorded blob are all exercised for real.
 *
 *   node qa/camera-qa.mjs
 */
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";

const BASE = process.env.QA_BASE ?? "http://127.0.0.1:5173";
const OUT = resolve(process.cwd(), "../docs");
mkdirSync(OUT, { recursive: true });

const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok });
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
};

const browser = await chromium.launch({
  channel: "chrome",
  args: [
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    "--autoplay-policy=no-user-gesture-required",
  ],
});
const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, permissions: ["camera"] });
const page = await context.newPage();
const consoleErrors = [];
page.on("console", (message) => { if (message.type() === "error") consoleErrors.push(message.text()); });
page.on("pageerror", (error) => consoleErrors.push(`pageerror: ${error.message}`));

await page.goto(BASE, { waitUntil: "networkidle" });

// camera permission granted -> live preview
await page.getByRole("button", { name: /Use camera/ }).click();
await page.waitForSelector(".camera-feed", { timeout: 15_000 });
await page.waitForFunction(() => {
  const video = document.querySelector(".camera-feed");
  return video && video.videoWidth > 0;
}, { timeout: 15_000 });
check("camera preview starts after permission", await page.locator(".camera-feed").isVisible());
check("framing guide shown before recording", await page.locator(".frame-guide").isVisible());
check("flip camera control available", await page.getByTitle("Switch camera").isVisible());

// countdown then six-second recording
const started = Date.now();
await page.getByRole("button", { name: /Record 6 seconds/ }).click();
await page.waitForSelector(".countdown", { timeout: 5_000 });
check("countdown appears before recording", true);
await page.waitForSelector(".clip-status .ready", { timeout: 25_000 });
const elapsed = (Date.now() - started) / 1000;
check("recording produces a clip", true, `${elapsed.toFixed(1)}s including countdown`);
check("recording button returns to idle", await page.getByRole("button", { name: /Record 6 seconds/ }).isVisible());

const clipInfo = await page.evaluate(() => {
  const status = document.querySelector(".clip-status");
  return status ? status.innerText : "";
});
check("clip size reported", /\d+(\.\d+)? MB/.test(clipInfo), clipInfo.replace(/\n/g, " "));
await page.screenshot({ path: `${OUT}/screenshot-recording.png`, fullPage: false });

// analyse the recorded blob
await page.getByRole("button", { name: /Analyse my action/ }).click();
await page.waitForSelector(".results-page", { timeout: 180_000 });
check("recorded clip analyses end to end", await page.locator(".results-page").isVisible());

const replay = await page.evaluate(() => {
  const video = document.querySelector(".replay-stage video");
  return { readyState: video?.readyState ?? 0, width: video?.videoWidth ?? 0, error: video?.error?.code ?? null };
});
check("recorded clip is replayable in the browser", replay.readyState >= 2 && replay.error === null,
  `readyState=${replay.readyState} width=${replay.width} error=${replay.error}`);
check("no console errors", consoleErrors.length === 0, consoleErrors.slice(0, 3).join(" | "));

await browser.close();
const failed = results.filter((result) => !result.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length) {
  console.log("FAILED:", failed.map((result) => result.name).join(", "));
  process.exit(1);
}
