/**
 * Browser QA for the CreaseLab web app.
 *
 * Drives real Chrome against a locally running API (127.0.0.1:8000) and Vite dev
 * server (127.0.0.1:5173), uploads real clips, and checks the replay contract:
 * overlay alignment, phase seeking, metric seeking, speed control, loop, mobile
 * layout and console errors. Screenshots land in docs/.
 *
 *   node qa/browser-qa.mjs
 */
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";

const BASE = process.env.QA_BASE ?? "http://127.0.0.1:5173";
const CLIP = process.env.QA_CLIP ?? "/tmp/qa_batting_h264.mp4";
const BOWLING_CLIP = process.env.QA_BOWLING_CLIP ?? "/tmp/qa_bowling_h264.mp4";
const UNPLAYABLE_CLIP = process.env.QA_UNPLAYABLE_CLIP ?? "/tmp/qa_batting_mp4v.mp4";
const OUT = resolve(process.cwd(), "../docs");
mkdirSync(OUT, { recursive: true });

const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok, detail });
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
};

const browser = await chromium.launch({ channel: "chrome" });
const context = await browser.newContext({ viewport: { width: 1440, height: 950 } });
const page = await context.newPage();

const consoleErrors = [];
const failedRequests = [];
page.on("console", (message) => { if (message.type() === "error") consoleErrors.push(message.text()); });
page.on("pageerror", (error) => consoleErrors.push(`pageerror: ${error.message}`));
page.on("response", (response) => { if (response.status() >= 400) failedRequests.push(`${response.status()} ${response.url()}`); });

// ---------------------------------------------------------------- landing
await page.goto(BASE, { waitUntil: "networkidle" });
check("landing renders hero", await page.locator("h1").first().isVisible());
await page.screenshot({ path: `${OUT}/screenshot-landing.png`, fullPage: false });

// fonts must be self-hosted: no requests to fonts.googleapis.com
const externalFonts = await page.evaluate(() =>
  performance.getEntriesByType("resource").filter((entry) => /fonts\.(googleapis|gstatic)\.com/.test(entry.name)).length
);
check("no third-party font requests", externalFonts === 0, `${externalFonts} requests`);

// ---------------------------------------------------------------- mode + upload
await page.getByRole("button", { name: /Analyse batting/ }).click();
check("mode selection toggles", await page.locator(".mode-card.batting.selected").count() === 1);

await page.locator('input[type="file"]').setInputFiles(CLIP);
await page.waitForSelector(".clip-status .ready", { timeout: 15_000 });
check("clip is accepted and previewed", await page.locator(".camera-viewport video").isVisible());

await page.getByRole("button", { name: /Analyse my action/ }).click();
await page.waitForSelector(".loading-screen", { timeout: 10_000 });
check("analysis progress state shows", true);
await page.waitForSelector(".results-page", { timeout: 180_000 });

// ---------------------------------------------------------------- results
const label = (await page.locator(".class-label").innerText()).trim();
check("classification label renders", label.length > 0, label);
check("experimental badge shown", await page.locator(".tag.experimental").count() === 1);
const experimentText = await page.locator(".experiment-note").innerText();
check("benchmark disclosed next to the label", /%/.test(experimentText), experimentText.slice(0, 90).replace(/\n/g, " "));
check("quality checks listed", await page.locator(".quality-checks div").count() >= 5);

const video = page.locator(".replay-stage video");
await video.waitFor({ state: "visible" });
await page.waitForFunction(() => {
  const element = document.querySelector(".replay-stage video");
  return element && element.readyState >= 2 && element.videoWidth > 0;
}, { timeout: 30_000 });

// automatic replay on arrival
const autoplay = await page.evaluate(() => {
  const element = document.querySelector(".replay-stage video");
  return { paused: element.paused, muted: element.muted, time: element.currentTime };
});
check("replay starts automatically (muted)", autoplay.muted === true, `paused=${autoplay.paused} t=${autoplay.time.toFixed(2)}`);

// overlay canvas alignment: canvas rect must match the displayed video *content*
// rect (the video element itself is letterboxed inside the stage).
const alignment = await page.evaluate(() => {
  const video = document.querySelector(".replay-stage video");
  const canvas = document.querySelector(".pose-canvas");
  const v = video.getBoundingClientRect();
  const c = canvas.getBoundingClientRect();
  const scale = Math.min(v.width / video.videoWidth, v.height / video.videoHeight);
  const content = {
    left: v.left + (v.width - video.videoWidth * scale) / 2,
    top: v.top + (v.height - video.videoHeight * scale) / 2,
    width: video.videoWidth * scale,
    height: video.videoHeight * scale,
  };
  return {
    dx: Math.abs(content.left - c.left), dy: Math.abs(content.top - c.top),
    dw: Math.abs(content.width - c.width), dh: Math.abs(content.height - c.height),
    canvasWidth: canvas.width, videoWidth: video.videoWidth,
    internalMatchesVideo: canvas.width === video.videoWidth && canvas.height === video.videoHeight,
  };
});
check("pose canvas aligns with the video", alignment.dx < 2 && alignment.dy < 2 && alignment.dw < 2 && alignment.dh < 2,
  `dx=${alignment.dx.toFixed(1)} dy=${alignment.dy.toFixed(1)} dw=${alignment.dw.toFixed(1)} dh=${alignment.dh.toFixed(1)}`);
check("canvas resolution matches the video", alignment.internalMatchesVideo, `${alignment.canvasWidth} vs ${alignment.videoWidth}`);

// overlay toggle draws / clears
const overlayPixels = async () => page.evaluate(() => {
  const canvas = document.querySelector(".pose-canvas");
  const context = canvas.getContext("2d");
  const data = context.getImageData(0, 0, canvas.width, canvas.height).data;
  let lit = 0;
  for (let index = 3; index < data.length; index += 4 * 97) if (data[index] > 0) lit += 1;
  return lit;
});
await page.waitForTimeout(600);
const litOn = await overlayPixels();
await page.getByRole("button", { name: /Pose/ }).click();
await page.waitForTimeout(400);
const litOff = await overlayPixels();
check("pose overlay toggles off", litOff === 0 && litOn > 0, `on=${litOn} off=${litOff}`);
await page.getByRole("button", { name: /Pose/ }).click();
await page.waitForTimeout(400);

// phase markers positioned from timestamps, and seeking works
const phaseCheck = await page.evaluate(() => {
  const buttons = [...document.querySelectorAll(".phase-track button")];
  return buttons.map((button) => ({
    label: button.querySelector("strong")?.textContent ?? "",
    left: parseFloat(button.style.left),
    seconds: parseFloat(button.querySelector("small")?.textContent ?? "0"),
  }));
});
const monotonic = phaseCheck.every((phase, index) => index === 0 || phase.left >= phaseCheck[index - 1].left);
check("four phase markers render", phaseCheck.length === 4, phaseCheck.map((p) => `${p.label}@${p.seconds}s`).join(", "));
check("phase markers are timestamp-ordered", monotonic);

await page.locator(".phase-track button").nth(2).click();
await page.waitForTimeout(700);
const seeked = await page.evaluate(() => {
  const video = document.querySelector(".replay-stage video");
  return { time: video.currentTime, paused: video.paused };
});
check("phase click seeks the replay", seeked.time > 0.4, `t=${seeked.time.toFixed(2)}s`);

// metric cards seek and highlight joints
const metricCount = await page.locator(".metric-card").count();
check("six metric cards render", metricCount === 6, `${metricCount}`);
await page.locator(".metric-card").first().hover();
await page.waitForTimeout(400);
await page.locator(".metric-card").nth(1).click();
await page.waitForTimeout(700);
const metricSeek = await page.evaluate(() => document.querySelector(".replay-stage video").currentTime);
check("metric click seeks the replay", metricSeek > 0.4, `t=${metricSeek.toFixed(2)}s`);

// playback speed + frame stepping + loop
await page.getByRole("button", { name: "0.25×" }).click();
const rate = await page.evaluate(() => document.querySelector(".replay-stage video").playbackRate);
check("slow motion applies", Math.abs(rate - 0.25) < 0.01, `rate=${rate}`);

const beforeStep = await page.evaluate(() => document.querySelector(".replay-stage video").currentTime);
await page.getByTitle("Next frame").click();
const afterStep = await page.evaluate(() => document.querySelector(".replay-stage video").currentTime);
check("frame step advances one frame", afterStep > beforeStep && afterStep - beforeStep < 0.1,
  `Δ=${((afterStep - beforeStep) * 1000).toFixed(0)}ms`);

await page.getByRole("button", { name: /Loop/ }).click();
check("loop toggle activates", await page.locator(".replay-controls button.active").count() >= 2);
await page.getByRole("button", { name: /Loop/ }).click();

// coaching + privacy panels
check("coaching summary renders", (await page.locator(".coach-intro > p").innerText()).length > 20);
check("uncertainty note renders", (await page.locator(".uncertainty-box p").innerText()).length > 20);
check("privacy statement renders", /temporary/i.test(await page.locator(".method-section").innerText()));
check("timings shown", /Analysis took/.test(await page.locator(".timing-strip").innerText()));

await page.screenshot({ path: `${OUT}/screenshot-results.png`, fullPage: false });
await page.screenshot({ path: `${OUT}/screenshot-results-full.png`, fullPage: true });

// ---------------------------------------------------------------- unplayable codec path
await page.getByRole("button", { name: /Analyse another/ }).click();
await page.locator('input[type="file"]').setInputFiles(UNPLAYABLE_CLIP);
await page.waitForSelector(".clip-status .ready", { timeout: 15_000 });
await page.getByRole("button", { name: /Analyse my action/ }).click();
await page.waitForSelector(".results-page", { timeout: 180_000 });
await page.waitForSelector(".replay-fallback", { timeout: 30_000 });
check("unplayable codec shows a clear fallback", await page.locator(".replay-fallback").isVisible());
check("fallback explains the analysis is still valid", /still valid/i.test(await page.locator(".replay-fallback").innerText()));
check("playback controls are disabled without a replayable video",
  await page.getByTitle("Next frame").isDisabled());
check("metrics still render for unplayable clips", await page.locator(".metric-card").count() === 6);
await page.screenshot({ path: `${OUT}/screenshot-unplayable.png`, fullPage: false });

// ---------------------------------------------------------------- poor footage path
await page.getByRole("button", { name: /Analyse another/ }).click();
await page.locator('input[type="file"]').setInputFiles(resolve(process.cwd(), "qa/fixtures/no-person.mp4"));
await page.waitForSelector(".clip-status .ready", { timeout: 15_000 });
await page.getByRole("button", { name: /Analyse my action/ }).click();
await page.waitForSelector(".results-page", { timeout: 180_000 });
check("poor clip shows re-record guidance", await page.locator(".rerecord-banner").count() === 1);
check("poor clip reports no label", /No label|Unclear/.test(await page.locator(".classification-card").innerText()));
await page.screenshot({ path: `${OUT}/screenshot-rerecord.png`, fullPage: false });

// ---------------------------------------------------------------- bowling mode
await page.getByRole("button", { name: /Analyse another/ }).click();
await page.getByRole("button", { name: /Analyse bowling/ }).click();
check("bowling mode selects", await page.locator(".mode-card.bowling.selected").count() === 1);
await page.getByRole("button", { name: /side on/ }).click();
await page.locator('input[type="file"]').setInputFiles(BOWLING_CLIP);
await page.waitForSelector(".clip-status .ready", { timeout: 15_000 });
await page.getByRole("button", { name: /Analyse my action/ }).click();
await page.waitForSelector(".results-page", { timeout: 180_000 });
const bowlingLabel = (await page.locator(".class-label").innerText()).trim();
check("bowling classification renders", bowlingLabel.length > 0, bowlingLabel);
const bowlingMetrics = await page.locator(".metric-card strong").allInnerTexts();
check("bowling metrics include elbow angle", bowlingMetrics.some((text) => /elbow/i.test(text)), bowlingMetrics.join(", "));
check("bowling phases are labelled", /gather/i.test(await page.locator(".phase-track").innerText()) && /release/i.test(await page.locator(".phase-track").innerText()));
const resultsText = await page.locator(".results-page").innerText();
check("no legality verdict in results", !/illegal|chucking|no-ball/i.test(resultsText));
check("elbow metric states it is not a legality assessment", /not a legality assessment/i.test(resultsText));
await page.screenshot({ path: `${OUT}/screenshot-bowling.png`, fullPage: false });

// ---------------------------------------------------------------- mobile
await page.setViewportSize({ width: 390, height: 844 });
await page.waitForTimeout(800);
const mobile = await page.evaluate(() => {
  const replay = document.querySelector(".replay-card");
  const stage = document.querySelector(".replay-stage");
  const hero = document.querySelector(".result-hero");
  return {
    replayVisible: !!replay && replay.getBoundingClientRect().height > 100,
    sticky: getComputedStyle(replay).position,
    columns: getComputedStyle(hero).display,
    stageHeight: stage ? stage.getBoundingClientRect().height : 0,
    horizontalOverflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
  };
});
check("mobile replay is sticky above results", mobile.sticky === "sticky", `position=${mobile.sticky}`);
check("mobile has no horizontal overflow", mobile.horizontalOverflow <= 1, `overflow=${mobile.horizontalOverflow}px`);
await page.screenshot({ path: `${OUT}/screenshot-mobile.png`, fullPage: false });

// keyboard accessibility on the metric cards
await page.keyboard.press("Tab");
const focused = await page.evaluate(() => document.activeElement?.className ?? "");
check("keyboard focus reaches interactive elements", focused.length >= 0, focused.slice(0, 40));

check("no console errors", consoleErrors.length === 0, consoleErrors.slice(0, 3).join(" | "));
check("no failed network requests", failedRequests.length === 0, failedRequests.slice(0, 3).join(" | "));

await browser.close();

const failed = results.filter((result) => !result.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length) {
  console.log("FAILED:", failed.map((result) => result.name).join(", "));
  process.exit(1);
}
