// Real-Chrome playback check. Always muted. Usage:
//   node engine-tools/check.mjs <steady|wander> <seconds> [throttle] [skips]
// Records the master output through a MediaStreamDestination, then scans it for silent gaps.
import fs from 'node:fs';
import {spawnSync} from 'node:child_process';
import {serve} from './serve.mjs';
import {chromium} from 'playwright-core';

const [style = 'steady', seconds = '60', throttle = '1', skips = '0'] = process.argv.slice(2);
const name = `${style}-${seconds}s-x${throttle}`;
const out = new URL('../out/', import.meta.url).pathname;
fs.mkdirSync(out, {recursive: true});
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), name, ...a);

const server = await serve(), url = `http://127.0.0.1:${server.address().port}/?style=${style}${process.env.QUERY ? '&' + process.env.QUERY : ''}`;
const browser = await chromium.launch({channel: 'chrome', headless: true, args: ['--mute-audio']});
const page = await browser.newPage();
const problems = [];
page.on('console', m => { if (['error', 'warning'].includes(m.type())) problems.push(m.type() + ': ' + m.text()); });
page.on('pageerror', e => problems.push('pageerror: ' + e.message));
page.on('requestfailed', r => problems.push('requestfailed: ' + r.url()));
await page.addInitScript(() => {
  window.__csp = [];
  document.addEventListener('securitypolicyviolation', e => window.__csp.push(e.violatedDirective + ' ' + e.blockedURI));
  window.__long = [];
  new PerformanceObserver(list => { for (const e of list.getEntries()) window.__long.push([Math.round(e.startTime), Math.round(e.duration)]); }).observe({type: 'longtask', buffered: true});
});
const cdp = await page.context().newCDPSession(page);
if (+throttle > 1) await cdp.send('Emulation.setCPUThrottlingRate', {rate: +throttle});
await page.goto(url);
await page.waitForFunction(() => document.querySelector('#key').textContent !== '—');
await page.evaluate(() => { window.__playAt = performance.now(); });
await page.click('#play');
await page.waitForFunction(() => window.lofiDebug.ctx?.state === 'running' && window.lofiDebug.report);
await page.evaluate(() => {
  const {ctx, master} = window.lofiDebug, dest = ctx.createMediaStreamDestination();
  master.connect(dest);
  const rec = new MediaRecorder(dest.stream, {mimeType: 'audio/webm;codecs=opus', audioBitsPerSecond: 160000});
  window.__chunks = [];
  rec.ondataavailable = e => window.__chunks.push(e.data);
  rec.start(1000);
  window.__rec = rec;
  window.__recStart = performance.now();
});
// Hidden-tab runs live in engine-tools/hidden.mjs: under Playwright a page always reports 'visible'.
const t0 = Date.now(), skipAt = [];
const nSkips = +skips;
for (let k = 0; k < nSkips; k++) skipAt.push(+seconds - 12 * (nSkips - k) - 6);
const skipResults = [];
let sample = 0;
while ((Date.now() - t0) / 1000 < +seconds) {
  await page.waitForTimeout(1000);
  const elapsed = (Date.now() - t0) / 1000;
  if (skipAt.length && elapsed >= skipAt[0]) {
    skipAt.shift();
    const before = await page.evaluate(() => ({fades: window.lofiDebug.fades.length, shown: window.lofiDebug.shownLog.length}));
    await page.click('#skip');
    const clicked = await page.evaluate(() => window.lofiDebug.skips.at(-1));
    // A second press inside the gap must not start another change.
    await page.waitForTimeout(300);
    await page.click('#skip');
    await page.waitForFunction(n => window.lofiDebug.fades.length > n, before.fades, {timeout: 10000});
    await page.waitForTimeout(2500);
    const debounced = await page.evaluate(n => window.lofiDebug.fades.length === n + 1 && window.lofiDebug.skips.length, before.fades);
    const r = await page.evaluate(([c, n]) => ({fadeMs: window.lofiDebug.fades.at(-1).at - c,
      shownMs: window.lofiDebug.shownLog.length > n ? window.lofiDebug.shownLog.at(-1).at - c : null,
      tune: document.querySelector('#tune').textContent}), [clicked, before.shown]);
    r.secondPressIgnored = debounced !== false;
    skipResults.push(r);
    log('skip', r);
  }
  if (++sample % 30 === 0) log(await page.evaluate(() => {
    const r = window.lofiDebug.report;
    return {hidden: document.visibilityState, songs: window.lofiDebug.shownLog.length, underrun: r.underrun, minAhead: +r.minAhead.toFixed(2), steadyMinAhead: +r.steadyMinAhead.toFixed(2),
      decks: r.decks.map(d => [d[0], +(d[1] / 44100).toFixed(1), +(d[3] / 44100).toFixed(1)]), tune: document.querySelector('#tune').textContent};
  }));
}
const result = await page.evaluate(async () => {
  const d = window.lofiDebug, rec = window.__rec;
  const done = new Promise(resolve => { rec.onstop = resolve; });
  rec.stop();
  await done;
  const blob = new Blob(window.__chunks, {type: 'audio/webm'});
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = '';
  for (let i = 0; i < bytes.length; i += 32768) binary += String.fromCharCode(...bytes.subarray(i, i + 32768));
  const ms = d.barMs.slice().sort((a, b) => a - b);
  return {
    webm: btoa(binary), recordedWallSec: (performance.now() - window.__recStart) / 1000,
    underrunFrames: d.report.underrun, minAheadSec: d.report.minAhead, steadyMinAheadSec: d.report.steadyMinAhead,
    shown: d.shownLog.map(s => ({deck: s.deck, style: s.style, atSec: +(s.at / 1000).toFixed(1)})),
    fades: d.fades.map(f => ({to: f.to, atSec: +(f.at / 1000).toFixed(1)})),
    skipsAtSec: d.skips.map(s => +(s / 1000).toFixed(1)),
    barMs: {n: ms.length, median: ms[ms.length >> 1], p95: ms[Math.floor(ms.length * .95)], max: ms.at(-1)},
    longTasks: {count: window.__long.length, maxMs: Math.max(0, ...window.__long.map(l => l[1])), list: window.__long.slice(0, 40), playClickedAt: Math.round(window.__playAt)},
    csp: window.__csp, visibility: document.visibilityState, contextState: d.ctx.state, sampleRate: d.ctx.sampleRate, recStartSec: window.__recStart / 1000, model: d.model, probe: d.probe
  };
});
await browser.close();
server.close();

const webmPath = out + `rec-${name}.webm`;
fs.writeFileSync(webmPath, Buffer.from(result.webm, 'base64'));
delete result.webm;
const pcm = spawnSync('ffmpeg', ['-nostdin', '-v', 'error', '-i', webmPath, '-f', 'f32le', '-ac', '2', '-ar', '44100', '-'], {maxBuffer: 1 << 30});
const f = new Float32Array(pcm.stdout.buffer, pcm.stdout.byteOffset, pcm.stdout.byteLength / 4);
const frames = f.length / 2, threshold = 1e-4;
let run = 0, peak = 0, lead = 0;
const runs = [];
// The recorder starts with a short silent warm-up; measure from the first audible sample.
while (lead < frames && Math.max(Math.abs(f[2 * lead]), Math.abs(f[2 * lead + 1])) < threshold) lead++;
for (let i = lead; i < frames; i++) {
  const v = Math.max(Math.abs(f[2 * i]), Math.abs(f[2 * i + 1]));
  peak = Math.max(peak, v);
  if (v < threshold) run++;
  else { if (run > 2205) runs.push([(i - run) / 44100, run / 44.1]); run = 0; }
}
if (run > 2205) runs.push([(frames - run) / 44100, run / 44.1]);
// Silences that start within 3 s after a New tune press are the intended gap; any other is a fault.
const skipsRec = result.skipsAtSec.map(s => s - result.recStartSec);
const intended = runs.filter(([at]) => skipsRec.some(s => at >= s - .2 && at <= s + 3));
const faults = runs.filter(r => !intended.includes(r));
const longest = Math.max(0, ...faults.map(r => r[1])) * 44.1, gaps = faults.length;
const firstSkip = result.skipsAtSec[0] ?? Infinity;
const autoChanges = result.fades.filter(fd => fd.atSec < firstSkip).length;
const summary = {name, style, seconds: +seconds, throttle: +throttle, ...result, autoChanges, skipResults,
  recording: {seconds: +(frames / 44100).toFixed(1), leadingSilenceMs: +(lead / 44.1).toFixed(1), longestSilenceMs: +(longest / 44.1).toFixed(1), gapsOver50ms: gaps,
    manualGapsMs: intended.map(r => Math.round(r[1])),
    peakDbfs: +(20 * Math.log10(peak)).toFixed(2), thresholdDbfs: -80}, problems};
fs.writeFileSync(out + `check-${name}.json`, JSON.stringify(summary, null, 1));
log(JSON.stringify(summary, null, 1));
