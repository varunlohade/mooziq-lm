// Hidden-tab playback check: the player runs in a background tab for a long time, as a
// listener who leaves it playing all day. Always muted, headless.
// Usage: node engine-tools/hidden.mjs <style> <seconds>
// It drives Chrome over the DevTools protocol directly, because under Playwright a page always
// reports 'visible'. A second tab sits in front, so the player's document is 'hidden' and
// Chrome throttles it like any background tab. The master output is recorded and scanned for
// silent gaps, as in engine-tools/check.mjs.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawn, spawnSync} from 'node:child_process';
import {serve} from './serve.mjs';

const [style = 'jazzhop', seconds = '1860'] = process.argv.slice(2);
const name = `hidden-${style}-${seconds}s`;
const out = new URL('../out/', import.meta.url).pathname;
fs.mkdirSync(out, {recursive: true});
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), name, ...a);
const sleep = ms => new Promise(r => setTimeout(r, ms));

const server = await serve(), url = `http://127.0.0.1:${server.address().port}/?style=${style}`;
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'lofi-chrome-'));
const chrome = spawn(process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [`--user-data-dir=${profile}`, '--remote-debugging-port=0',
  '--headless=new', '--mute-audio', '--no-first-run', '--no-default-browser-check', 'about:blank'], {stdio: ['ignore', 'ignore', 'pipe']});
const wsUrl = await new Promise((resolve, reject) => {
  let buf = '';
  chrome.stderr.on('data', d => { buf += d; const m = buf.match(/DevTools listening on (ws:\S+)/); if (m) resolve(m[1]); });
  chrome.on('exit', () => reject(new Error('Chrome exited')));
});
const ws = new WebSocket(wsUrl);
await new Promise(r => { ws.onopen = r; });
let nextId = 0;
const pending = new Map(), problems = [];
ws.onmessage = e => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); return; }
  if (m.method === 'Runtime.exceptionThrown') problems.push('pageerror: ' + m.params.exceptionDetails.exception?.description);
  if (m.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(m.params.type)) problems.push(m.params.type + ': ' + m.params.args.map(a => a.value ?? a.description).join(' '));
  if (m.method === 'Log.entryAdded' && ['error', 'warning'].includes(m.params.entry.level)) problems.push(m.params.entry.level + ': ' + m.params.entry.text);
};
const send = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
  const id = ++nextId;
  pending.set(id, m => (m.error ? reject(new Error(method + ': ' + m.error.message)) : resolve(m.result)));
  ws.send(JSON.stringify({id, method, params, sessionId}));
});
const {targetInfos} = await send('Target.getTargets');
const {sessionId} = await send('Target.attachToTarget', {targetId: targetInfos.find(t => t.type === 'page').targetId, flatten: true});
// Run code in the player page; userGesture lets the Play click start audio.
const run = async (expression, userGesture = false) => {
  const r = await send('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true, userGesture}, sessionId);
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result.value;
};
const until = async (expression, ms = 60000) => { const t = Date.now(); while (!(await run(expression))) { if (Date.now() - t > ms) throw new Error('timeout: ' + expression); await sleep(250); } };
for (const domain of ['Page', 'Runtime', 'Log']) await send(domain + '.enable', {}, sessionId);
await send('Page.addScriptToEvaluateOnNewDocument', {source: `
  window.__csp = [];
  document.addEventListener('securitypolicyviolation', e => window.__csp.push(e.violatedDirective + ' ' + e.blockedURI));
  window.__vis = [];
  document.addEventListener('visibilitychange', () => window.__vis.push([Math.round(performance.now()), document.visibilityState]));`}, sessionId);
await send('Page.navigate', {url}, sessionId);
await until(`document.querySelector('#key')?.textContent !== '—' && !!document.querySelector('#key')`);
await run(`document.querySelector('#play').click()`, true);
await until(`window.lofiDebug.ctx?.state === 'running' && !!window.lofiDebug.report`);
await run(`(() => {
  const {ctx, master} = window.lofiDebug, dest = ctx.createMediaStreamDestination();
  master.connect(dest);
  const rec = new MediaRecorder(dest.stream, {mimeType: 'audio/webm;codecs=opus', audioBitsPerSecond: 160000});
  window.__chunks = [];
  rec.ondataavailable = e => window.__chunks.push(e.data);
  rec.start(1000);
  window.__rec = rec;
  window.__recStart = performance.now();
})()`);
// Put another tab in front: the player is now a background tab.
const {targetId: cover} = await send('Target.createTarget', {url: 'about:blank'});
await send('Target.activateTarget', {targetId: cover});
await until(`document.visibilityState === 'hidden'`, 10000);
log('player tab hidden');

const t0 = Date.now(), timeline = [];
while ((Date.now() - t0) / 1000 < +seconds) {
  await sleep(30000);
  const s = await run(`(() => { const r = window.lofiDebug.report; return {t: Math.round(performance.now() / 1000), visibility: document.visibilityState,
    songs: window.lofiDebug.shownLog.length, underrun: r.underrun, minAhead: +r.minAhead.toFixed(2), steadyMinAhead: +r.steadyMinAhead.toFixed(2),
    state: window.lofiDebug.ctx.state, heapMB: performance.memory ? +(performance.memory.usedJSHeapSize / 1e6).toFixed(1) : null}; })()`);
  timeline.push(s);
  log(JSON.stringify(s));
}
const result = await run(`(async () => {
  const d = window.lofiDebug, rec = window.__rec;
  const done = new Promise(resolve => { rec.onstop = resolve; });
  rec.stop();
  await done;
  window.__blob = new Blob(window.__chunks, {type: 'audio/webm'});
  const ms = d.barMs.slice().sort((a, b) => a - b);
  return {
    bytes: window.__blob.size, recordedWallSec: (performance.now() - window.__recStart) / 1000,
    underrunFrames: d.report.underrun, minAheadSec: d.report.minAhead, steadyMinAheadSec: d.report.steadyMinAhead,
    songs: d.shownLog.length, styles: [...new Set(d.shownLog.map(s => s.style))], fades: d.fades.length,
    barMs: {n: ms.length, median: ms[ms.length >> 1], p95: ms[Math.floor(ms.length * .95)], max: ms.at(-1)},
    csp: window.__csp, visibilityChanges: window.__vis, visibility: document.visibilityState, contextState: d.ctx.state
  };
})()`);
// Fetch the recording in 4 MB pieces (one protocol message per piece).
const parts = [];
for (let at = 0; at < result.bytes; at += 4 << 20) {
  parts.push(Buffer.from(await run(`(async () => { const b = new Uint8Array(await window.__blob.slice(${at}, ${at + (4 << 20)}).arrayBuffer());
    let s = ''; for (let i = 0; i < b.length; i += 32768) s += String.fromCharCode(...b.subarray(i, i + 32768)); return btoa(s); })()`), 'base64'));
}
ws.close();
chrome.kill();
server.close();
await sleep(800);
fs.rmSync(profile, {recursive: true, force: true});

const webmPath = out + `rec-${name}.webm`;
fs.writeFileSync(webmPath, Buffer.concat(parts));
const pcm = spawnSync('ffmpeg', ['-nostdin', '-v', 'error', '-i', webmPath, '-f', 'f32le', '-ac', '2', '-ar', '44100', '-'], {maxBuffer: 2 ** 31});
const f = new Float32Array(pcm.stdout.buffer, pcm.stdout.byteOffset, pcm.stdout.byteLength / 4);
const frames = f.length / 2, threshold = 1e-4;
let silent = 0, peak = 0, lead = 0;
const gaps = [];
// The recorder starts with a short silent warm-up; measure from the first audible sample.
while (lead < frames && Math.max(Math.abs(f[2 * lead]), Math.abs(f[2 * lead + 1])) < threshold) lead++;
for (let i = lead; i < frames; i++) {
  const v = Math.max(Math.abs(f[2 * i]), Math.abs(f[2 * i + 1]));
  peak = Math.max(peak, v);
  if (v < threshold) silent++;
  else { if (silent > 2205) gaps.push([+((i - silent) / 44100).toFixed(1), Math.round(silent / 44.1)]); silent = 0; }
}
if (silent > 2205) gaps.push([+((frames - silent) / 44100).toFixed(1), Math.round(silent / 44.1)]);
const summary = {name, style, seconds: +seconds, ...result, timeline,
  recording: {seconds: +(frames / 44100).toFixed(1), leadingSilenceMs: Math.round(lead / 44.1), gapsOver50ms: gaps.length, gaps: gaps.slice(0, 20),
    peakDbfs: +(20 * Math.log10(peak)).toFixed(2), thresholdDbfs: -80}, problems};
fs.writeFileSync(out + `check-${name}.json`, JSON.stringify(summary, null, 1));
log(JSON.stringify({...summary, timeline: undefined}, null, 1));
process.exit(0);
