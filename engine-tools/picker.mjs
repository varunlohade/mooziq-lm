// Style picker check (muted, headless): screenshots on desktop and phone, keyboard use,
// the crossfade when the style changes during play, the last style remembered, and the
// phone row scrolling sideways without moving the page.
// Usage: node tools/picker.mjs
import fs from 'node:fs';
import {spawnSync} from 'node:child_process';
import {serve} from './serve.mjs';
import {chromium} from 'playwright-core';
const out = new URL('../out/', import.meta.url).pathname;
fs.mkdirSync(out, {recursive: true});
const server = await serve(), base = `http://127.0.0.1:${server.address().port}/`;
const browser = await chromium.launch({channel: 'chrome', headless: true, args: ['--mute-audio']});
const result = {problems: []};
const ready = page => page.waitForFunction(() => document.querySelector('#key').textContent !== '—');
const pressed = page => page.evaluate(() => document.querySelector('[data-style][aria-pressed=true]')?.dataset.style);
const watch = page => {
  page.on('console', m => { if (['error', 'warning'].includes(m.type())) result.problems.push(m.type() + ': ' + m.text()); });
  page.on('pageerror', e => result.problems.push('pageerror: ' + e.message));
  page.on('requestfailed', r => result.problems.push('requestfailed: ' + r.url()));
};

// Desktop: screenshot, keyboard, change during play, remembered style.
{
  const context = await browser.newContext({viewport: {width: 1440, height: 900}, deviceScaleFactor: 2});
  const page = await context.newPage();
  watch(page);
  await page.goto(base + '?mood=night&style=jazzhop');
  await ready(page);
  await page.click('#play');
  await page.waitForFunction(() => window.lofiDebug.ctx?.state === 'running' && window.lofiDebug.report);
  await page.waitForTimeout(14000);
  await page.screenshot({path: out + 'picker-desktop.png'});
  // Keyboard: focus the chosen style, then arrows, End and Home move focus without choosing.
  await page.focus('[data-style=jazzhop]');
  const keys = [];
  for (const key of ['ArrowRight', 'ArrowRight', 'ArrowLeft', 'End', 'ArrowRight', 'Home', 'ArrowLeft']) {
    await page.keyboard.press(key);
    keys.push([key, await page.evaluate(() => document.activeElement.dataset.style)]);
  }
  result.keyboard = {moves: keys, stillPlaying: await pressed(page)};
  // Record the output, then choose Chillhop with the keyboard (Enter) while playing.
  await page.evaluate(() => {
    const {ctx, master} = window.lofiDebug, dest = ctx.createMediaStreamDestination();
    master.connect(dest);
    const rec = new MediaRecorder(dest.stream, {mimeType: 'audio/webm;codecs=opus', audioBitsPerSecond: 160000});
    window.__chunks = [];
    rec.ondataavailable = e => window.__chunks.push(e.data);
    rec.start(500);
    window.__rec = rec;
    window.__recStart = performance.now();
  });
  await page.waitForTimeout(4000);
  await page.focus('[data-style=chillhop]');
  const before = await page.evaluate(() => ({fades: window.lofiDebug.fades.length, shown: window.lofiDebug.shownLog.length}));
  const pressAt = await page.evaluate(() => performance.now());
  await page.keyboard.press('Enter');
  // A second choice during the handoff should become the active style.
  await page.waitForTimeout(300);
  await page.click('[data-style=sleep]');
  await page.waitForFunction(n => window.lofiDebug.shownLog.length > n, before.shown, {timeout: 30000});
  await page.waitForTimeout(6000);
  const change = await page.evaluate(async ([n, at]) => {
    const d = window.lofiDebug, rec = window.__rec;
    const done = new Promise(r => { rec.onstop = r; });
    rec.stop();
    await done;
    const bytes = new Uint8Array(await new Blob(window.__chunks).arrayBuffer());
    let s = '';
    for (let i = 0; i < bytes.length; i += 32768) s += String.fromCharCode(...bytes.subarray(i, i + 32768));
    return {webm: btoa(s), pressSec: (at - window.__recStart) / 1000, newFades: d.fades.length - n.fades, newSong: d.shownLog.at(-1).style,
      shownMs: Math.round(d.shownLog.at(-1).at - at), chosen: document.querySelector('[data-style][aria-pressed=true]').dataset.style,
      stored: localStorage.getItem('lofi-style')};
  }, [before, pressAt]);
  const webm = out + 'rec-picker.webm';
  fs.writeFileSync(webm, Buffer.from(change.webm, 'base64'));
  delete change.webm;
  const pcm = spawnSync('ffmpeg', ['-nostdin', '-v', 'error', '-i', webm, '-f', 'f32le', '-ac', '2', '-ar', '44100', '-'], {maxBuffer: 1 << 30}).stdout;
  const f = new Float32Array(pcm.buffer, pcm.byteOffset, pcm.byteLength / 4), frames = f.length / 2;
  let silent = 0, lead = 0;
  const gaps = [];
  while (lead < frames && Math.max(Math.abs(f[2 * lead]), Math.abs(f[2 * lead + 1])) < 1e-4) lead++;
  for (let i = lead; i < frames; i++) {
    if (Math.max(Math.abs(f[2 * i]), Math.abs(f[2 * i + 1])) < 1e-4) silent++;
    else { if (silent > 2205) gaps.push({atSec: +((i - silent) / 44100).toFixed(2), ms: Math.round(silent / 44.1)}); silent = 0; }
  }
  result.change = {...change, gaps};
  // Reload with no style in the address: the last choice comes back.
  await page.goto(base + '?mood=night');
  await ready(page);
  result.remembered = await pressed(page);
  await context.close();
}

// Phone: the style buttons wrap, and a selection stays inside the viewport.
{
  const context = await browser.newContext({viewport: {width: 390, height: 844}, deviceScaleFactor: 2, isMobile: true, hasTouch: true});
  const page = await context.newPage();
  watch(page);
  await page.goto(base + '?mood=day&style=jazzhop');
  await ready(page);
  await page.click('#play');
  await page.waitForTimeout(14000);
  await page.screenshot({path: out + 'picker-phone.png'});
  const row = () => page.evaluate(() => { const r = document.querySelector('#styles'); return {scrollLeft: Math.round(r.scrollLeft), scrollWidth: r.scrollWidth, clientWidth: r.clientWidth, pageY: scrollY}; });
  const start = await row();
  await page.evaluate(() => { const r = document.querySelector('#styles'); r.scrollLeft = r.scrollWidth; });
  await page.waitForTimeout(300);
  await page.tap('[data-style=nature]');
  await page.waitForTimeout(1500);
  const after = await row();
  const box = await page.evaluate(() => { const r = document.querySelector('#styles').getBoundingClientRect(), b = document.querySelector('[data-style=nature]').getBoundingClientRect();
    return {rowLeft: Math.round(r.left), rowRight: Math.round(r.right), left: Math.round(b.left), right: Math.round(b.right)}; });
  await page.waitForTimeout(9000);
  await page.screenshot({path: out + 'picker-phone-nature.png'});
  // Touch targets: every style button is at least 44 px tall.
  const heights = await page.evaluate(() => [...document.querySelectorAll('[data-style]')].map(b => Math.round(b.getBoundingClientRect().height)));
  result.phone = {start, after, natureBox: box, chosen: await pressed(page), minButtonHeight: Math.min(...heights),
    pageWidth: await page.evaluate(() => document.documentElement.scrollWidth)};
  await context.close();
}
await browser.close();
server.close();
fs.writeFileSync(out + 'picker.json', JSON.stringify(result, null, 1));
console.log(JSON.stringify(result, null, 1));
