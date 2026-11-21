// Accelerated soak test (no audio device): write and render many whole songs in Node with
// the site's own engine and samples, as a long listening session would, and report:
// memory after each song (heap and RSS, after GC), each song's integrated loudness
// (ITU-R BS.1770, gated), the time to write a song, distinct-song fingerprints and styles.
// Usage: node --expose-gc tools/soak.mjs [songs=210] [station=7]
import {learnedModels} from './learned-models.mjs';
import fs from 'node:fs';
import {spawnSync} from 'node:child_process';
import {STYLES, ORDER} from '../site/public/engine/styles/index.js';
import {createArrangedSong} from '../site/public/engine/arranger.js';
import {createRenderer, createBank, addSample, SR} from '../site/public/engine/render.js';
import {random} from '../site/public/engine/neural/random.js';

const songs = +(process.argv[2] || 210), station = +(process.argv[3] || 7);
const root = new URL('../site/public/', import.meta.url).pathname, engines = STYLES;
const meta = JSON.parse(fs.readFileSync(root + 'model/model.json')), bin = fs.readFileSync(root + 'model/model.bin');
const bytes = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), learned = learnedModels();
// Samples, decoded with ffmpeg and trimmed the way engine/loader.js trims them.
const bank = createBank(), index = JSON.parse(fs.readFileSync(root + 'samples/index.json'));
for (const [kind, names] of Object.entries(index)) for (const name of names) {
  const out = spawnSync('ffmpeg', ['-v', 'error', '-i', `${root}samples/${kind}/${name}.m4a`, '-f', 'f32le', '-ac', '2', '-ar', String(SR), '-'], {maxBuffer: 1 << 28}).stdout;
  const all = new Float32Array(out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength)), n = all.length / 2;
  const left = new Float32Array(n), right = new Float32Array(n);
  for (let i = 0; i < n; i++) { left[i] = all[2 * i]; right[i] = all[2 * i + 1]; }
  let peak = 0, start = 0;
  for (const v of left) peak = Math.max(peak, Math.abs(v));
  while (start < n && Math.abs(left[start]) < peak * .01) start++;
  start = Math.max(0, start - 32);
  addSample(bank, {kind, name, left: left.slice(start), right: kind === 'bass' ? null : right.slice(start)});
}
// Ambience loops (rain, cafe, forest, water), untrimmed so they stay seamless.
for (const f of fs.existsSync(root + 'samples/ambience') ? fs.readdirSync(root + 'samples/ambience') : []) {
  const out = spawnSync('ffmpeg', ['-v', 'error', '-i', `${root}samples/ambience/${f}`, '-f', 'f32le', '-ac', '2', '-ar', String(SR), '-'], {maxBuffer: 1 << 28}).stdout;
  const all = new Float32Array(out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength)), n = all.length / 2;
  const left = new Float32Array(n), right = new Float32Array(n);
  for (let i = 0; i < n; i++) { left[i] = all[2 * i]; right[i] = all[2 * i + 1]; }
  addSample(bank, {kind: 'ambience', name: f.replace(/\.m4a$/, ''), left, right});
}
// BS.1770 K-weighting at 44.1 kHz, 400 ms blocks with 75% overlap, -70 LUFS and -10 LU gates.
function lufs(L, R) {
  const k = [[1.53084123, -2.65097009, 1.16916686, -1.66363795, .71279073], [1, -2, 1, -1.98992552, .99000488]];
  const filt = x => { let y = x; for (const [b0, b1, b2, a1, a2] of k) { const o = new Float32Array(y.length); let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < y.length; i++) { const v = b0 * y[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2; x2 = x1; x1 = y[i]; y2 = y1; y1 = v; o[i] = v; } y = o; } return y; };
  const fl = filt(L), fr = filt(R), block = Math.round(.4 * SR), hop = block / 4, powers = [];
  for (let s = 0; s + block <= fl.length; s += hop) { let p = 0; for (let i = s; i < s + block; i++) p += fl[i] ** 2 + fr[i] ** 2; powers.push(p / block); }
  const loud = p => -.691 + 10 * Math.log10(p + 1e-20);
  const abs = powers.filter(p => loud(p) > -70), mean = a => a.reduce((x, y) => x + y, 0) / a.length;
  const rel = abs.filter(p => loud(p) > loud(mean(abs)) - 10);
  return loud(mean(rel));
}
const rows = [], mem = [], fingerprints = {motif: new Map(), progression: new Map(), song: new Map()};
const styleOrder = process.env.STYLES?.split(',') || ORDER;
const t0 = performance.now();
for (let n = 1; n <= songs; n++) {
  const r = random(station ^ Math.imul(n, 0x9e3779b9)); r();
  const seed = 1 + Math.floor(r() * 0x7ffffffe), style = styleOrder[(n - 1) % styleOrder.length];
  const w0 = performance.now(), song = createArrangedSong(engines[style], meta, bytes, seed, style, undefined, undefined, learned), writeMs = performance.now() - w0;
  const render = createRenderer(song, seed, bank, style), frames = Math.round(song.totalBars * 4 * 60 / song.bpm * SR);
  const L = new Float32Array(frames), R = new Float32Array(frames);
  const progression = [];
  for (let at = 0; at < frames;) {
    const bar = render(), k = Math.min(bar.left.length, frames - at);
    L.set(bar.left.subarray(0, k), at); R.set(bar.right.subarray(0, k), at); at += k;
    const info = bar.events.info;
    if (info.material === 'A' && info.chordStart) progression.push(`${info.root}:${info.quality}`);
  }
  const loudness = lufs(L, R);
  const motif = JSON.stringify(song.development.motif ? song.development.occurrences.length + ':' + song.development.motif.notes : null);
  const prog = progression.slice(0, 8).join(' '), full = `${style}|${song.bpm}|${song.shift}|${prog}|${song.plan.sections.map(s => s.type + s.bars).join(',')}`;
  for (const [key, value] of [['progression', prog], ['song', full]]) fingerprints[key].set(value, (fingerprints[key].get(value) || 0) + 1);
  rows.push({n, style, seed, bpm: song.bpm, bars: song.totalBars, minutes: +(frames / SR / 60).toFixed(2), lufs: +loudness.toFixed(2), writeMs: Math.round(writeMs)});
  global.gc?.();
  const m = process.memoryUsage();
  mem.push({n, heapMB: +(m.heapUsed / 1e6).toFixed(1), buffersMB: +(m.arrayBuffers / 1e6).toFixed(1), rssMB: +(m.rss / 1e6).toFixed(1)});
  if (n % 10 === 0) console.error(`song ${n}: ${rows.at(-1).lufs} LUFS, heap ${mem.at(-1).heapMB} MB, rss ${mem.at(-1).rssMB} MB, ${((performance.now() - t0) / 60000).toFixed(1)} min`);
}
const by = s => rows.filter(r => r.style === s), spread = a => Math.max(...a) - Math.min(...a), median = a => [...a].sort((x, y) => x - y)[a.length >> 1];
const p95 = a => [...a].sort((x, y) => x - y)[Math.floor(a.length * .95)];
const summary = {
  songs: rows.length, hoursOfMusic: +(rows.reduce((a, r) => a + r.minutes, 0) / 60).toFixed(1),
  styles: Object.fromEntries(styleOrder.map(s => [s, by(s).length])),
  loudness: {all: {min: Math.min(...rows.map(r => r.lufs)), max: Math.max(...rows.map(r => r.lufs)), median: median(rows.map(r => r.lufs))},
    spreadPerStyle: Object.fromEntries(styleOrder.map(s => [s, +spread(by(s).map(r => r.lufs)).toFixed(2)])),
    within1LUofStyleMedian: Object.fromEntries(styleOrder.map(s => { const m = median(by(s).map(r => r.lufs)); return [s, by(s).filter(r => Math.abs(r.lufs - m) <= 1).length + '/' + by(s).length]; }))},
  writeMs: {median: median(rows.map(r => r.writeMs)), p95: p95(rows.map(r => r.writeMs)), max: Math.max(...rows.map(r => r.writeMs))},
  memory: {heapFirst10: mem.slice(0, 10).map(m => m.heapMB), heapLast10: mem.slice(-10).map(m => m.heapMB), buffersFirst10: mem.slice(0, 10).map(m => m.buffersMB), buffersLast10: mem.slice(-10).map(m => m.buffersMB), rssAt10: mem[9]?.rssMB, rssLast: mem.at(-1).rssMB},
  distinct: {songs: fingerprints.song.size, repeatedSongs: [...fingerprints.song.values()].filter(v => v > 1).length,
    progressions: fingerprints.progression.size, repeatedProgressions: [...fingerprints.progression.values()].filter(v => v > 1).length},
  bpmRanges: Object.fromEntries(styleOrder.map(s => [s, [Math.min(...by(s).map(r => r.bpm)), Math.max(...by(s).map(r => r.bpm))]]))
};
fs.mkdirSync(new URL('../out/', import.meta.url), {recursive:true});
fs.writeFileSync(new URL('../out/soak.json', import.meta.url), JSON.stringify({summary, rows, mem}, null, 1));
console.log(JSON.stringify(summary, null, 1));
