// Compares candidate models inside the real lofi composers (Node, no audio).
// For each candidate: shipped size, int8/int4 dequant time, per-bar time of model + composer
// (median and p95 per style), and the clash audit over a 60 s song per style.
// Usage: node engine-tools/candidates.mjs <spec> [spec ...] > out.json
//   live                          the shipped weights (public/model)
//   web:<dir>                     a model.bin + model.json folder (e.g. exported npz candidates)
//   q:<path without .bin/.json>   a tinylofi-q8 / tinylofi-q4 student (dequantised with qload.js)
import fs from 'node:fs';
import path from 'node:path';
import * as flow from '../site/public/engine/flow/compose.js';
import * as steady from '../site/public/engine/steady/music.js';
import * as jazzhop from '../site/public/engine/jazzhop/compose.js';
import {padEvents} from '../site/public/engine/pad.js';
import {RELEASE} from '../site/public/engine/render.js';

const root = new URL('..', import.meta.url).pathname;
const engines = {jazzhop, steady, flow}, STYLE_TOKEN = {jazzhop: 'JAZZHOP', steady: 'STEADY', flow: 'FLOW'};
const asBuffer = b => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
export async function loadCandidate(spec) {
  if (spec === 'live') {
    const bin = fs.readFileSync(root + 'site/public/model/model.bin');
    return {name: 'LIVE', meta: JSON.parse(fs.readFileSync(root + 'site/public/model/model.json')), bytes: asBuffer(bin), shippedBytes: bin.length, dequantMs: 0};
  }
  if (spec.startsWith('web:')) {
    const dir = spec.slice(4), bin = fs.readFileSync(dir + '/model.bin');
    return {name: path.basename(dir), meta: JSON.parse(fs.readFileSync(dir + '/model.json')), bytes: asBuffer(bin), shippedBytes: bin.length, dequantMs: 0};
  }
  const base = spec.slice(2), bin = fs.readFileSync(base + '.bin'), metaQ = JSON.parse(fs.readFileSync(base + '.json'));
  const {dequantToF16} = await import(root + 'models/qload.js');
  const t0 = performance.now(), {meta, f16buf} = dequantToF16(metaQ, asBuffer(bin)), dequantMs = performance.now() - t0;
  return {name: path.basename(base), meta, bytes: f16buf, shippedBytes: bin.length + fs.statSync(base + '.json').size, dequantMs};
}
// The live composers prompt with meta.LOFI. Candidates that know style tokens get their own style's token there.
export const styleMeta = (meta, style) => ({...meta, LOFI: meta[STYLE_TOKEN[style]] ?? meta.LOFI});

const pct = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
function timing(candidate, style, bars = +(process.env.BARS || 48)) {
  const song = engines[style].createSong(styleMeta(candidate.meta, style), candidate.bytes, 1), ms = [];
  for (let i = 0; i < bars; i++) { const t = performance.now(); song.nextBar(); ms.push(performance.now() - t); }
  return {median: +pct(ms, .5).toFixed(1), p95: +pct(ms, .95).toFixed(1), max: +Math.max(...ms).toFixed(1)};
}
// Clash audit (logic of engine-tools/audit.mjs) plus a few melody statistics, for one song.
function audit(candidate, style, seed, seconds = 60) {
  const rate = 44100, {keys, felt, glass, bass, pad, drone} = RELEASE;
  const tails = {steady: {felt, keys, bass, pad}, jazzhop: {felt, keys, bass}, flow: {felt, keys, pad, drone}}[style];
  const song = engines[style].createSong(styleMeta(candidate.meta, style), candidate.bytes, seed), beat = 60 / song.bpm, frames = seconds * rate, notes = [];
  let strong = 0, matches = 0, bar = 0, previous = [], lead = 0;
  const chords = new Set();
  for (; Math.round(bar * 4 * beat * rate) < frames; bar++) {
    const origin = Math.round(bar * 4 * beat * rate), events = song.nextBar();
    for (const c of events.info.chords || [[events.info.root, events.info.quality]]) chords.add(c.join(':'));
    const pads = style === 'flow' ? (events.info.chordStart ? padEvents(events.info, song.shift, song.bpm, previous, 8) : [])
      : style === 'steady' ? padEvents(events.info, song.shift, song.bpm, previous) : [];
    previous = events;
    for (const event of events.concat(pads)) {
      if (!(event.type in tails)) continue;
      const start = origin + Math.round(event.beat * beat * rate);
      if (start >= frames) continue;
      notes.push({...event, start, end: Math.min(frames, start + Math.ceil((event.duration * beat + (event.releaseSec ?? tails[event.type])) * rate))});
      if (event.type === 'felt') {
        lead++;
        if (event.strong ?? event.slot % 2 === 0) { strong++; if ((event.stable ?? events.info.stable).includes(event.pitch % 12)) matches++; }
      }
    }
  }
  const union = spans => { let length = 0, end = 0; for (const [a, b] of spans.sort((x, y) => x[0] - y[0])) { length += Math.max(0, b - Math.max(a, end)); end = Math.max(end, b); } return length; };
  const overlap = [], clashes = [];
  notes.sort((a, b) => a.start - b.start);
  for (const [i, a] of notes.entries()) for (let j = i + 1; j < notes.length && notes[j].start < a.end; j++) {
    const b = notes[j], end = Math.min(a.end, b.end);
    if ((a.type === 'felt') === (b.type === 'felt')) continue;
    overlap.push([b.start, end]);
    if (Math.abs(a.pitch - b.pitch) % 12 === 1) clashes.push([b.start, end]);
  }
  return {clashPercent: +(100 * union(clashes) / Math.max(1, union(overlap))).toFixed(2), strongBeat: strong ? +(100 * matches / strong).toFixed(1) : null,
    leadNotesPerBar: +(lead / bar).toFixed(2), distinctChords: chords.size, bpm: song.bpm};
}
if (import.meta.url === `file://${process.argv[1]}`) {
  const rows = [];
  for (const spec of process.argv.slice(2)) {
    const c = await loadCandidate(spec), row = {name: c.name, params: c.meta.params ?? null, shippedMB: +(c.shippedBytes / 1e6).toFixed(2), dequantMs: Math.round(c.dequantMs)};
    for (const style of ['jazzhop', 'steady', 'flow']) row[style] = {perBarMs: timing(c, style), audit: audit(c, style, 1)};
    const worst = Math.max(...['jazzhop', 'steady', 'flow'].map(s => row[s].perBarMs.p95));
    row.p95WorstMs = worst;
    row.tooSlowForPhones = worst > 60;
    rows.push(row);
    console.error(JSON.stringify(row));
  }
  console.log(JSON.stringify(rows, null, 1));
}
