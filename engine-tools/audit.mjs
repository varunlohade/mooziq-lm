// Symbolic clash audit over whole planned songs,
// every arranged part included, written by the same engine modules the site runs.
// Usage: node engine-tools/audit.mjs [seed...]
import fs from 'node:fs';
import {webcrypto} from 'node:crypto';
import {STYLES, ORDER} from '../site/public/engine/styles/index.js';
import {RELEASE} from '../site/public/engine/render.js';
import {createArrangedSong} from '../site/public/engine/arranger.js';

const rate = 44100;
const meta = JSON.parse(fs.readFileSync(new URL('../site/public/model/model.json', import.meta.url)));
const bin = fs.readFileSync(new URL('../site/public/model/model.bin', import.meta.url));
const bytes = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
// Direct-voice release lengths in each sound.js. The steady felt voice can carry its own releaseSec.
// Release tails as the band plays them (engine/render.js). Felt notes can carry their own releaseSec.
// Every pitched part: the lead ('felt', its harmony included) against all accompaniment.
const tail = {felt: RELEASE.felt, keys: RELEASE.keys, counter: RELEASE.counter, arp: RELEASE.arp, swell: RELEASE.swell,
  bass: RELEASE.bass, drone: RELEASE.drone, pad: RELEASE.pad};
const engines = STYLES;
function unionLength(spans) {
  let length = 0, end = 0;
  for (const [a, b] of spans.sort((x, y) => x[0] - y[0])) {
    length += Math.max(0, b - Math.max(a, end));
    end = Math.max(end, b);
  }
  return length;
}
function score(style, seed) {
  const song = createArrangedSong(engines[style], meta, bytes, seed, style);
  const beat = 60 / song.bpm, frames = song.totalBars * 4 * beat * rate, notes = [];
  let strong = 0, matches = 0, bar = 0;
  for (; bar < song.totalBars; bar++) {
    const origin = Math.round(bar * 4 * beat * rate), events = song.nextBar();
    for (const event of events) {
      if (!(event.type in tail)) continue;
      const start = origin + Math.round(event.beat * beat * rate);
      if (start >= frames) continue;
      const end = Math.min(frames, start + Math.ceil((event.duration * beat + (event.releaseSec ?? tail[event.type])) * rate));
      notes.push({...event, start, end});
      // Jazz Hop marks its strong notes (on the beat) and carries each note's own chord.
      const onStrong = event.strong ?? event.slot % 2 === 0;
      if (event.type === 'felt' && onStrong) {
        strong++;
        if ((event.stable ?? events.info.stable).includes(event.pitch % 12)) matches++;
      }
    }
  }
  const overlap = [], clashes = [];
  notes.sort((a, b) => a.start - b.start);
  for (const [i, a] of notes.entries()) {
    for (let j = i + 1; j < notes.length && notes[j].start < a.end; j++) {
      const b = notes[j], end = Math.min(a.end, b.end);
      if ((a.type === 'felt') === (b.type === 'felt')) continue;
      overlap.push([b.start, end]);
      if (Math.abs(a.pitch - b.pitch) % 12 === 1) clashes.push([b.start, end]);
    }
  }
  const lead = notes.filter(n => n.type === 'felt').length;
  const clashPercent = 100 * unionLength(clashes) / unionLength(overlap);
  return {style, seed, bpm: song.bpm, shift: song.shift, bars: bar, leadNotes: lead,
    overlapSeconds: +(unionLength(overlap) / rate).toFixed(2), clashSeconds: +(unionLength(clashes) / rate).toFixed(2),
    clashPercent: +clashPercent.toFixed(3), strongBeatFit: `${matches}/${strong}`, strongBeatPercent: +(100 * matches / strong).toFixed(1),
    pass: clashPercent < 2 && matches / strong > .9};
}
const seeds = process.argv.slice(2).map(Number);
while (seeds.length < 3) seeds.push(1 + webcrypto.getRandomValues(new Uint32Array(1))[0] % 0x7ffffffe);
const rows = [];
for (const style of (process.env.STYLES?.split(',') || ORDER)) for (const seed of seeds) rows.push(score(style, seed));
console.table(rows);
fs.mkdirSync(new URL('../out/', import.meta.url), {recursive: true});
fs.writeFileSync(new URL('../out/audit.json', import.meta.url), JSON.stringify(rows, null, 1));
if (!rows.every(r => r.pass)) process.exitCode = 1;
