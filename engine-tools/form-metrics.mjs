// Song-form metrics from the engine's own events (no audio), across styles and seeds.
// Usage: node engine-tools/form-metrics.mjs [raw|arranged] [seeds=5]
//   raw       the composers on their own (the site before song plans), 72 bars
//   arranged  full songs from planSong() through the arranger
// Per song: the longest stretch with no audible change, changes per minute, the largest
// gap between big moments (section change, build, drop, tape stop), distinct progressions,
// active layers (min/median/max), ear-candy events per minute.
import fs from 'node:fs';
import {STYLES, ORDER} from '../site/public/engine/styles/index.js';
import {createArrangedSong} from '../site/public/engine/arranger.js';

const mode = process.argv[2] || 'arranged', nSeeds = +(process.argv[3] || 5);
const engines = STYLES;
const meta = JSON.parse(fs.readFileSync(new URL('../site/public/model/model.json', import.meta.url)));
const bin = fs.readFileSync(new URL('../site/public/model/model.bin', import.meta.url));
const bytes = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
const LAYER = {keys: 'keys', felt: 'lead', glass: 'lead', bass: 'bass', drone: 'bass', kick: 'drums', brush: 'drums', snare: 'drums', rim: 'drums', hat: 'drums',
  counter: 'counter', pad: 'pad', shaker: 'perc', texture: 'texture', swell: 'fx', rumble: 'fx', riser: 'fx', arp: 'fx'};
const CANDY = new Set(['swell', 'rumble', 'riser', 'arp']);
const median = a => [...a].sort((x, y) => x - y)[a.length >> 1];
const key = list => JSON.stringify(list);
function sigs(events) {
  const pick = (types, f) => key(events.filter(e => types.includes(e.type)).map(f).sort());
  return {
    drums: pick(['kick', 'brush', 'snare', 'rim', 'hat', 'shaker'], e => `${e.type}@${Math.round(e.beat * 8)}`),
    bass: pick(['bass', 'drone'], e => `${e.pitch}@${Math.round(e.beat * 8)}/${Math.round(e.duration * 4)}`),
    lead: pick(['felt', 'glass'], e => `${e.pitch}@${Math.round(e.beat * 8)}:${e.inst || ''}`),
    keys: pick(['keys', 'counter'], e => `${e.pitch}@${Math.round(e.beat * 8)}`),
    chords: key(events.info.chords || [[events.info.root, events.info.quality]])
  };
}
function measure(style, seed) {
  const engine = engines[style];
  const song = mode === 'raw' ? engine.createSong(meta, bytes, seed) : createArrangedSong(engine, meta, bytes, seed, style);
  const barSec = 4 * 60 / song.bpm, total = mode === 'raw' ? 72 : song.totalBars;
  const bars = [], all = [];
  for (let i = 0; i < total; i++) {
    const events = song.nextBar(), info = events.info;
    all.push(events);
    bars.push({sig: sigs(events), layers: [...new Set(events.map(e => LAYER[e.type]).filter(Boolean))].sort(),
      filter: info.filter ?? 20000, gain: info.gain ?? 1, section: info.section ?? 0,
      big: events.some(e => e.type === 'riser') || info.tapeStop != null, candy: events.filter(e => e.candy || CANDY.has(e.type)).length,
      chords: sigs(events).chords});
  }
  let run = 0, longest = 0, changed = 0, sinceBig = 0, bigGap = 0;
  for (let i = 0; i < bars.length; i++) {
    const b = bars[i], kinds = [];
    if (i > 0) {
      const p = bars[i - 1];
      if (key(b.layers) !== key(p.layers)) kinds.push('layer');
      if (b.section !== p.section) kinds.push('section');
      const back = bars[i - 4];
      if (back && (Math.abs(Math.log2(b.filter / back.filter)) >= .5 || Math.abs(b.gain - back.gain) >= .15)) kinds.push('energy');
      for (const part of ['drums', 'bass', 'lead', 'keys', 'chords'])
        if (i >= 4 && b.sig[part] !== bars[i - 4].sig[part] && (i < 8 || b.sig[part] !== bars[i - 8].sig[part])) kinds.push(part);
      if (b.candy) kinds.push('candy');
    }
    if (kinds.length) { changed++; longest = Math.max(longest, run); run = 0; } else run++;
    const big = i > 0 && (b.section !== bars[i - 1].section || b.big);
    if (big) { bigGap = Math.max(bigGap, sinceBig); sinceBig = 0; } else sinceBig++;
  }
  longest = Math.max(longest, run); bigGap = Math.max(bigGap, sinceBig);
  const phrases = new Set();
  for (let i = 0; i + 4 <= bars.length; i += 4) phrases.add(bars.slice(i, i + 4).map(b => b.chords).join('|'));
  const minutes = bars.length * barSec / 60, layerCounts = bars.map(b => b.layers.length);
  // Song development, checked from the notes: each motif occurrence must share most of its
  // onsets with the original (allowing an eighth-note shift), and kinds must differ.
  let development = null;
  if (mode !== 'raw') {
    const lead = i => [0, 1].flatMap(b => (all[i + b] || []).filter(e => e.type === 'felt' && !e.harmony).map(e => e.beat + 4 * b));
    const occ = song.development.occurrences, origin = occ.find(o => o.kind === 'original');
    const onsets = origin ? lead(origin.bar) : [];
    const verified = occ.filter(o => {
      const mine = lead(o.bar);
      return mine.length >= 2 && Math.max(...[0, .5].map(d => mine.filter(t => onsets.some(u => Math.abs(u + d - t) < .1)).length / mine.length)) >= .6;
    });
    const mode2 = song.development.ending, last = all.at(-1);
    const tonicChord = (mode2.mode === 'major' ? [0, 4, 7, 11] : [0, 3, 7, 10]).map(d => (mode2.tonicPc + d) % 12);
    const lastPcs = [...new Set(last.filter(e => e.type === 'keys' || e.type === 'pad').map(e => e.pitch % 12))];
    const energies = song.plan.sections.map(s => Math.max(...s.energy)), peak = Math.max(...energies);
    const second = [...energies].sort((a, b) => b - a)[1];
    development = {motifTimes: verified.length, motifKinds: [...new Set(verified.map(o => o.kind))].length,
      kinds: verified.map(o => o.kind), peakMargin: +(peak - second).toFixed(2), peakSection: song.plan.sections[energies.indexOf(peak)].type,
      endsOnTonic: lastPcs.length > 0 && lastPcs.every(pc => tonicChord.includes(pc)) && tonicChord.slice(0, 3).every(pc => lastPcs.includes(pc))};
  }
  return {style, seed, bars: bars.length, minutes: +minutes.toFixed(2), longestNoChangeBars: longest, changesPerMin: +(changed / minutes).toFixed(1),
    maxBarsBetweenBigMoments: bigGap, distinctProgressions: phrases.size,
    layers: {min: Math.min(...layerCounts), median: median(layerCounts), max: Math.max(...layerCounts)},
    candyPerMin: +(bars.reduce((a, b) => a + b.candy, 0) / minutes).toFixed(1),
    layerCountChanges: bars.slice(1).filter((b, i) => b.layers.length !== bars[i].layers.length).length,
    sections: mode === 'raw' ? null : song.plan.sections.map(s => `${s.type}:${s.bars}`).join(' '), development};
}
const rows = [];
for (const style of (process.env.STYLES?.split(',') || ORDER)) for (let k = 0; k < nSeeds; k++) rows.push(measure(style, 1000003 * (k + 1) + 17));
console.table(rows.map(({sections, layers, ...r}) => ({...r, layers: `${layers.min}/${layers.median}/${layers.max}`})));
if (mode !== 'raw') for (const r of rows) console.log(r.style, r.seed, r.sections, JSON.stringify(r.development));
fs.writeFileSync(new URL(`../out/form-metrics-${mode}.json`, import.meta.url), JSON.stringify(rows, null, 1));
