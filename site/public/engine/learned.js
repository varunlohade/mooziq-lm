// The learned song path: a planner model
// writes the song's sections, and the song composer model (run by engine/slow.js) writes
// chords, melody and bass bar by bar under the same calm rules as the preset path.
// Both models are in public/model/learned/ (see their .json files for the vocabulary).
//
// Guard rails, at decode time:
//  - each planner slot is masked to its own token range; an outro ends the song;
//  - styles without drums never get a drum track (the training labels marked drums in
//    some of them by mistake);
//  - background rules: energy stays in the calm range with one gentle peak, the song lasts
//    about 3 to 4.5 minutes, tempo and key come from the style as before.
import {loadModel} from './neural/transformer.js';
import {random} from './neural/random.js';
import {PRESETS} from './plan.js';

const SONG_STYLE = {rainycafe: 'rainy_cafe', studypiano: 'study_piano'};
// The calm rules' rest bias was set on the shipped model; the song composer rests more in
// Flow, so its rest bias moves to match the preset path's melody density.
export const REST_SHIFT = {flow: -.8};
const pick = (rng, options) => {
  const total = options.reduce((a, [, w]) => a + w, 0);
  let v = rng() * total;
  for (const [value, w] of options) if ((v -= w) < 0) return value;
  return options.at(-1)[0];
};

// Sample a raw plan from the planner: [{type, len, energy (0-3), act (4 bits), transition}].
export function samplePlan(planner, style, seed, mode) {
  const {meta, bytes} = planner, model = loadModel(meta, bytes), draw = random((seed ^ 0x91a77e5) >>> 0);
  const beat = (PRESETS[style] || PRESETS.steady).beat;
  model.reset();
  let logits;
  for (const t of [meta.BOS, meta.STY0 + meta.STYLES.indexOf(SONG_STYLE[style] || style), mode === 'major' ? meta.MAJ : meta.MIN]) logits = model.step(t);
  const ranges = [[meta.TYPE0, meta.LEN0], [meta.LEN0, meta.EN0], [meta.EN0, meta.ACT0], [meta.ACT0, meta.TR0], [meta.TR0, meta.V]];
  const sections = [];
  let current = [], ended = false;
  for (let p = 3; p < meta.T - 1 && !ended; p++) {
    const slot = (p - 3) % meta.SLOT, [lo, hi] = ranges[slot];
    let ids = [];
    for (let t = lo; t < hi; t++) ids.push(t);
    if (slot === 0) {
      ids.push(meta.EOS);
      // After an outro, or after nine sections, the song ends.
      if (sections.at(-1)?.type === 'outro' || sections.length >= 9) ids = [meta.EOS];
      // The first section is never the end.
      if (!sections.length) ids = ids.filter(t => t !== meta.EOS);
    }
    const max = Math.max(...ids.map(t => logits[t]));
    let weights = ids.map(t => Math.exp(logits[t] - max));
    // Drumless styles: drums are taken out of each active set, and that set's probability
    // goes to the same set without drums (so the rest of the arrangement keeps its odds).
    if (slot === 3 && !beat) {
      weights = ids.map((t, i) => (t - meta.ACT0) & 1 ? 0 : weights[i] + weights[ids.indexOf(t + 1)]);
    }
    let v = draw() * weights.reduce((a, b) => a + b, 0), tok = ids.at(-1);
    for (let i = 0; i < ids.length; i++) if ((v -= weights[i]) < 0) { tok = ids[i]; break; }
    if (tok === meta.EOS) { ended = true; break; }
    current.push(tok);
    if (slot === 4) {
      const [ty, le, en, ac, tr] = current;
      sections.push({type: meta.TYPES[ty - meta.TYPE0], len: meta.LENS[le - meta.LEN0], energy: en - meta.EN0, act: ac - meta.ACT0, transition: meta.TRANS[tr - meta.TR0]});
      current = [];
    }
    logits = model.step(tok);
  }
  return sections;
}

// Our section names for the planner's: grooves are the first theme (A, then A2), hooks the
// second (B, then B2), builds the bridge.
const OURS = {intro: 'intro', groove: 'A', build: 'bridge', hook: 'B', break: 'break', outro: 'outro'};
const MATERIAL = {intro: 'A', A: 'A', A2: 'A', B: 'B', B2: 'B', break: 'A', bridge: 'C', outro: 'A'};
const ENERGY = [.42, .52, .6, .66];

// A full song plan (same shape as plan.js planSong) from the planner's sections.
export function learnedPlan(planner, style, seed, {bpm, mode}) {
  const preset = PRESETS[style] || PRESETS.steady, beat = preset.beat, rng = random((seed ^ 0x1ea47ed) >>> 0);
  const raw = samplePlan(planner, style, seed, mode);
  if (!raw.length || raw[0].type !== 'intro') {
    // Sometimes enter directly on the model's theme. When we add an intro,
    // let a melody or bass pickup introduce the harmony as well as solo keys.
    const opening = random(seed ^ 0x4f70656e);
    const startOnTheme = ['groove', 'hook'].includes(raw[0]?.type) && opening() < .5;
    if (!startOnTheme) {
      const tracks = [[4, 1], [12, 2]];
      if (!preset.solo) tracks.push([6, 1]);
      if (beat) tracks.push([13, 1]);
      const act = pick(opening, tracks);
      raw.unshift({type: 'intro', len: 4, energy: 0, act, transition: 'start'});
    }
  }
  if (raw.at(-1).type !== 'outro') raw.push({type: 'outro', len: 4, energy: 0, act: 4, transition: 'down'});
  const seen = {};
  const sections = raw.map(r => {
    // A long build (more than 8 bars) is a body section of its own, so it is a theme here;
    // short builds are bridges (which stay at 8 bars or less, as on the preset path).
    let type = r.type === 'build' && r.len > 8 ? 'A' : OURS[r.type];
    if (type === 'A' || type === 'B') { type = seen[type] ? type + '2' : type; seen[type[0]] = true; }
    return {type, bars: r.len, raw: r};
  });
  // Background length: about 3 to 4.5 minutes. Body sections grow or shrink in 4-bar steps
  // (up to 32 bars, later split into two phrases); breaks and bridges stay at 8 bars or less.
  const barSec = 4 * 60 / bpm, target = 190 + 70 * rng(), total = () => sections.reduce((a, s) => a + s.bars, 0);
  // Intros, outros, breaks and bridges stay at 8 bars or less, as on the preset path.
  for (const s of sections) if (['break', 'bridge', 'intro', 'outro'].includes(s.type) && s.bars > 8) s.bars = 8;
  for (let guard = 0; guard < 60 && Math.abs(total() * barSec - target) > 2 * barSec; guard++) {
    const grow = total() * barSec < target, max = x => x.type === 'break' || x.type === 'bridge' ? 8 : 32;
    const body = sections.filter(x => x.type !== 'intro' && x.type !== 'outro' && (grow ? x.bars < max(x) : x.bars > 4));
    if (!body.length) {
      if (!grow) break;
      // A short sampled plan can run out of expandable themes. Give it a
      // contrasting phrase before the outro, then continue fitting the duration.
      const type = sections.some(s => s.type.startsWith('A')) ? 'B' : 'A';
      sections.splice(-1, 0, {type, bars: 4, raw: {
        type: type === 'A' ? 'groove' : 'hook', energy: 2,
        act: preset.solo ? 12 : beat ? 15 : 14, transition: 'up'
      }});
      continue;
    }
    body[Math.floor(rng() * body.length)].bars += grow ? 4 : -4;
  }
  // A long theme (16 bars or more) is two phrases: the second is its '2' form (A2, B2), where
  // the motif comes back varied (engine/develop.js).
  for (let i = sections.length - 1; i >= 0; i--) {
    const s = sections[i];
    if ((s.type === 'A' || s.type === 'B') && s.bars >= 16) {
      const first = 4 * Math.round(s.bars / 8);
      sections.splice(i + 1, 0, {type: s.type + '2', bars: s.bars - first, raw: {...s.raw, transition: 'same'}});
      s.bars = first;
    }
  }
  const leadFor = () => pick(rng, preset.alt), mainLead = pick(rng, preset.main);
  const altLead = (() => { let l, k = 0; do l = leadFor(); while (l === mainLead && rng() < .8 && ++k < 20); return l; })();
  let start = 0;
  for (const [i, s] of sections.entries()) {
    const r = s.raw, e0 = Math.min(.66, ENERGY[r.energy] + (rng() - .5) * .04);
    const e1 = s.type === 'outro' ? .25 : s.type === 'intro' || s.type === 'break' ? e0 + .05 : Math.min(.66, e0 + .03);
    s.energy = [+e0.toFixed(2), +e1.toFixed(2)];
    s.material = MATERIAL[s.type];
    s.start = start; start += s.bars;
    const D = r.act & 1, B = r.act & 2, L = r.act & 8, t = {};
    // Keys carry every style's harmony, so they always play (the planner's K bit is ignored).
    t.keys = {inst: 'piano', voicing: pick(rng, [['close', 3], ['spread', 2], ['low', 1]])};
    if (D && beat) t.drums = {pattern: e0 >= .6 ? pick(rng, [['full', 4], ['busy', 1]]) : pick(rng, [['full', 2], ['half', 2], ['sparse', 1]])};
    if (B) t.bass = {pattern: 'composer'};
    if (L) t.lead = {inst: s.type[0] === 'B' || s.type === 'bridge' ? altLead : mainLead};
    if ((s.type === 'B' || s.type === 'B2') && t.lead && rng() < .6) t.counter = {inst: pick(rng, [['piano', 2], ['vibes', 1]]), kind: 'guide'};
    if (s.type === 'A2' && t.lead && rng() < .5) t.counter = {inst: t.lead.inst, kind: 'harmony'};
    if (s.type === 'break' || s.type === 'outro' || (!beat && rng() < .7) || (s.type[0] === 'B' && rng() < .5)) t.pad = {inst: 'pad'};
    if (t.drums && e0 >= .6 && !preset.light && rng() < .5) t.perc = {pattern: 'eighths'};
    if (preset.textures.length && (preset.ambience >= 1 || (s.type === 'intro' || s.type === 'break' || s.type === 'outro' ? rng() < .7 : rng() < preset.ambience)))
      t.texture = {level: +((.6 + .4 * rng()) * (preset.gain || 1)).toFixed(2), kinds: preset.textures};
    if (preset.solo) { for (const k of Object.keys(t)) if (k !== 'keys' && k !== 'lead') delete t[k]; t.keys ||= {inst: 'piano', voicing: 'close'}; }
    if (preset.quiet) delete t.counter;
    s.tracks = t;
    const cut = x => Math.round(1800 + 14000 * x * x);
    s.filter = s.type === 'intro' ? [1600, cut(e1)] : s.type === 'break' ? [cut(.55), 2200] : s.type === 'outro' ? [cut(e0), 2000] : [cut(e0), cut(e1)];
    // How the section ends follows how the planner says the next one arrives.
    const next = sections[i + 1]?.raw;
    s.out = !next ? null
      : next.type === 'outro' ? pick(rng, [['tapestop', .5], ['fill', 2], [null, 3], ['swell', 2]])
      : next.transition === 'up' ? pick(rng, [['build', 1], ['swell', 2], ['fill', 2]])
      : next.transition === 'drop' ? 'drop'
      : next.transition === 'down' ? pick(rng, [['swell', 2], [null, 2]])
      : pick(rng, [['fill', 2], [null, 3]]);
    if (!beat && (s.out === 'fill' || s.out === 'drop' || s.out === 'cut' || s.out === 'build')) s.out = pick(rng, [['swell', 2], [null, 1]]);
  }
  // One gentle peak: the strongest body section (latest on a tie) gets every layer.
  const main = sections.filter(s => !['intro', 'outro', 'break', 'bridge'].includes(s.type));
  const climax = main.length ? main.reduce((a, b) => (b.raw.energy > a.raw.energy || (b.raw.energy === a.raw.energy && b.start > a.start) ? b : a), main[0]) : null;
  for (const s of sections) if (s !== climax) s.energy = s.energy.map(e => Math.min(e, .66));
  if (climax) {
    climax.climax = true;
    climax.energy = [.75, .8];
    if (!preset.solo) Object.assign(climax.tracks, {
      ...(beat ? {drums: {pattern: 'full'}, ...(preset.light ? {} : {perc: {pattern: 'eighths'}})} : {}),
      keys: climax.tracks.keys || {inst: 'piano', voicing: 'spread'}, bass: climax.tracks.bass || {pattern: 'composer'},
      lead: climax.tracks.lead || {inst: mainLead}, pad: {inst: 'pad'}
    });
    else climax.tracks.lead ||= {inst: mainLead};
    climax.filter = [12000, 14000];
  }
  // The composer is told each section's type, energy and final set of tracks.
  for (const s of sections) {
    const t = s.tracks;
    s.learned = {sec: planner.meta.TYPES.indexOf(s.raw.type) * 4 + (s.climax ? 3 : s.raw.energy), act: (t.drums ? 1 : 0) | (t.bass ? 2 : 0) | (t.keys ? 4 : 0) | (t.lead ? 8 : 0)};
    delete s.raw;
  }
  const materials = {A: {seedOffset: 0}, B: {seedOffset: 1}, C: {seedOffset: 2}};
  const candy = {rate: beat ? .6 : .45, kinds: beat ? ['arp', 'rumble', 'ghost', 'swell'] : preset.solo ? ['arp', 'swell'] : ['arp', 'rumble', 'swell']};
  return {version: 1, style, seed, bpm, totalBars: start, sections, materials, candy, learned: true, raw: raw.map(r => `${r.type}${r.len}:e${r.energy}:${'DBKL'.split('').filter((_, k) => r.act >> k & 1).join('') || '-'}:${r.transition}`).join(' ')};
}
