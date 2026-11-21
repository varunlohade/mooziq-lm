// Song plans. A plan says which sections a song has, how long and how intense each one
// is, which tracks play in it and on which instrument, which musical material (chord
// progression and melody, written by the model) it uses, and how sections join.
// The engine (engine/arranger.js) plays any valid plan. planSong() is a small stochastic
// planner with weighted choices; a learned planner can replace it with the same interface.
import {random} from './neural/random.js';

export const TRACKS = ['drums', 'bass', 'keys', 'lead', 'counter', 'pad', 'perc', 'texture'];
export const INSTRUMENTS = {keys: ['piano'], lead: ['piano', 'vibes', 'guitar'], counter: ['piano', 'vibes'], pad: ['pad']};
export const DRUM_PATTERNS = ['full', 'half', 'sparse', 'busy'];
export const BASS_PATTERNS = ['composer', 'sustain', 'octave'];
export const TRANSITIONS = ['fill', 'build', 'drop', 'tapestop', 'swell', 'cut'];

// Check a plan before playing it. Returns a list of problems (empty when valid).
export function validatePlan(plan) {
  const problems = [];
  if (!plan.sections?.length) problems.push('no sections');
  for (const [i, s] of (plan.sections || []).entries()) {
    if (!(s.bars > 0)) problems.push(`section ${i}: bars`);
    if (!(s.energy?.length === 2)) problems.push(`section ${i}: energy`);
    if (!plan.materials?.[s.material]) problems.push(`section ${i}: material ${s.material}`);
    for (const t of Object.keys(s.tracks || {})) if (!TRACKS.includes(t)) problems.push(`section ${i}: track ${t}`);
    if (s.tracks?.drums && !DRUM_PATTERNS.includes(s.tracks.drums.pattern)) problems.push(`section ${i}: drum pattern`);
    if (s.tracks?.bass && !BASS_PATTERNS.includes(s.tracks.bass.pattern)) problems.push(`section ${i}: bass pattern`);
    if (s.out && !TRANSITIONS.includes(s.out)) problems.push(`section ${i}: transition ${s.out}`);
  }
  return problems;
}

const pick = (rng, options) => {
  const total = options.reduce((a, [, w]) => a + w, 0);
  let v = rng() * total;
  for (const [value, w] of options) if ((v -= w) < 0) return value;
  return options.at(-1)[0];
};
// Song shapes: the body between intro and outro. Weighted, then each part varies in length.
const SHAPES = {
  beat: [
    [['A', 'B', 'break', 'A2'], 3], [['A', 'A2', 'B', 'break', 'B2'], 2], [['A', 'B', 'A2', 'bridge', 'B2'], 2],
    [['A', 'break', 'B', 'A2'], 2], [['A', 'B', 'bridge', 'break', 'A2'], 1]
  ],
  flow: [[['A', 'B', 'break', 'A2'], 3], [['A', 'bridge', 'B', 'A2'], 2], [['A', 'B', 'A2', 'break', 'B2'], 2]]
};
// Background music: the whole song stays in a narrow, comfortable range.
const BASE_ENERGY = {intro: .4, A: .55, A2: .65, B: .6, B2: .65, break: .42, bridge: .5, outro: .4};
const MATERIAL = {intro: 'A', A: 'A', A2: 'A', B: 'B', B2: 'B', break: 'A', bridge: 'C', outro: 'A'};

// Per-style presets until a learned planner takes over: drums or not, which lead
// instruments, which ambience, and solo piano. Energy and form rules are shared.
const LEADS = {piano: [['piano', 3], ['guitar', 1], ['vibes', 1]], soft: [['piano', 4], ['vibes', 3]], jazz: [['piano', 4], ['vibes', 3], ['guitar', 3]]};
export const PRESETS = {
  // The first three styles keep their made-up rain ('softrain'); the others use recordings.
  // These three keep the planner's original weights exactly, so their songs do not change.
  jazzhop: {beat: true, main: LEADS.piano, alt: LEADS.jazz, textures: ['softrain'], ambience: .2},
  flow: {beat: false, main: LEADS.piano, alt: [['piano', 4], ['vibes', 4], ['guitar', 2]], textures: ['softrain'], ambience: .2},
  steady: {beat: true, main: LEADS.piano, alt: LEADS.jazz, textures: ['softrain'], ambience: .2},
  rainycafe: {beat: true, main: [['piano', 1]], alt: [['vibes', 2], ['piano', 2]], textures: ['rain', 'cafe'], ambience: 1, gain: 2, light: true},
  sleep: {beat: false, main: [['piano', 1]], alt: [['piano', 3], ['vibes', 1]], textures: ['rain'], ambience: 1, gain: 1.4, quiet: true},
  bossa: {beat: true, main: [['vibes', 2], ['piano', 1]], alt: [['vibes', 2], ['piano', 2], ['guitar', 1]], textures: [], ambience: 0, light: true},
  studypiano: {beat: false, main: [['piano', 1]], alt: [['piano', 1]], textures: [], ambience: 0, solo: true},
  chillhop: {beat: true, main: [['guitar', 2], ['vibes', 1], ['piano', 1]], alt: LEADS.jazz, textures: ['rain'], ambience: .3},
  nature: {beat: false, main: [['vibes', 3], ['piano', 2]], alt: [['vibes', 2], ['piano', 2]], textures: ['forest', 'water'], ambience: 1, gain: 3.5}
};
export function planSong(style, seed, {bpm}) {
  const preset = PRESETS[style] || PRESETS.steady;
  const rng = random((seed ^ 0x5eedf00d) >>> 0), beat = preset.beat;
  const barSec = 4 * 60 / bpm, target = 190 + 70 * rng();
  const body = pick(rng, SHAPES[beat ? 'beat' : 'flow']);
  const types = ['intro', ...body, 'outro'];
  const sections = types.map(type => {
    const bars = type === 'intro' || type === 'outro' ? pick(rng, [[4, 2], [8, 3]])
      : type === 'break' || type === 'bridge' ? pick(rng, [[4, 1], [8, 3]]) : pick(rng, [[8, 2], [12, 1], [16, 3]]);
    return {type, bars};
  });
  // Fit the song to about 3 to 4.5 minutes by growing or shrinking body sections in 4-bar steps.
  const total = () => sections.reduce((a, s) => a + s.bars, 0);
  for (let guard = 0; guard < 40 && Math.abs(total() * barSec - target) > 2 * barSec; guard++) {
    // Breaks and bridges stay short (at most 8 bars); themes can grow to 16.
    const grow = total() * barSec < target, max = x => x.type === 'break' || x.type === 'bridge' ? 8 : 16;
    const body = sections.filter(x => x.type !== 'intro' && x.type !== 'outro' && (grow ? x.bars < max(x) : x.bars > 4));
    if (!body.length) break;
    const s = body[Math.floor(rng() * body.length)];
    s.bars += grow ? 4 : -4;
  }
  let start = 0;
  const leadFor = () => pick(rng, preset.alt);
  const mainLead = pick(rng, preset.main);
  const altLead = (() => { let l, k = 0; do l = leadFor(); while (l === mainLead && rng() < .8 && ++k < 20); return l; })();
  for (const [i, s] of sections.entries()) {
    const e0 = Math.min(1, Math.max(.1, BASE_ENERGY[s.type] + (rng() - .5) * .06));
    const e1 = s.type === 'outro' ? .25 : s.type === 'intro' || s.type === 'break' ? e0 + .06 : Math.min(1, e0 + .03);
    s.energy = [+e0.toFixed(2), +e1.toFixed(2)];
    s.material = MATERIAL[s.type];
    s.start = start; start += s.bars;
    const t = {}, e = e0;
    t.keys = {inst: 'piano', voicing: pick(rng, [['close', 3], ['spread', 2], ['low', 1]])};
    if (beat && e >= .5) t.drums = {pattern: e >= .6 ? pick(rng, [['full', 4], ['busy', 1]]) : pick(rng, [['full', 2], ['half', 2], ['sparse', 1]])};
    if (s.type !== 'intro' && s.type !== 'break' && s.type !== 'outro') t.bass = {pattern: s.type === 'B' || s.type === 'B2' ? pick(rng, [['sustain', 2], ['octave', 2], ['composer', 1]]) : pick(rng, [['composer', 3], ['octave', 1]])};
    else if (!beat && s.type !== 'intro') t.bass = {pattern: 'composer'};
    if (s.type === 'A' || s.type === 'A2') t.lead = {inst: s.type === 'A2' && rng() < .3 ? altLead : mainLead};
    if (s.type === 'B' || s.type === 'B2' || s.type === 'bridge') t.lead = {inst: altLead};
    if (s.type === 'intro' && rng() < .3) t.lead = {inst: mainLead};
    if ((s.type === 'B' || s.type === 'B2') && rng() < .7) t.counter = {inst: pick(rng, [['piano', 2], ['vibes', 1]]), kind: 'guide'};
    if (s.type === 'A2' && rng() < .6) t.counter = {inst: t.lead.inst, kind: 'harmony'};
    if (s.type === 'break' || s.type === 'outro' || (!beat && rng() < .7) || ((s.type === 'B' || s.type === 'B2') && rng() < .5) || (s.type === 'intro' && rng() < .5)) t.pad = {inst: 'pad'};
    if (beat && e >= .6 && rng() < .5) t.perc = {pattern: 'eighths'};
    if (preset.textures.length && (preset.ambience >= 1 || (s.type === 'intro' || s.type === 'break' || s.type === 'outro' ? rng() < .7 : rng() < preset.ambience)))
      t.texture = {level: +((.6 + .4 * rng()) * (preset.gain || 1)).toFixed(2), kinds: preset.textures};
    // Solo piano: only the keys and the melody, all song long.
    if (preset.solo) for (const k of Object.keys(t)) if (k !== 'keys' && k !== 'lead') delete t[k];
    if (preset.quiet) delete t.counter;
    s.tracks = t;
    // Tone follows energy: muffled when quiet, open when full. The break sweeps down.
    // Tone follows energy within a gentle range: softened when quiet, never harsh when full.
    const cut = x => Math.round(1800 + 14000 * x * x);
    s.filter = s.type === 'intro' ? [1600, cut(e1)] : s.type === 'break' ? [cut(.55), 2200] : s.type === 'outro' ? [cut(e0), 2000] : [cut(e0), cut(e1)];
    // How the section ends (and so how the next one arrives).
    const next = sections[i + 1]?.type;
    s.out = !next ? null
      : next === 'outro' ? pick(rng, [['tapestop', .5], ['fill', 2], [null, 3], ['swell', 2]])
      : s.type === 'break' ? pick(rng, [['build', 1], ['swell', 3], [null, 1]])
      : s.type === 'intro' ? pick(rng, [['swell', 2], ['fill', 1], [null, 2]])
      : next === 'break' ? pick(rng, [['cut', 1], ['swell', 2], [null, 2], ['tapestop', .3]])
      : pick(rng, [['fill', 3], ['build', .5], ['swell', 2], [null, 2]]);
    if (!beat && (s.out === 'fill' || s.out === 'drop' || s.out === 'cut')) s.out = pick(rng, [['swell', 2], [null, 1]]);
  }
  // The climax: the most intense body section gets every layer and a clear energy peak.
  const main = sections.filter(s => !['intro', 'outro', 'break', 'bridge'].includes(s.type));
  const climax = main.reduce((a, b) => (b.energy[0] > a.energy[0] || (b.energy[0] === a.energy[0] && b.start > a.start) ? b : a), main[0]);
  for (const s of sections) if (s !== climax) s.energy = s.energy.map(e => Math.min(e, .66));
  climax.climax = true;
  climax.energy = [.75, .8];
  if (!preset.solo) Object.assign(climax.tracks, {
    ...(beat ? {drums: {pattern: 'full'}, ...(preset.light ? {} : {perc: {pattern: 'eighths'}})} : {}),
    bass: climax.tracks.bass || {pattern: 'composer'}, lead: climax.tracks.lead || {inst: mainLead},
    counter: climax.tracks.counter || {inst: 'vibes', kind: climax.type === 'A2' ? 'harmony' : 'guide'}, pad: {inst: 'pad'}
  });
  else climax.tracks.lead = climax.tracks.lead || {inst: mainLead};
  if (climax.tracks.counter?.kind === 'harmony') climax.tracks.counter.inst = climax.tracks.lead.inst;
  if (preset.quiet) delete climax.tracks.counter;
  climax.filter = [12000, 14000];
  const materials = {A: {seedOffset: 0}, B: {seedOffset: 1}, C: {seedOffset: 2}};
  // Little extras between transitions: how often per 8 bars, by style.
  const candy = {rate: beat ? .6 : .45, kinds: beat ? ['arp', 'rumble', 'ghost', 'swell'] : ['arp', 'rumble', 'swell']};
  return {version: 1, style, seed, bpm, totalBars: start, sections, materials, candy};
}
