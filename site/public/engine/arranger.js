// The arranger plays a song plan (engine/plan.js). For each bar it asks the model-driven
// composer of the section's material for notes, keeps only the plan's tracks, sets their
// instruments and patterns, and adds the parts the composers do not write: counter-lines,
// pads, shaker, ambience, transitions (fills, builds, drops, tape stops, swells) and small
// extras. Bars that only repeat get a change at least every few bars.
//
// Clash safety: every added pitched part uses pitch classes the keys or bass already
// sound in that bar, and the lead was written to avoid semitones against those.
import {accompaniment} from './accompaniment.js';
import {random} from './neural/random.js';
import {padEvents} from './pad.js';
import {planSong, validatePlan} from './plan.js';
import {learnedPlan, REST_SHIFT} from './learned.js';
import {develop} from './develop.js';
import {refineRegister} from './register.js';

const mod = (n, m) => (n % m + m) % m;
const TRACK_OF = {keys: 'keys', glass: 'lead', felt: 'lead', bass: 'bass', drone: 'bass', sub: 'bass', kick: 'drums', brush: 'drums', snare: 'drums', rim: 'drums', hat: 'drums'};
// A style can put an event on another track (a shaker that plays as part of the kit).
const trackOf = e => e.track ?? TRACK_OF[e.type];

// Another seed for the same composer with the same tempo, key and mode: a second song's
// worth of progression and melody that still belongs to this song.
function siblingSeed(engine, seed, offset) {
  if (!offset) return seed;
  const base = engine.songPlan(seed);
  for (let k = 1; k < 4000; k++) {
    const s = 1 + ((Math.imul(seed ^ (offset * 0x27d4eb2d), 0x9e3779b1) + Math.imul(k, 0x85ebca6b)) >>> 1) % 0x7ffffffe;
    const p = engine.songPlan(s);
    if (p.bpm === base.bpm && p.shift === base.shift && p.mode === base.mode) return s;
  }
  return seed;
}
// Chords as the keys play them: notes that start within a beat of each other form one chord
// (Flow rolls its chords). chordAt() gives the chord sounding at a beat.
function chordGroups(keys) {
  const groups = [];
  for (const e of [...keys].sort((a, b) => a.beat - b.beat)) {
    const g = groups.at(-1);
    if (g && e.beat - g.beat < .8) g.notes.push(e); else groups.push({beat: e.beat, notes: [e]});
  }
  return groups.filter(g => g.notes.length >= 3);
}
const chordAt = (keys, beat) => chordGroups(keys).filter(g => g.beat <= beat + .05).at(-1)?.notes ?? [];
// learned: {planner, composer} ({meta, bytes} each) runs the learned song path instead of
// planSong() and the shipped composer model (engine/learned.js).
export function createArrangedSong(engine, meta, bytes, seed, style, plan, onSection, learned) {
  const first = engine.songPlan(seed);
  plan = plan || (learned ? learnedPlan(learned.planner, style, seed, {bpm: first.bpm, mode: first.mode}) : planSong(style, seed, {bpm: first.bpm}));
  const problems = validatePlan(plan);
  if (problems.length) throw new Error('Invalid song plan: ' + problems.join(', '));
  const rng = random((seed ^ 0xa11a9e) >>> 0), instances = {};
  // The learned composer reads the section it is writing for from ctx (see engine/slow.js).
  const ctx = {sec: 0, act: 15}, songMeta = learned ? {...learned.composer.meta, style, ctx, restShift: REST_SHIFT[style] || 0} : meta, songBytes = learned ? learned.composer.bytes : bytes;
  const material = name => instances[name] ||= engine.createSong(songMeta, songBytes, siblingSeed(engine, seed, plan.materials[name].seedOffset));
  const arrangeKeys = style === 'jazzhop' ? null : accompaniment(seed, style).arrange;
  const lead = material('A');
  const {bpm, shift} = lead, beatSec = 60 / bpm;
  const sectionAt = bar => plan.sections.find(s => bar >= s.start && bar < s.start + s.bars);
  let bar = 0, previous = [], candyCooldown = 0;

  function drums(events, pattern, local, section) {
    let out = events;
    if (pattern === 'half') out = events.filter(e => (e.type === 'kick' && e.beat < 1) || ((e.type === 'snare' || e.type === 'brush') && e.beat >= 2 && e.beat < 3.5) || (e.type === 'hat' && Math.abs(e.beat * 2 - Math.round(e.beat * 2)) < .1));
    else if (pattern === 'sparse') out = events.filter(e => e.type === 'kick' || e.type === 'snare' || e.type === 'brush');
    else if (pattern === 'busy') {
      out = [...events];
      for (const b of [.75, 2.75]) out.push({type: 'rim', beat: b + .02, pitch: 0, duration: .15, gain: .03 + .015 * rng(), pan: .1, seed: Math.floor(rng() * 4294967296)});
    }
    // Small variety every so often: a dropped hat, an extra ghost note.
    if (local % 4 === 3 && rng() < .5) out = out.filter(e => !(e.type === 'hat' && e.beat > 3.4));
    return out;
  }
  function bass(events, pattern) {
    const lines = events.filter(e => trackOf(e) === 'bass');
    if (pattern === 'sustain') {
      // One long note per chord: keep the notes that start a chord, hold them until the next.
      const starts = lines.filter(e => e.type === 'drone' || e.beat < .1 || (Math.abs(e.beat - 2) < .1 && e.duration > .5));
      return starts.map((e, i) => {
        const next = starts[i + 1]?.beat ?? 4;
        return e.type === 'drone' ? e : {...e, duration: Math.max(.5, next - e.beat - .2 / beatSec - .05)};
      });
    }
    if (pattern === 'octave') return lines.flatMap(e => e.type === 'drone' ? [e]
      : [e, {...e, beat: e.beat + .5, pitch: e.pitch + 12, duration: Math.min(.3, Math.max(.05, e.duration - .6)), gain: e.gain * .55}].filter(x => x.duration > .05));
    return lines;
  }
  function voicing(e, kind) {
    if (kind === 'spread' && e.pitch >= 60) return {...e, pitch: e.pitch + 12, pan: e.pan * 1.4};
    if (kind === 'low' && e.pitch >= 64) return {...e, pitch: e.pitch - 12};
    return e;
  }
  // Counter-line: the guide tones (third and seventh) of each chord, held, above the keys;
  // or a harmony a chord tone below each lead note.
  function counter(keys, leads, spec, extra = []) {
    const out = [];
    if (spec.kind === 'harmony') {
      // Only chord tones that sit no semitone from anything else sounding (as for the lead).
      const around = [...keys, ...extra].map(e => mod(e.pitch, 12));
      for (const n of leads) {
        const pcs = new Set(chordAt(keys, n.beat).map(e => mod(e.pitch, 12)).filter(pc => !around.some(q => [1, 11].includes(mod(pc - q, 12)))));
        let h = n.pitch - 3;
        while (h > n.pitch - 10 && !pcs.has(mod(h, 12))) h--;
        if (h > n.pitch - 10) out.push({...n, type: 'felt', pitch: h, gain: n.gain * .6, pan: .25, inst: spec.inst, harmony: true});
      }
      return out;
    }
    const chords = chordGroups(keys);
    chords.forEach((g, i) => {
      const end = chords[i + 1]?.beat ?? 4;
      const tones = [...g.notes].sort((a, b) => a.pitch - b.pitch).slice(1, 3);
      tones.forEach((t, j) => {
        let p = t.pitch; while (p < 64) p += 12; while (p > 76) p -= 12;
        out.push({type: 'counter', beat: g.beat + .5 + j * .5, pitch: p, duration: Math.max(.3, end - g.beat - .5 - j * .5 - .45 / beatSec), gain: .05, pan: j ? .5 : -.5, inst: spec.inst, seed: t.seed + 7});
      });
    });
    return out;
  }
  function pads(info, prev) {
    if (info.chords && info.chords.length === 2) {
      return info.chords.flatMap(([root, quality], h) => padEvents({root, quality}, shift, bpm, h ? [] : prev, 2)
        .map(e => ({...e, beat: e.beat + 2 * h, pan: e.pan * 2.2})));
    }
    // The slowest styles hold each pad across both bars of the chord.
    if (style === 'flow' || style === 'sleep' || style === 'nature') return info.chordStart ? padEvents(info, shift, bpm, prev, 8).map(e => ({...e, pan: e.pan * 2.2})) : [];
    return padEvents(info, shift, bpm, prev).map(e => ({...e, pan: e.pan * 2.2}));
  }
  function shaker(pattern) {
    const out = [], step = pattern === 'sixteenths' ? .25 : .5;
    for (let b = 0; b < 4; b += step) out.push({type: 'shaker', beat: b + (b % .5 ? .03 : 0), pitch: 0, duration: .1,
      gain: (b % 1 === 0 ? .06 : .035) * (.85 + .3 * rng()), pan: .35, seed: Math.floor(rng() * 4294967296)});
    return out;
  }
  // The section's last bar(s): fills, builds, drops, tape stops and swells into the next part.
  function transition(kind, local, section, events, keys) {
    const last = local === section.bars - 1, out = [];
    // Everything here stays soft: background music should never jump out.
    if (kind === 'fill' && last) {
      events = events.filter(e => !(trackOf(e) === 'drums' && e.beat >= 3));
      for (let k = 0; k < 2; k++) out.push({type: 'snare', beat: 3 + k * .5 + .03, pitch: 0, duration: .15, gain: .06 + .03 * k, pan: (k - .5) * .2, seed: Math.floor(rng() * 4294967296)});
    }
    if (kind === 'build' && local >= section.bars - 2) {
      // A gentle lift: quarter-note snares that grow a little, and a soft rising noise.
      events = events.filter(e => e.type !== 'snare' && e.type !== 'brush');
      for (let b = 0; b < 4; b += last ? .5 : 1) out.push({type: 'snare', beat: b + .03, pitch: 0, duration: .12, gain: .03 + .04 * ((local - section.bars + 2) * 4 + b) / 8, pan: 0, seed: Math.floor(rng() * 4294967296)});
      if (local === section.bars - 2) out.push({type: 'riser', beat: 0, pitch: 0, duration: 8, gain: .035, pan: 0});
    }
    if ((kind === 'drop' || kind === 'cut') && last) events = events.filter(e => e.beat < 3 || trackOf(e) === 'keys');
    if (kind === 'tapestop' && last) events = events.filter(e => e.beat < 3.2 || trackOf(e) !== 'drums');
    if (kind === 'swell' && last && keys.length) {
      // A reversed piano chord that blooms into the next downbeat.
      for (const k of chordAt(keys, 2.4).slice(0, 3)) out.push({type: 'swell', beat: 2.4, pitch: k.pitch, duration: 1.6 - .015 / beatSec, gain: .06, pan: k.pan, inst: 'piano'});
    }
    return {events, extra: out, tapeStop: kind === 'tapestop' && last ? 3.2 : null};
  }
  // Small extras inside sections: a keys run, a vinyl rumble swell, ghost notes, a swell.
  function candy(section, local, keys, beatTrack) {
    if (candyCooldown > 0 || local >= section.bars - 1 || rng() > plan.candy.rate / 8) return [];
    candyCooldown = 3;
    const kind = plan.candy.kinds[Math.floor(rng() * plan.candy.kinds.length)];
    const pcs = [...new Set(chordAt(keys, 2.4).map(e => e.pitch))].sort((a, b) => a - b);
    // The run ends (with its release) before the bar line, so it never rings into the next chord.
    if (kind === 'arp' && pcs.length >= 3) return pcs.slice(-3).map((p, j) => ({type: 'arp', beat: 2.4 + j * .33, pitch: p + 12, duration: .3, gain: .035, pan: (j - 1.5) * .25, inst: 'piano', candy: 'arp'}));
    if (kind === 'rumble') return [{type: 'rumble', beat: 0, pitch: 0, duration: 4, gain: .035, pan: 0, candy: 'rumble'}];
    if (kind === 'ghost' && beatTrack) return [1.75, 2.25].map(b => ({type: 'rim', beat: b + .03, pitch: 0, duration: .12, gain: .03, pan: .15, seed: Math.floor(rng() * 4294967296), candy: 'ghost'}));
    if (kind === 'swell' && pcs.length >= 2) return pcs.slice(0, 2).map(p => ({type: 'swell', beat: 2.4, pitch: p, duration: 1.2, gain: .04, pan: 0, inst: 'piano', candy: 'swell'}));
    return [];
  }
  function nextBar() {
    const section = sectionAt(bar);
    if (!section) {
      bar++;
      const events = [];
      events.info = {...previous.info, silent: true, tracks: [], section: -1, sectionType: 'end', energy: 0, filter: 600, gain: 0, arranged: true};
      return events;
    }
    const local = bar - section.start, t = section.tracks;
    if (learned) Object.assign(ctx, section.learned);
    let raw = material(section.material).nextBar();
    const info = raw.info, line = learned && info.tokens?.info?.bassLine;
    // Learned path: the composer's bass line replaces the style's bass pattern (when it wrote one).
    if (line && raw.some(e => e.type === 'bass')) {
      const gain = Math.max(...raw.filter(e => e.type === 'bass').map(e => e.gain)), keep = raw.filter(e => e.type !== 'bass');
      raw = [...keep, ...line.map(n => ({type: 'bass', beat: n.at * .5 + .03, pitch: n.pitch, gain: gain * (n.at === 0 ? 1 : .85), pan: 0,
        duration: Math.max(.25, Math.min(n.len, 4 - n.at * .5) - .15 / beatSec - .05), seed: (n.pitch * 7919 + bar * 31 + n.at) >>> 0}))];
      raw.info = info;
    }
    candyCooldown--;
    let events = [];
    for (const e of raw) {
      const track = trackOf(e);
      if (!track || !t[track]) continue;
      if (track === 'keys') events.push(voicing({...e, inst: e.inst || t.keys.inst}, t.keys.voicing));
      else if (track === 'lead') events.push({...e, inst: t.lead.inst});
      else if (track === 'drums') events.push(e);
    }
    if (arrangeKeys && t.keys) {
      const chord = events.filter(e => e.type === 'keys');
      events = [...events.filter(e => e.type !== 'keys'), ...arrangeKeys(chord, local % 2)];
    }
    if (t.drums) events = [...events.filter(e => trackOf(e) !== 'drums'), ...drums(events.filter(e => trackOf(e) === 'drums'), t.drums.pattern, local, section)];
    if (t.bass) events.push(...bass(raw, t.bass.pattern));
    const keys = events.filter(e => e.type === 'keys'), leads = events.filter(e => e.type === 'felt');
    if (t.counter) events.push(...counter(keys, leads, t.counter, events.filter(e => e.type === 'bass' || e.type === 'drone')));
    if (t.pad) events.push(...pads(info, previous));
    if (t.perc) events.push(...shaker(t.perc.pattern));
    if (t.texture) for (const kind of t.texture.kinds || ['rain']) events.push({type: 'texture', inst: kind, beat: 0, pitch: 0, duration: 4, gain: t.texture.level, pan: 0});
    const tr = transition(section.out, local, section, events, keys);
    events = [...tr.events, ...tr.extra, ...candy(section, local, keys, !!t.drums)];
    const f = (local + 1) / section.bars, lerp = ([a, b]) => a + (b - a) * f;
    const energy = lerp(section.energy);
    events.info = {...info, arranged: true, section: plan.sections.indexOf(section), sectionType: section.type, bar: local,
      energy: +energy.toFixed(3), filter: Math.round(lerp(section.filter)), gain: section.type === 'outro' ? Math.max(.45, 1 - f * .6) : .9 + .1 * energy,
      reverb: 1.25 - .5 * energy, tapeStop: tr.tapeStop, tracks: Object.keys(t), material: section.material, leadInst: t.lead?.inst ?? null};
    previous = raw;
    bar++;
    return events;
  }
  // Write the whole song first, then develop it: the main motif returns varied, and the
  // ending settles on the tonic. Playback then reads the finished bars in order.
  const bars = [];
  for (let i = 0; i < plan.totalBars; i++) {
    const section = sectionAt(i);
    if (onSection && section.start === i) onSection(plan.sections.indexOf(section) + 1, plan.sections.length);
    bars.push(nextBar());
  }
  const development = develop(bars, plan, {shift, mode: first.mode, beatSec, rehits: !!engine.rehits});
  refineRegister(bars, plan);
  let cursor = 0;
  return {bpm, shift, plan, totalBars: plan.totalBars, development,
    nextBar: () => cursor < bars.length ? bars[cursor++] : nextBar()};
}
