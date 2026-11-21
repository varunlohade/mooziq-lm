import {PRESETS} from './plan.js';
import {voice, intervals} from './slow.js';
import {random} from './neural/random.js';
// Song development over a finished, arranged song (every bar's events known).
//  - The main motif (two bars of the model's lead in the first theme) comes back varied:
//    restated, shifted in rhythm and moved up a step, fragmented, and moved into the second
//    theme. Every placed note is snapped to a pitch that fits the target bar.
//  - The last two bars form a cadence into the tonic chord.
// Clash safety: a placed lead note never sits a semitone from any accompanying pitch class
// in its bar, and notes on the beat are chord tones of the chord sounding there.
const mod = (n, m) => (n % m + m) % m;
const SCALE = [0, 2, 4, 5, 7, 9, 11];
const ACCOMPANY = new Set(['keys', 'bass', 'drone', 'sub', 'pad', 'counter', 'arp', 'swell']);

function chordGroups(keys) {
  const groups = [];
  for (const e of [...keys].sort((a, b) => a.beat - b.beat)) {
    const g = groups.at(-1);
    if (g && e.beat - g.beat < .8) g.notes.push(e); else groups.push({beat: e.beat, notes: [e]});
  }
  const chords = groups.filter(g => g.notes.length >= 3);
  // A broken chord (one note at a time, each left ringing) is still one chord for the bar.
  const sorted = [...keys].sort((a, b) => a.beat - b.beat);
  if (!chords.length && sorted.length >= 3 && new Set(sorted.map(e => mod(e.pitch, 12))).size <= 4) return [{beat: sorted[0].beat, notes: sorted}];
  return chords;
}
function fitter(events, shift, beatSec, before = [], rehits = false) {
  // Notes held over from the previous bar (Flow's two-bar chords) still count here.
  const held = before.filter(e => ACCOMPANY.has(e.type) && e.beat + e.duration > 4);
  const keys = events.filter(e => e.type === 'keys'), groups = chordGroups(keys);
  const carried = chordGroups(before.filter(e => e.type === 'keys')).at(-1);
  if (carried && !groups.some(g => g.beat < .1)) groups.unshift({beat: 0, notes: carried.notes});
  const accompany = new Set([...events, ...held].filter(e => ACCOMPANY.has(e.type)).map(e => mod(e.pitch, 12)));
  const bass = [...events, ...held].filter(e => e.type === 'bass' || e.type === 'drone' || e.type === 'sub').map(e => mod(e.pitch, 12));
  const scale = SCALE.map(d => mod(d + shift, 12));
  const chordAt = beat => groups.filter(g => g.beat <= beat + .05).at(-1);
  const ok = (p, beat) => {
    const pc = mod(p, 12);
    if (!scale.includes(pc) || [...accompany].some(q => [1, 11].includes(mod(pc - q, 12)))) return false;
    if (Math.abs(beat - Math.round(beat)) < .06) {
      const chord = chordAt(beat);
      return chord ? chord.notes.some(n => mod(n.pitch, 12) === pc) || bass.includes(pc) : bass.includes(pc);
    }
    return true;
  };
  // Nearest fitting pitch to p, in the lead's range.
  const snap = (p, beat) => {
    for (let d = 0; d <= 7; d++) for (const q of d ? [p + d, p - d] : [p]) if (q >= 57 && q <= 80 && ok(q, beat)) return q;
    return null;
  };
  // A note may not ring into the next chord. With `rehits`, the same chord struck again
  // (a guitar comping figure) is not a new chord, so the note may ring through it.
  const pcs = g => new Set(g.notes.map(n => mod(n.pitch, 12)));
  const same = (g, beat) => { const now = chordAt(beat); if (!now) return false; const a = pcs(now); return [...pcs(g)].every(p => a.has(p)); };
  const endLimit = beat => (groups.find(g => g.beat > beat + .05 && !(rehits && same(g, beat)))?.beat ?? 4) - .3 / beatSec - .03;
  const stableAt = beat => [...new Set([...(chordAt(beat)?.notes ?? []).map(n => mod(n.pitch, 12)), ...bass])];
  return {snap, ok, endLimit, stableAt};
}
const leadNotes = bar => bar.filter(e => e.type === 'felt' && !e.harmony);
// Harmonize a returning theme as a phrase. Keep the cheapest path ending on
// each legal pitch so that one note can anticipate the direction of the next.
function shapeReturn(notes) {
  if (notes.length < 2) return;
  let paths = [{cost: 0, pitches: []}];
  for (const [index, {target, choices, event}] of notes.entries()) {
    paths = choices.map(pitch => {
      const pitchCost = Math.abs(pitch - target) + .05 * Math.abs(pitch - event.pitch);
      let bestCost = Infinity, bestPath;
      for (const path of paths) {
        let cost = path.cost + pitchCost;
        if (index > 0) {
          const intended = target - notes[index - 1].target;
          const actual = pitch - path.pitches.at(-1);
          cost += .8 * Math.abs(actual - intended);
          if (Math.sign(actual) !== Math.sign(intended)) cost += 2.5;
        }
        if (cost < bestCost) {
          bestCost = cost;
          bestPath = path;
        }
      }
      return {cost: bestCost, pitches: [...bestPath.pitches, pitch]};
    });
  }
  const best = paths.reduce((a, b) => a.cost <= b.cost ? a : b);
  notes.forEach(({event}, i) => { event.pitch = best.pitches[i]; });
}

// Place a motif variant (notes with beat offsets across two bars) into bars [at, at+1].
function place(bars, at, motif, {shift, beatSec, inst, step = 0, delay = 0, keep = Infinity, label, rehits}) {
  let placed = 0;
  const replacements = [], returnNotes = [];
  for (let b = 0; b < 2; b++) {
    const bar = bars[at + b];
    if (!bar) continue;
    const fit = fitter(bar, shift, beatSec, bars[at + b - 1] || [], rehits);
    const notes = motif.slice(0, keep).filter(n => n.beat + delay >= 4 * b && n.beat + delay < 4 * (b + 1));
    const kept = bar.filter(e => !(e.type === 'felt'));
    const add = [];
    for (const n of notes) {
      const beat = n.beat + delay - 4 * b, p = fit.snap(n.pitch + step, beat);
      if (p === null) continue;
      const duration = Math.min(n.duration, fit.endLimit(beat) - beat, 4 - .3 / beatSec - .03 - beat);
      if (duration < .2) continue;
      add.push({type: 'felt', beat, pitch: p, duration, gain: .11, pan: -.08, inst, releaseSec: .3, seed: (p * 7919 + at) >>> 0,
        slot: Math.min(7, Math.floor(beat * 2)), strong: Math.abs(beat - Math.round(beat)) < .06, stable: fit.stableAt(beat), motif: label});
      if (label === 'refrain' || label === 'restated') {
        const target = n.pitch + step, choices = [p];
        // A small neighborhood preserves the register of the original arrangement.
        for (let q = Math.max(57, p - 3); q <= Math.min(80, p + 3); q++) {
          if (q !== p && Math.abs(q - target) <= 7 && fit.ok(q, beat)) choices.push(q);
        }
        returnNotes.push({target, choices, event: add.at(-1)});
      }
    }
    const info = bar.info;
    const replacement = [...kept, ...add];
    replacement.info = info;
    replacements.push(replacement);
    placed += add.length;
  }
  // Keep the original phrase if the new harmony cannot support an audible reply.
  if (placed < 2) return 0;
  shapeReturn(returnNotes);
  bars.splice(at, replacements.length, ...replacements);
  return placed;
}
export function develop(bars, plan, {shift, mode, beatSec, rehits = false}) {
  const record = {motif: null, occurrences: [], ending: null};
  const sections = plan.sections, first = sections.find(s => s.type === 'A');
  // The motif: the busiest two-bar window of the first theme (else any theme), at least two notes.
  let motif = null;
  const themes = sections.filter(s => ['A','A2','B','B2'].includes(s.type));
  const recurring = random(plan.seed ^ 0x31fc)() < .8;
  for (const s of (recurring ? themes : [first, ...themes.filter(x => x !== first)])) {
    if (!s || motif) continue;
    let best = null;
    for (let i = s.start; i < s.start + s.bars - 1; i += 2) {
      const notes = [0, 1].flatMap(b => leadNotes(bars[i + b]).map(e => ({beat: e.beat + 4 * b, pitch: e.pitch, duration: e.duration})));
      if (notes.length >= 2 && (!best || notes.length > best.notes.length)) best = {i, notes};
      if (best?.notes.length >= 4) break;
    }
    if (best) { motif = best.notes; record.motif = {bar: best.i, notes: motif.length}; record.occurrences.push({bar: best.i, kind: 'original'}); }
  }
  if (motif) {
    // A diatonic step up: the next scale tone above each note (then snapped to fit).
    const leadOf = s => s.tracks.lead?.inst ?? first?.tracks.lead?.inst ?? 'piano';
    const plans = [];
    // The motif's theme is A when the song has one; a song with only the second theme (B)
    // develops the motif within B, and A sections are then the other theme.
    const source = sections.find(s => record.motif.bar >= s.start && record.motif.bar < s.start + s.bars);
    const theme = source.type[0], other = theme === 'A' ? 'B' : 'A';
    // Let the listener hear the idea twice within its own theme before the
    // arrangement moves on. The answer keeps its rhythm with a shorter ending.
    const answer = record.motif.bar + 4;
    if (answer + 1 < source.start + source.bars) {
      plans.push({at: answer, label: 'answer', section: source,
        opts: {keep: Math.max(2, motif.length - 1)}});
    }
    // Repeat the opening idea in its original rhythm after a contrasting phrase.
    // Quieter arrangements can leave the lead out for a whole section.
    if (recurring) {
      for (const s of themes.filter(s => s.start >= source.start + source.bars && s.tracks.lead)) {
        plans.push({at: s.start, label: 'refrain', section: s, opts: {}});
      }
    }
    for (const s of sections) {
      if (s.type === theme + '2') {
        plans.push({at: s.start, label: 'restated', opts: {}});
        if (s.bars >= 12) plans.push({at: s.start + 8, label: 'shifted up a step', opts: {delay: .5, step: 2}});
      }
      if ((s.type === other || s.type === other + '2') && s.bars >= 8) plans.push({at: s.start + s.bars - 4, label: 'moved into the second theme', opts: {step: -3}, section: s});
      if (s.type === 'bridge' && s.bars >= 4) plans.push({at: s.start, label: 'fragment', opts: {keep: 2}, section: s});
      if (s.type === 'outro' && s.bars >= 4) plans.push({at: s.start, label: 'fragment', opts: {keep: 3}, section: s});
    }
    const occupied = new Set([record.motif.bar, record.motif.bar + 1]);
    for (const p of plans) {
      if (occupied.has(p.at) || occupied.has(p.at + 1)) continue;
      const s = p.section ?? sections.find(x => p.at >= x.start && p.at < x.start + x.bars);
      if (place(bars, p.at, motif, {shift, beatSec, rehits, inst: leadOf(s), label: p.label, ...p.opts}) >= 2) {
        if (p.label === 'refrain') {
          // A beat of breathing room before the downbeat. Let the chord and bass ring.
          const pickup = bars[p.at - 1];
          const parts = ['felt','counter','arp','hat','shaker','snare','rim','brush','kick'];
          for (let i = pickup.length - 1; i >= 0; i--) {
            const e = pickup[i];
            if (e.beat >= 3 && parts.includes(e.type)) pickup.splice(i, 1);
            else if (e.beat < 3 && ['felt','counter','arp'].includes(e.type))
              e.duration = Math.min(e.duration, 3 - e.beat);
          }
        }
        occupied.add(p.at);
        occupied.add(p.at + 1);
        record.occurrences.push({bar: p.at, kind: p.label});
      }
    }
  }
  // A soft subdominant approach for the drumless styles, V7 for the beat
  // styles. In minor the dominant borrows the raised seventh to lead home.
  const tonic = mod(shift + (mode === 'major' ? 0 : 9), 12);
  const tonicQuality = mode === 'major' ? 0 : 1;
  const dominant = PRESETS[plan.style]?.beat;
  let previousVoicing = [50, 55, 59, 64];
  const preceding = bars.at(-3)?.filter(e => e.type === 'keys') ?? [];
  const previousPitches = [...new Set(preceding.map(e => e.pitch))].sort((a,b) => a-b);
  if (previousPitches.length >= 4) previousVoicing = previousPitches.slice(-4);
  for (const at of [bars.length - 2, bars.length - 1]) {
    const bar = bars[at];
    if (!bar) continue;
    const home = at === bars.length - 1;
    const root = mod(tonic + (home ? 0 : dominant ? 7 : 5), 12);
    const quality = home || !dominant ? tonicQuality : 2;
    const voicing = voice(intervals[quality].map(d => mod(root+d,12)), previousVoicing);
    previousVoicing = voicing;
    const keep = bar.filter(e => !ACCOMPANY.has(e.type) && e.type !== 'felt');
    const chord = voicing.map((pitch, j) => ({type: 'keys', beat: .03 + j * .02, pitch,
      duration: 3.4, gain: home ? .09 : .1, pan: [-.3,.2,-.12,.3][j], inst: 'piano', seed: 4242+j}));
    if (!PRESETS[plan.style]?.solo) {
      for (const interval of [0, 7]) chord.push({type: 'pad', beat: 0, pitch: 36 + root + interval,
        duration: 3.4, gain: interval ? .035 : .045, pan: interval ? .4 : -.4, releaseSec: .35});
    }
    bars[at] = [...keep, ...chord];
    bars[at].info = {...bar.info, root: mod(root-shift,12), quality, chords: undefined, tonic: home};
  }
  // Ambient pads can ring into the next bar. Leave space for the finished
  // melody when a sustained pad voice would rub a semitone against it.
  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    if (!bar.some(e => e.type === 'pad')) continue;
    for (let k = bar.length - 1; k >= 0; k--) {
      const pad = bar[k];
      if (pad.type !== 'pad') continue;
      const end = pad.beat + pad.duration + (pad.releaseSec ?? .35) / beatSec;
      for (let j = Math.max(0, i - 2); j <= Math.min(bars.length - 1, i + Math.floor(end / 4)); j++) {
        const offset = 4 * (j - i);
        if (bars[j].some(n => n.type === 'felt' && [1, 11].includes(mod(n.pitch - pad.pitch, 12))
          && offset + n.beat < end && offset + n.beat + n.duration + (n.releaseSec ?? .3) / beatSec > pad.beat)) {
          bar.splice(k, 1);
          break;
        }
      }
    }
  }
  record.ending = {tonicPc: tonic, mode};
  return record;
}
