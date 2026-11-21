// Song development over a finished, arranged song (every bar's events known).
//  - The main motif (two bars of the model's lead in the first theme) comes back varied:
//    restated, shifted in rhythm and moved up a step, fragmented, and moved into the second
//    theme. Every placed note is snapped to a pitch that fits the target bar.
//  - The last two bars resolve to the tonic chord.
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
  return {snap, endLimit, stableAt};
}
const leadNotes = bar => bar.filter(e => e.type === 'felt' && !e.harmony);
// Place a motif variant (notes with beat offsets across two bars) into bars [at, at+1].
function place(bars, at, motif, {shift, beatSec, inst, step = 0, delay = 0, keep = Infinity, label, rehits}) {
  let placed = 0;
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
    }
    const info = bar.info;
    bars[at + b] = [...kept, ...add];
    bars[at + b].info = info;
    placed += add.length;
  }
  return placed;
}
export function develop(bars, plan, {shift, mode, beatSec, rehits = false}) {
  const record = {motif: null, occurrences: [], ending: null};
  const sections = plan.sections, first = sections.find(s => s.type === 'A');
  // The motif: the busiest two-bar window of the first theme (else any theme), at least two notes.
  let motif = null;
  for (const s of [first, ...sections.filter(x => x !== first && ['A', 'A2', 'B', 'B2'].includes(x.type))]) {
    if (!s || motif) continue;
    let best = null;
    for (let i = s.start; i < s.start + s.bars - 1; i++) {
      const notes = [0, 1].flatMap(b => leadNotes(bars[i + b]).map(e => ({beat: e.beat + 4 * b, pitch: e.pitch, duration: e.duration})));
      if (notes.length >= 2 && (!best || notes.length > best.notes.length)) best = {i, notes};
      if (best?.notes.length >= 4) break;
    }
    if (best) { motif = best.notes; record.motif = {bar: best.i, notes: motif.length}; record.occurrences.push({bar: best.i, kind: 'original'}); }
  }
  if (motif) {
    // A diatonic step up: the next scale tone above each note (then snapped to fit).
    const leadOf = s => s.tracks.lead?.inst ?? first.tracks.lead?.inst ?? 'piano';
    const plans = [];
    for (const s of sections) {
      if (s.type === 'A2') {
        plans.push({at: s.start, label: 'restated', opts: {}});
        if (s.bars >= 12) plans.push({at: s.start + 8, label: 'shifted up a step', opts: {delay: .5, step: 2}});
      }
      if ((s.type === 'B' || s.type === 'B2') && s.bars >= 8) plans.push({at: s.start + s.bars - 4, label: 'moved into the second theme', opts: {step: -3}, section: s});
      if (s.type === 'bridge' && s.bars >= 4) plans.push({at: s.start, label: 'fragment', opts: {keep: 2}, section: s});
      if (s.type === 'outro' && s.bars >= 4) plans.push({at: s.start, label: 'fragment', opts: {keep: 3}, section: s});
    }
    for (const p of plans) {
      const s = p.section ?? sections.find(x => p.at >= x.start && p.at < x.start + x.bars);
      if (place(bars, p.at, motif, {shift, beatSec, rehits, inst: leadOf(s), label: p.label, ...p.opts}) >= 2) record.occurrences.push({bar: p.at, kind: p.label});
    }
  }
  // The ending: the last two bars hold the tonic chord (keys, pad, bass), with no melody.
  const tonic = mod(shift + (mode === 'major' ? 0 : 9), 12), quality = mode === 'major' ? [0, 4, 7, 11] : [0, 3, 7, 10];
  for (const at of [bars.length - 2, bars.length - 1]) {
    const bar = bars[at];
    if (!bar) continue;
    const info = bar.info, root = 48 + mod(tonic - 48, 12);
    const keep = bar.filter(e => !['keys', 'felt', 'bass', 'drone', 'sub', 'pad', 'counter', 'arp', 'swell'].includes(e.type));
    // Third, fifth, seventh and the root on top, around middle C.
    const voicing = [quality[1], quality[2], quality[3], 12].map(d => root + d);
    const chord = voicing.map((p, j) => ({type: 'keys', beat: .03 + j * .02, pitch: p, duration: 3.4, gain: .1, pan: [-.3, .2, -.12, .3][j], inst: 'piano', seed: 4242 + j}));
    const low = [{type: 'pad', beat: 0, pitch: 36 + mod(tonic - 36, 12), duration: 3.4, gain: .045, pan: -.4, releaseSec: .35},
      {type: 'pad', beat: 0, pitch: 36 + mod(tonic - 36, 12) + 7, duration: 3.4, gain: .035, pan: .4, releaseSec: .35}];
    bars[at] = [...keep, ...chord, ...low];
    bars[at].info = {...info, root: mod(tonic - shift, 12), quality: mode === 'major' ? 0 : 1, chords: undefined, tonic: true};
  }
  record.ending = {tonicPc: tonic, mode};
  return record;
}
