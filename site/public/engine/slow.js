// The calm composer core shared by all styles. The transformer writes one chord per two
// bars (the chord token is repeated in the second bar) and a melody over the 16 eighth
// notes of those bars. Each style sets how many notes, where they may start, how long they
// hold, the register and the voicing; the style module then arranges the parts.
// Decoding rules: diatonic chords; notes on the beat are chord tones or the ninth; no
// melody pitch a semitone from the sounding bass or keys; mostly stepwise motion.
import {loadModel} from './neural/transformer.js';
import {random} from './neural/random.js';
export const mod = (n, m) => (n % m + m) % m;
export const scale = [0, 2, 4, 5, 7, 9, 11];
export const intervals = [[0, 4, 7, 11], [0, 3, 7, 10], [0, 4, 7, 10], [0, 3, 6, 10], [0, 5, 7, 10]];
// Nearest voicing to the last one, around a target register.
export function voice(pcs, last, target = [50, 55, 58, 62]) {
  let best, score = Infinity;
  for (let inv = 0; inv < 4; inv++) for (let octave = 3; octave <= 4; octave++) {
    const notes = [12 * octave + pcs[inv]];
    for (let i = 1; i < 4; i++) {
      let p = pcs[(inv + i) % 4] + 12 * octave;
      while (p <= notes[i - 1]) p += 12;
      notes.push(p);
    }
    const cost = notes.reduce((s, p, i) => s + Math.abs(p - last[i]) + .34 * Math.abs(p - target[i]), 0)
      + 4 * Math.max(0, target[0] - 3 - notes[0]) + 3 * Math.max(0, notes[3] - target[3] - 4)
      + notes.slice(1).reduce((penalty, p, i) => penalty + (p - notes[i] === 1 ? 20 : 0), 0);
    if (cost < score) { score = cost; best = notes; }
  }
  return best;
}
// opts: styleToken, mode, shift, maxNotes (per two bars), offbeats (allow eighth-note
// offbeat starts), hold (bias towards holding), rest (bias towards resting), step (penalty
// per semitone of leap), low/high (sounding melody range), colour(root, quality) -> the
// fourth voicing tone, chordBias(root, quality, lastRoot), voicingTarget.
export function createCore(meta, bytes, seed, opts) {
  const model = loadModel(meta, bytes), draw = random(seed ^ 0x2c1b);
  const {CH0, NQ, M0, LO, HI, T} = meta, {shift, mode} = opts;
  const chords = [];
  for (let r = 0; r < 12; r++) for (let q = 0; q < NQ; q++)
    if (intervals[q].every(d => scale.includes((r + d) % 12))) chords.push(CH0 + r * NQ + q);
  let logits, used = 0, history = [];
  const feed = t => { logits = model.step(t); used++; };
  function start() {
    model.reset(); used = 0;
    for (const t of [meta.BOS, meta[opts.styleToken] ?? meta.LOFI, mode === 'major' ? meta.MAJ : meta.MIN]) feed(t);
    for (const t of history.slice(-20)) feed(t);
  }
  function sample(ids, temp, bias = () => 0) {
    const scores = ids.map(t => logits[t] / temp + bias(t));
    const max = Math.max(...scores), weights = scores.map(s => Math.exp(s - max));
    let v = draw() * weights.reduce((a, b) => a + b, 0);
    for (let i = 0; i < ids.length; i++) if ((v -= weights[i]) < 0) return ids[i];
    return ids.at(-1);
  }
  let lastKeys = opts.voicingTarget || [50, 55, 58, 62], lastPitch = Math.round((opts.low + opts.high) / 2), lastRoot = null;
  start();
  // One chord over two bars. `token` forces the chord (a repeat of a progression).
  function compose(token) {
    if (used + 20 > T) start();
    const tokens = [[], []];
    let chord = token, line = [], active = null;
    for (let half = 0; half < 2; half++) {
      feed(meta.BAR);
      if (half === 0 && chord === undefined)
        chord = sample(chords, .9, t => {
          const r = Math.floor((t - CH0) / NQ), q = (t - CH0) % NQ;
          return (r === lastRoot ? -3 : 0) + (opts.chordBias ? opts.chordBias(r, q, lastRoot) : 0);
        });
      feed(chord);
      history.push(meta.BAR, chord);
      const root = Math.floor((chord - CH0) / NQ), quality = (chord - CH0) % NQ;
      if (half === 0) lastRoot = root;
      const pcs = intervals[quality].map(d => mod(root + d + shift, 12));
      const colour = mod((opts.colour ? opts.colour(root, quality) : scale.includes((root + 2) % 12) ? root + 2 : root) + shift, 12);
      const keys = half ? lastKeys : voice([...pcs.slice(1), colour], lastKeys, opts.voicingTarget);
      lastKeys = keys;
      const bass = 28 + mod(root + shift - 28, 12);
      const stable = [...pcs, mod(root + 2 + shift, 12)];
      for (let slot = 0; slot < 8; slot++) {
        const ids = [M0], at = half * 8 + slot;
        if (active) ids.push(M0 + 1);
        const onBeat = slot % 2 === 0;
        if ((onBeat || opts.offbeats) && line.length < opts.maxNotes && at < 14)
          for (let p = LO; p <= HI; p++) {
            const sounding = p + shift;
            if (sounding < opts.low || sounding > opts.high || !scale.includes(p % 12)) continue;
            if (onBeat && !stable.includes(mod(sounding, 12))) continue;
            if ([bass, ...keys].some(q => [1, 11].includes(mod(sounding - q, 12)))) continue;
            ids.push(M0 + 2 + p - LO);
          }
        const t = sample(ids, 1.05, t => {
          // Rests after a phrase, holds while a note sounds, small steps between notes.
          if (t === M0) return active ? -1.2 : opts.rest + (line.length === 0 && at >= 6 ? -1.5 : 0) + (line.length && at % 8 > 4 ? .6 : 0);
          if (t === M0 + 1) return opts.hold;
          const pitch = LO + t - M0 - 2 + shift;
          return -opts.step * Math.abs(pitch - lastPitch) - (pitch === lastPitch ? 1 : 0);
        });
        feed(t); tokens[half].push(t); history.push(t);
        if (history.length > 200) history = history.slice(-40);
        if (t === M0) active = null;
        else if (t === M0 + 1) active.len += .5;
        else {
          lastPitch = LO + t - M0 - 2 + shift;
          active = {pitch: lastPitch, at, len: .5};
          line.push(active);
        }
      }
      tokens[half].info = {root, quality, stable, keys, bass, pcs};
    }
    return {chord, tokens, line};
  }
  // Melody notes of one bar of a two-bar chord, as events: they (and their release) end
  // before the next chord.
  function melody(pair, half, beatSec, gain, release = .3) {
    return pair.line.filter(n => Math.floor(n.at / 8) === half).map(n => {
      const beat = (n.at % 8) * .5 + .04, end = Math.min(n.at * .5 + n.len, 8) - half * 4;
      return {type: 'felt', beat, pitch: n.pitch, duration: Math.max(.35, end - beat - release / beatSec - .02), gain, pan: -.08,
        slot: n.at % 8, strong: n.at % 2 === 0, releaseSec: release};
    }).filter(e => e.beat + e.duration + release / beatSec <= 8 - half * 4);
  }
  return {compose, melody, sample: (ids, temp, bias) => sample(ids, temp, bias)};
}
// A progression of four two-bar chords, repeated; `renew` passes before the model writes
// a new one (Infinity keeps one progression for the whole material).
export function progressionPlayer(core, renew) {
  let progression = [], bar = 0, pair = null;
  return function next() {
    const half = bar % 2, chordIndex = Math.floor(bar / 2) % 4, cycle = Math.floor(bar / 8);
    if (half === 0) {
      if (chordIndex === 0 && cycle % renew === 0) progression = [];
      const repeat = progression.length > chordIndex ? progression[chordIndex] : undefined;
      pair = core.compose(repeat);
      if (repeat === undefined) progression.push(pair.chord);
    }
    bar++;
    return {pair, half, info: pair.tokens[half].info, tokens: pair.tokens[half], chordStart: half === 0, bar: bar - 1};
  };
}
