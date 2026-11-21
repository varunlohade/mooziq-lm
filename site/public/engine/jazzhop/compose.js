// Jazz Hop: laid-back and calm. One chord per two bars, leaning towards ii-V motion and
// seventh chords with ninths, elevenths and thirteenths in the voicings; a sparse, stepwise
// melody that may start off the beat; a slow walking bass; soft boom-bap drums that sit late.
import {random} from '../neural/random.js';
import {createCore, progressionPlayer, mod, scale} from '../slow.js';
export function songPlan(seed) {
  const rng = random(seed ^ 0x3c6ef372);
  const groove = Math.floor(random(seed ^ 0x19cfba1)() * 4);
  return {rng, groove, bpm: [74, 77, 80, 82, 84][mod(seed, 5)], shift: [-5, -2, 0, 2, 3][Math.floor(rng() * 5)], mode: rng() < .55 ? 'minor' : 'major'};
}
// Colour tone: the thirteenth on dominants, else the ninth where diatonic, else the eleventh.
const colour = (root, quality) => quality === 2 && scale.includes((root + 9) % 12) ? root + 9
  : scale.includes((root + 2) % 12) ? root + 2 : quality !== 0 && scale.includes((root + 5) % 12) ? root + 5 : root;
export function createSong(meta, bytes, seed) {
  const {rng, bpm, shift, mode, groove} = songPlan(seed), beatSec = 60 / bpm;
  const core = createCore(meta, bytes, seed, {styleToken: 'JAZZHOP', shift, mode, maxNotes: 7, offbeats: true, hold: .5, rest: -2.2, step: .25,
    low: 60, high: 72, colour, voicingTarget: [52, 57, 60, 64],
    chordBias: (r, q, last) => (last !== null && mod(r - last, 12) === 5 ? 1.2 : 0) + [.35, .45, .5, -.4, -.6][q]});
  const next = progressionPlayer(core, 4), swing = .1 + .04 * rng();
  // Compose a repeating two-bar accompaniment for this material. The feel
  // ranges from spacious sustained chords to syncopated stabs and broken chords.
  const choose = values => values[Math.floor(rng() * values.length)];
  const entrances = [choose([0, 0, .5]), choose([0, .5, 1])];
  const answers = [choose([1.5, 2, 2.5, 3]), choose([1.5, 2.5, 3])];
  const spacing = groove === 2 ? choose([.25, .5, .75]) : .018;
  const instrument = choose(['piano', 'piano', 'guitar', 'vibes']);
  const kickAnswer = choose([1.5, 2, 2.5, 3]);
  const hatStep = choose([.5, .5, 1]);
  const descending = rng() < .5;
  let bar = 0;
  function nextBar() {
    const {pair, half, info, tokens} = next(), events = [];
    const hit = (type, beat, pitch, duration, gain, pan = 0, extra = {}) =>
      events.push({type, beat, pitch, duration, gain, pan, seed: Math.floor(rng() * 4294967296), ...extra});
    const release = .4 / beatSec, end = 4 * (2 - half);
    const entrance = groove === 0 ? 0 : entrances[half];
    const pitches = descending ? [...info.keys].reverse() : info.keys;
    const strike = (beat, notes, spacing, gate, gain) => {
      notes.forEach((pitch, i) => {
        const at = beat + .03 + i * spacing;
        hit('keys', at, pitch, Math.min(gate, end - at - release - .05),
          gain, [-.3, .2, -.12, .3][i], {inst: instrument});
      });
    };
    if (groove === 0) {
      // Leave room in the second bar for the melody to answer a held chord.
      if (!half) strike(0, pitches, .025, 5.5, .09);
      else strike(answers[1], pitches.slice(1), .02, .55, .06);
    } else {
      strike(entrance, pitches, spacing, groove === 1 ? .7 : 1.45, .085);
      if (groove === 1 || groove === 3) {
        strike(answers[half], groove === 1 ? pitches : pitches.slice(1), .02, .65, .065);
      }
    }
    // Bass: root, then the fifth; in the second bar root and third. Nothing rings into the next chord.
    const fifth = info.quality === 3 ? 6 : 7, third = [4, 3, 4, 3, 5][info.quality];
    hit('bass', .03, info.bass, 1.7, .33);
    hit('bass', 2.04, info.bass + (half ? third : fifth), half ? 1.4 : 1.7, .26);
    // Soft boom-bap: kick on 1 and 3, a late snare on 2 and 4, gentle swung eighths on the hat.
    hit('kick', 0, 0, .5, .42);
    hit('kick', kickAnswer, 0, .45, .34);
    if (bar % 4 === 3) hit('kick', 3.5, 0, .3, .16);
    for (const b of [1, 3]) hit('snare', b + .05 + .01 * rng(), 0, .3, .3, .05);
    for (let k = 0; k < 8; k += hatStep * 2) {
      if (k === 7 && bar % 2) continue;
      hit('hat', k * .5 + (k % 2 ? swing * .5 : 0), 0, .08, k % 2 ? .04 : .06, -.25);
    }
    if (bar % 2 === 1) hit('rim', 3.75 + .03, 0, .15, .035, .1);
    events.push(...core.melody(pair, half, beatSec, .09).map(e => ({...e, seed: Math.floor(rng() * 4294967296)})));
    bar++;
    events.info = {root: info.root, quality: info.quality, tokens, stable: info.stable, chordStart: half === 0};
    return events;
  }
  return {bpm, shift, nextBar};
}
