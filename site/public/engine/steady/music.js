// Steady: one progression of four two-bar chords for the whole theme, warm block chords,
// the bass on the root, a brushed backbeat and quiet eighth-note hats. The melody is sparse
// and stepwise. (The original Steady composer wrote one chord per bar; this calmer version
// holds each chord for two bars.)
import {random} from '../neural/random.js';
import {createCore, progressionPlayer, mod} from '../slow.js';
export function songPlan(seed) {
  const rng = random(seed);
  return {rng, bpm: [68, 71, 74][mod(seed, 3)], shift: [-5, -2, 0, 2, 3][Math.floor(rng() * 5)], mode: seed % 2 ? 'major' : 'minor'};
}
export function createSong(meta, bytes, seed) {
  const {rng, bpm, shift, mode} = songPlan(seed), beatSec = 60 / bpm;
  const core = createCore(meta, bytes, seed, {styleToken: 'STEADY', shift, mode, maxNotes: 4, offbeats: true, hold: 1, rest: -.5, step: .22,
    low: 60, high: 72, voicingTarget: [51, 55, 60, 64]});
  const next = progressionPlayer(core, Infinity);
  let bar = 0;
  function nextBar() {
    const {pair, half, info, tokens} = next(), events = [];
    const hit = (type, beat, pitch, duration, gain, pan = 0, extra = {}) => {
      const fraction = beat % 1, swung = beat + .092 * (1 - Math.abs(2 * fraction - 1));
      events.push({type, beat: swung, pitch, duration, gain, pan, seed: Math.floor(rng() * 4294967296), ...extra});
    };
    const release = .4 / beatSec;
    info.keys.forEach((p, j) => hit('keys', .024 + j * .014, p, (half ? 3.5 - release : 2.8) - j * .014, (half ? .07 : .1) * (j === 3 ? .82 : 1), [-.38, .23, -.17, .38][j]));
    hit('bass', .035, info.bass, 1.55, .31);
    hit('bass', half ? 2.5 : 2.03, info.bass, half ? .78 : 1.12, .22);
    hit('kick', 0, 0, .5, .34);
    hit('kick', half ? 2.5 : 2, 0, .46, .255);
    for (const b of [1, 3]) hit('brush', b + .035 + (rng() - .5) * .012, 0, .26, b === 3 ? .15 : .14, .09);
    for (let j = 0; j < 8; j++) {
      if (half && j === 7) continue;
      hit('hat', j * .5 + .01 + (rng() - .5) * .012, 0, .08, (j % 2 ? .04 : .055) * (.85 + .25 * rng()), -.27);
    }
    events.push(...core.melody(pair, half, beatSec, .1, .32).map(e => ({...e, seed: Math.floor(rng() * 4294967296)})));
    bar++;
    events.info = {root: info.root, quality: info.quality, tokens, stable: info.stable, chordStart: half === 0};
    return events;
  }
  return {bpm, shift, nextBar};
}
