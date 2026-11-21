// Flow: slow ambient writing. Each chord lasts two bars, the melody is sparse with long held
// notes on the beat, the sub drone glides between roots, and most songs are in minor keys.
import {random} from '../neural/random.js';
import {createCore, progressionPlayer, mod} from '../slow.js';
// Tempo, transposition and mode follow from the seed alone. Minor keys lean towards C, D and F minor.
export function songPlan(seed) {
  const rng = random(seed ^ 0x6a09e667);
  const mode = rng() < .8 ? 'minor' : 'major';
  const shift = mode === 'minor' ? [3, 5, -4, 3, 5, -4, -2, 0][Math.floor(rng() * 8)] : [-2, 0, 3, -4][Math.floor(rng() * 4)];
  return {rng, bpm: [58, 60, 63, 66][mod(seed, 4)], shift, mode};
}
export function createSong(meta, bytes, seed) {
  const {rng, bpm, shift, mode} = songPlan(seed), beatSec = 60 / bpm;
  const core = createCore(meta, bytes, seed, {styleToken: 'FLOW', shift, mode, maxNotes: 3, offbeats: false, hold: 1.6, rest: -.3, step: .2, low: 60, high: 74});
  const next = progressionPlayer(core, 2);
  let lastDrone = null;
  function nextBar() {
    const {pair, half, info, tokens} = next(), events = [];
    const hit = (type, beat, pitch, duration, gain, pan = 0, extra = {}) =>
      events.push({type, beat, pitch, duration, gain, pan, seed: Math.floor(rng() * 4294967296), ...extra});
    const drone = 24 + mod(info.bass - 24, 12);
    if (half === 0) {
      // Keys roll slowly up the chord and ring through both bars.
      info.keys.forEach((p, j) => hit('keys', .03 + j * .25, p, 8 - .03 - j * .25 - .4 / beatSec, .08 * (j === 3 ? .85 : 1), [-.3, .2, -.12, .3][j]));
      hit('drone', 0, drone, 8 - .6 / beatSec, .30, 0, {from: lastDrone ?? drone});
      info.keys.slice(2).forEach((p, j) => hit('keys', 2.5 + j * .3, p, 5.5 - j * .3 - .4 / beatSec, .035, j ? .3 : -.12));
      lastDrone = drone;
    } else info.keys.slice(1).forEach((p, j) => hit('keys', 1.02 + j * .3, p, 3 - j * .3 - .4 / beatSec, .04, [-.12, .2, .3][j]));
    events.push(...core.melody(pair, half, beatSec, .12).map(e => ({...e, seed: Math.floor(rng() * 4294967296)})));
    events.info = {root: info.root, quality: info.quality, tokens, stable: info.stable, chordStart: half === 0};
    return events;
  }
  return {bpm, shift, nextBar};
}
