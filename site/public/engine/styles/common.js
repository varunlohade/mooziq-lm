// Shared helpers for the style modules built on the calm composer core (engine/slow.js).
import {random} from '../neural/random.js';
import {createCore, progressionPlayer, mod, scale} from '../slow.js';
export {mod, scale};
// Colour tones for jazz-leaning styles: the thirteenth on dominants, else the ninth where
// diatonic, else the eleventh.
export const jazzColour = (root, quality) => quality === 2 && scale.includes((root + 9) % 12) ? root + 9
  : scale.includes((root + 2) % 12) ? root + 2 : quality !== 0 && scale.includes((root + 5) % 12) ? root + 5 : root;
export const iiV = (r, q, last) => (last !== null && mod(r - last, 12) === 5 ? 1.2 : 0) + [.35, .45, .5, -.4, -.6][q];
// Build a style: plan(seed) gives tempo, key and mode; arrange(parts) turns one bar of the
// composer's parts into events. `core` holds the melody settings for createCore.
// rehits: the comping strikes the same chord again within a bar (see develop.js).
export function makeStyle({planSalt, bpms, shifts = [-5, -2, 0, 2, 3], minor = .55, token, core, renew = 2, rehits = false, arrange}) {
  function songPlan(seed) {
    const rng = random(seed ^ planSalt);
    return {rng, bpm: bpms[mod(seed, bpms.length)], shift: shifts[Math.floor(rng() * shifts.length)], mode: rng() < minor ? 'minor' : 'major'};
  }
  function createSong(meta, bytes, seed) {
    const {rng, bpm, shift, mode} = songPlan(seed), beatSec = 60 / bpm;
    const c = createCore(meta, bytes, seed, {styleToken: token, shift, mode, ...core}), next = progressionPlayer(c, renew);
    let bar = 0;
    function nextBar() {
      const parts = next(), events = [];
      const hit = (type, beat, pitch, duration, gain, pan = 0, extra = {}) =>
        events.push({type, beat, pitch, duration, gain, pan, seed: Math.floor(rng() * 4294967296), ...extra});
      const melody = (gain, release, extra = {}) => events.push(...c.melody(parts.pair, parts.half, beatSec, gain, release)
        .map(e => ({...e, ...extra, seed: Math.floor(rng() * 4294967296)})));
      arrange({...parts, hit, melody, bar, beatSec, rng});
      bar++;
      const {info} = parts;
      events.info = {root: info.root, quality: info.quality, tokens: parts.tokens, stable: info.stable, chordStart: parts.half === 0};
      return events;
    }
    return {bpm, shift, nextBar};
  }
  return {songPlan, createSong, rehits};
}
