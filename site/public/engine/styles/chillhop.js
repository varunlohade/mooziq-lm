// Chillhop: a bouncy, swung kit, a round sub bass, soft keys and a guitar or vibraphone lead.
import {makeStyle, jazzColour, iiV, mod} from './common.js';
export const {songPlan, createSong} = makeStyle({
  planSalt: 0xc4111, bpms: [80, 83, 86, 88], token: 'JAZZHOP', renew: 4,
  core: {maxNotes: 6, offbeats: true, hold: .6, rest: -1.6, step: .25, low: 60, high: 72, colour: jazzColour, chordBias: iiV, voicingTarget: [52, 57, 60, 64]},
  arrange({info, half, hit, melody, bar, beatSec, rng}) {
    const release = .4 / beatSec, swing = .12;
    if (half === 0) info.keys.forEach((p, j) => hit('keys', .03 + j * .02, p, 2.2, .08 * (j === 3 ? .85 : 1), [-.3, .2, -.12, .3][j]));
    else info.keys.slice(1).forEach((p, j) => hit('keys', 1.5 + j * .02, p, Math.min(1.6, 4 - 1.5 - release - .05), .055, [-.12, .2, .3][j]));
    // Round sub bass: short, bouncing notes on the root and its octave.
    const root = 28 + mod(info.bass - 28, 12);
    for (const [b, d, o] of [[0, .9, 0], [1.75, .4, 0], [2.5, .9, 12]]) hit('sub', b + .03, root + o, d, .3, 0);
    hit('kick', 0, 0, .5, .42);
    hit('kick', 1.75, 0, .4, .26);
    hit('kick', 2.5, 0, .45, .34);
    for (const b of [1, 3]) hit('snare', b + .04, 0, .3, .26, .05);
    for (let k = 0; k < 8; k++) hit('hat', k * .5 + (k % 2 ? swing * .5 : 0), 0, .08, k % 2 ? .035 : .05, -.25);
    melody(.085, .3, {inst: bar % 32 < 16 ? 'guitar' : 'vibes'});
  }
});
