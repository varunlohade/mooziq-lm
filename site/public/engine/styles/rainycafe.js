// Rainy Café: a quiet piano trio by a rainy window. Jazz chords held for two bars, soft
// brushes and a shaker instead of a kit, a slow walking bass, the piano singing a sparse line.
import {makeStyle, jazzColour, iiV} from './common.js';
export const {songPlan, createSong} = makeStyle({
  planSalt: 0x4a5ca1e, bpms: [66, 69, 72, 75], token: 'JAZZHOP', renew: 4,
  core: {maxNotes: 5, offbeats: true, hold: .7, rest: -1.4, step: .25, low: 60, high: 72, colour: jazzColour, chordBias: iiV, voicingTarget: [52, 57, 60, 64]},
  arrange({info, half, hit, melody, bar, beatSec, rng}) {
    const release = .4 / beatSec;
    if (half === 0) info.keys.forEach((p, j) => hit('keys', .04 + j * .03, p, 3.2 - j * .03, .085 * (j === 3 ? .85 : 1), [-.35, .2, -.15, .35][j]));
    else info.keys.slice(1).forEach((p, j) => hit('keys', 1.5 + j * .03, p, Math.min(1.8, 4 - 1.5 - release - .05), .055, [-.15, .2, .35][j]));
    const fifth = info.quality === 3 ? 6 : 7, third = [4, 3, 4, 3, 5][info.quality];
    hit('bass', .03, info.bass, 1.7, .3);
    hit('bass', 2.03, info.bass + (half ? third : fifth), half ? 1.4 : 1.7, .24);
    // Brushes: a soft swish on 2 and 4, a shaker on the eighths.
    for (const b of [1, 3]) hit('brush', b + .04, 0, .3, .1, .1);
    for (let k = 0; k < 8; k++) hit('shaker', k * .5 + (k % 2 ? .05 : 0), 0, .1, k % 2 ? .025 : .04, .3, {track: 'drums'});
    melody(.085);
  }
});
