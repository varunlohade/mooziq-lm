// Nature: field sounds (birds, water, wind) under slow, sparse keys and vibraphone. No drums.
import {makeStyle, mod} from './common.js';
export const {songPlan, createSong} = makeStyle({
  planSalt: 0x9a7e, bpms: [56, 58, 60, 62], shifts: [-2, 0, 2, 3, 5], minor: .4, token: 'FLOW', renew: 2,
  core: {maxNotes: 3, offbeats: false, hold: 1.6, rest: -.5, step: .22, low: 62, high: 76},
  arrange({info, half, hit, melody, beatSec}) {
    if (half === 0) {
      info.keys.forEach((p, j) => hit('keys', .05 + j * .4, p, 8 - .05 - j * .4 - .5 / beatSec, .06, [-.35, .2, -.15, .35][j]));
      hit('drone', 0, 24 + mod(info.bass - 24, 12), 8 - .6 / beatSec, .16, 0);
    }
    melody(.08, .5, {inst: 'vibes'});
  }
});
