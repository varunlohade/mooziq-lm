// Sleep: very slow and soft. Long warm pads carry the harmony, a deep drone sits under it,
// and the piano places a few quiet notes. No drums.
import {makeStyle, mod} from './common.js';
export const {songPlan, createSong} = makeStyle({
  planSalt: 0x5ee9, bpms: [50, 52, 54, 56], shifts: [3, 5, -4, -2, 0], minor: .7, token: 'FLOW', renew: 2,
  core: {maxNotes: 2, offbeats: false, hold: 2, rest: -1, step: .3, low: 60, high: 70},
  arrange({info, half, hit, melody, beatSec}) {
    if (half === 0) {
      info.keys.forEach((p, j) => hit('keys', .05 + j * .5, p, 8 - .05 - j * .5 - .5 / beatSec, .05, [-.4, .25, -.2, .4][j]));
      hit('drone', 0, 24 + mod(info.bass - 24, 12), 8 - .6 / beatSec, .22, 0);
    }
    melody(.07, .4);
  }
});
