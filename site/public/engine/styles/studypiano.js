// Study Piano: solo piano. The left hand rolls a gentle broken chord in eighth notes, the
// right hand plays a calm, sparse melody. No drums and no other instruments.
import {makeStyle} from './common.js';
export const {songPlan, createSong} = makeStyle({
  planSalt: 0x57d9, bpms: [66, 70, 74, 78], token: 'FOLK', renew: 2,
  core: {maxNotes: 5, offbeats: true, hold: .9, rest: -1, step: .22, low: 62, high: 76},
  arrange({info, half, hit, melody, beatSec}) {
    const release = .35 / beatSec, low = info.bass + 12, notes = [low, ...info.keys.slice(0, 3)];
    // Broken chord: bass, then the voicing upward and back, each note ringing until the bar ends.
    const order = [0, 1, 2, 3, 2, 1, 2, 3];
    order.forEach((n, k) => {
      const beat = k * .5 + .02, end = (half ? 4 : 8) - release - .05;
      hit('keys', beat, notes[n], Math.max(.3, Math.min(2.5, end - beat)), k === 0 ? .075 : .05, [-.3, -.1, .1, .3][n]);
    });
    melody(.095, .35);
  }
});
