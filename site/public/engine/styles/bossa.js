// Bossa: nylon guitar comping in a gentle bossa rhythm, bass on the root and fifth, a shaker
// and a soft rim click; the melody on vibraphone or piano. Laid back, never busy.
import {makeStyle, jazzColour, iiV} from './common.js';
export const {songPlan, createSong, rehits} = makeStyle({
  planSalt: 0xb055a, bpms: [72, 75, 78, 80], token: 'JAZZHOP', renew: 4, rehits: true,
  core: {maxNotes: 5, offbeats: true, hold: .8, rest: -1.2, step: .25, low: 62, high: 74, colour: jazzColour, chordBias: iiV, voicingTarget: [52, 57, 60, 64]},
  arrange({info, half, hit, melody, beatSec}) {
    const release = .35 / beatSec;
    // Guitar comping: the chord on 1, the "and" of 2 and on 4 (a calm bossa figure).
    for (const [b, g] of [[0, .08], [1.5, .06], [3, .065]]) {
      const len = Math.min(1.2, (half ? 4 : 8) - b - release - .05);
      info.keys.forEach((p, j) => hit('keys', b + .02 + j * .018, p, len, g, [-.25, .15, -.1, .25][j], {inst: 'guitar'}));
    }
    const fifth = info.quality === 3 ? 6 : 7;
    hit('bass', .03, info.bass, 1.3, .3);
    hit('bass', 2.03, info.bass + fifth, 1.3, .25);
    for (let k = 0; k < 8; k++) hit('shaker', k * .5, 0, .1, k % 2 ? .03 : .045, .3, {track: 'drums'});
    for (const b of [.75, 2.5]) hit('rim', b + .02, 0, .12, .04, .15);
    melody(.085, .3, {inst: 'vibes'});
  }
});
