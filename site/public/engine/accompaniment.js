// Write a two-bar accompaniment for the song's instrument and pulse. The
// arranger supplies the composer's actual chord tones and their safe end time.
import {random} from './neural/random.js';

export function accompaniment(seed, style) {
  const rng = random(seed ^ 0x56f5be25);
  const choose = xs => xs[Math.floor(rng() * xs.length)];
  const quiet = ['flow', 'sleep', 'nature'].includes(style);
  const broken = style === 'studypiano' || (quiet ? rng() < .75 : rng() < .25);
  const spread = broken ? choose([.25, .5, .75]) : .015;
  const reverse = rng() < .5, rotation = Math.floor(rng() * 4);
  const pulse = Array.from({length: 2}, () => {
    const first = choose([0, 0, .5]);
    const later = style === 'bossa' ? choose([[1.5, 3], [1, 2.5], [1.5, 2.5]]) :
      quiet ? choose([[], [], [2.5]]) : choose([[], [1.5], [2], [2.5], [1.5, 3]]);
    return [first, ...later];
  });
  function arrange(keys, half) {
    if (!keys.length) return keys;
    const tones = [...new Map(keys.map(e => [e.pitch, e])).values()].sort((a,b) => a.pitch-b.pitch);
    if (reverse) tones.reverse();
    const end = Math.max(...keys.map(e => e.beat + e.duration));
    const out = [];
    const beats = pulse[half];
    for (const [i, start] of beats.entries()) {
      const next = beats[i+1] ?? 4;
      for (let j = 0; j < tones.length; j++) {
        const beat = start + j * spread + .03;
        if (beat >= next || beat + .25 >= end) break;
        const e = tones[(j + rotation + i) % tones.length];
        const duration = Math.min(end - beat, quiet ? end : Math.max(.35, next - beat - .1));
        out.push({...e, beat, duration, gain: e.gain * (i ? .82 : 1)});
      }
    }
    return out;
  }
  return {arrange, rhythm: JSON.stringify([spread, pulse[0]])};
}
