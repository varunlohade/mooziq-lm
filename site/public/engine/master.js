// Master bus for one song, run in the worker on each finished bar. It keeps the sub tidy,
// takes the harshness off the top, holds short-term loudness steady with a soft-knee
// compressor, and catches peaks with a look-ahead limiter. All state carries across bars.
const SR = 44100;
function biquad(type, f0, q, gainDb = 0) {
  const w = 2 * Math.PI * f0 / SR, cos = Math.cos(w), alpha = Math.sin(w) / (2 * q), a = 10 ** (gainDb / 40);
  let b0, b1, b2, a0, a1, a2;
  if (type === 'highpass') { b0 = (1 + cos) / 2; b1 = -(1 + cos); b2 = b0; a0 = 1 + alpha; a1 = -2 * cos; a2 = 1 - alpha; }
  else if (type === 'lowpass') { b0 = (1 - cos) / 2; b1 = 1 - cos; b2 = b0; a0 = 1 + alpha; a1 = -2 * cos; a2 = 1 - alpha; }
  else if (type === 'peaking') { b0 = 1 + alpha * a; b1 = -2 * cos; b2 = 1 - alpha * a; a0 = 1 + alpha / a; a1 = -2 * cos; a2 = 1 - alpha / a; }
  else if (type === 'highshelf' || type === 'lowshelf') {
    const s = type === 'highshelf' ? 1 : -1, sq = 2 * Math.sqrt(a) * alpha;
    b0 = a * ((a + 1) + s * (a - 1) * cos + sq); b1 = -2 * s * a * ((a - 1) + s * (a + 1) * cos); b2 = a * ((a + 1) + s * (a - 1) * cos - sq);
    a0 = (a + 1) - s * (a - 1) * cos + sq; a1 = 2 * s * ((a - 1) - s * (a + 1) * cos); a2 = (a + 1) - s * (a - 1) * cos - sq;
  }
  b0 /= a0; b1 /= a0; b2 /= a0; a1 /= a0; a2 /= a0;
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  return x => { const y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2; x2 = x1; x1 = x; y2 = y1; y1 = y; return y; };
}
// Per style: tone filters, compressor threshold (dBFS mean square), ratio, make-up gain.
const STYLES = {
  steady: {filters: [['highpass', 32, .7], ['peaking', 250, .8, 5], ['highshelf', 1500, .7, -4], ['lowpass', 10000, .5]], threshold: -24, ratio: 2.5, makeup: 5},
  // Jazz Hop: a sampled-record tone, deep bass, softened top.
  jazzhop: {filters: [['highpass', 32, .7], ['peaking', 3500, .9, 3], ['lowpass', 11000, .5]], threshold: -24, ratio: 2.5, makeup: 4.5},
  flow: {filters: [['highpass', 26, .7], ['lowpass', 1900, .5], ['lowpass', 1900, .7], ['highshelf', 800, .7, -7]], threshold: -24, ratio: 2.5, makeup: 8},
  rainycafe: {filters: [['highpass', 32, .7], ['peaking', 250, .8, 3], ['highshelf', 3000, .7, -3], ['lowpass', 10000, .5]], threshold: -24, ratio: 2.5, makeup: 7.8},
  sleep: {filters: [['highpass', 26, .7], ['lowpass', 2400, .5], ['highshelf', 900, .7, -6]], threshold: -24, ratio: 2.5, makeup: 10},
  bossa: {filters: [['highpass', 32, .7], ['highshelf', 3500, .7, -2], ['lowpass', 11000, .5]], threshold: -24, ratio: 2.5, makeup: 2.3},
  studypiano: {filters: [['highpass', 40, .7], ['highshelf', 3000, .7, -3], ['lowpass', 11000, .5]], threshold: -24, ratio: 2.5, makeup: 8.8},
  chillhop: {filters: [['highpass', 30, .7], ['peaking', 3500, .9, 2], ['lowpass', 10500, .5]], threshold: -24, ratio: 2.5, makeup: 5},
  nature: {filters: [['highpass', 30, .7], ['highshelf', 4000, .7, -3], ['lowpass', 9000, .5]], threshold: -24, ratio: 2.5, makeup: 10.9}
};
export const CEILING = 10 ** (-4.5 / 20);
export function createMaster(style) {
  const opt = STYLES[style];
  const chains = [0, 1].map(() => opt.filters.map(([type, f, q, g]) => biquad(type, f, q, g)));
  const ms = 1 - Math.exp(-1 / (.3 * SR)), atk = 1 - Math.exp(-1 / (.05 * SR)), rel = 1 - Math.exp(-1 / (.4 * SR));
  const knee = 6, slope = 1 - 1 / opt.ratio, makeup = 10 ** (opt.makeup / 20);
  const look = Math.round(.004 * SR), dl = new Float32Array(look + 1), dr = new Float32Array(look + 1);
  const limAtk = 1 - Math.exp(-4 / look), limRel = 1 - Math.exp(-1 / (.12 * SR));
  let power = 1e-6, gainDb = 0, limGain = 1, w = 0;
  const peaks = new Float32Array(look + 1);
  return function process(left, right, n) {
    for (let i = 0; i < n; i++) {
      let l = left[i], r = right[i];
      for (const f of chains[0]) l = f(l);
      for (const f of chains[1]) r = f(r);
      power += ms * (.5 * (l * l + r * r) - power);
      const over = 10 * Math.log10(power + 1e-12) - opt.threshold;
      const target = over <= -knee / 2 ? 0 : over >= knee / 2 ? -slope * over : -slope * (over + knee / 2) ** 2 / (2 * knee);
      gainDb += (target - gainDb) * (target < gainDb ? atk : rel);
      const g = 10 ** (gainDb / 20) * makeup;
      l *= g; r *= g;
      // Look-ahead limiter: the output is delayed by `look` samples, so the gain is already
      // down when a peak arrives.
      peaks[w] = Math.max(Math.abs(l), Math.abs(r));
      dl[w] = l; dr[w] = r;
      w = (w + 1) % (look + 1);
      let peak = 0;
      for (let k = 0; k <= look; k++) if (peaks[k] > peak) peak = peaks[k];
      const want = peak > CEILING ? CEILING / peak : 1;
      limGain += (want - limGain) * (want < limGain ? limAtk : limRel);
      const outL = dl[w] * limGain, outR = dr[w] * limGain;
      left[i] = Math.max(-CEILING, Math.min(CEILING, outL));
      right[i] = Math.max(-CEILING, Math.min(CEILING, outR));
    }
  };
}
