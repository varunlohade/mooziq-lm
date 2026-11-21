// The room under the music: a constant, slowly moving hiss-and-rain bed with sparse
// crackle and soft droplets. Two looping buffers of different lengths, built once; after
// that the audio graph does all the work, with no JavaScript per sample.
import {random} from './neural/random.js';

const NOISE_SEC = 19.7, DUST_SEC = 31.3, SEAM = .08;
// Pink-to-brown noise: a pink filter plus a leaky integrator, crossfaded over its loop seam.
function noiseLoop(sr, rng) {
  const n = Math.round(NOISE_SEC * sr), seam = Math.round(SEAM * sr), out = [];
  for (let ch = 0; ch < 2; ch++) {
    const x = new Float32Array(n + seam);
    let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0, brown = 0;
    for (let i = 0; i < x.length; i++) {
      const w = rng() * 2 - 1;
      b0 = .99886 * b0 + w * .0555179; b1 = .99332 * b1 + w * .0750759; b2 = .96900 * b2 + w * .1538520;
      b3 = .86650 * b3 + w * .3104856; b4 = .55000 * b4 + w * .5329522; b5 = -.7616 * b5 - w * .0168980;
      const pink = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * .5362) * .11;
      b6 = w * .115926;
      brown = .998 * brown + w * .045;
      x[i] = pink + brown;
    }
    for (let i = 0; i < seam; i++) {
      const a = i / seam;
      x[i] = x[i] * Math.sin(a * Math.PI / 2) + x[n + i] * Math.cos(a * Math.PI / 2);
    }
    out.push(x.subarray(0, n));
  }
  return out;
}
// Crackle: short, bright ticks at random times. Droplets: soft falling blips.
// Events alternate between two loops, so a style can play half of them.
function dustLoop(sr, rng) {
  const n = Math.round(DUST_SEC * sr), sets = [0, 1].map(() => [new Float32Array(n), new Float32Array(n)]);
  let out;
  const ticks = Math.round(DUST_SEC * 7), drops = Math.round(DUST_SEC * 1.1);
  for (let k = 0; k < ticks; k++) {
    out = sets[k % 2];
    const at = Math.floor(rng() * n), len = 30 + Math.floor(rng() * 160), amp = .05 * Math.exp(-2.2 * rng());
    const ch = rng() < .5 ? 0 : 1, other = .45 + .3 * rng();
    let prev = 0;
    for (let i = 0; i < len; i++) {
      const w = (rng() * 2 - 1) * amp * Math.exp(-6 * i / len), v = w - prev;
      prev = w;
      out[ch][(at + i) % n] += v;
      out[1 - ch][(at + i) % n] += v * other;
    }
  }
  for (let k = 0; k < drops; k++) {
    out = sets[k % 2];
    const at = Math.floor(rng() * n), len = Math.round(sr * (.03 + .05 * rng())), amp = .012 * Math.exp(-1.5 * rng());
    const f0 = 1100 + 1900 * rng(), pan = rng();
    let phase = 0;
    for (let i = 0; i < len; i++) {
      phase += (f0 * (1 + .35 * i / len)) / sr;
      const v = amp * Math.sin(2 * Math.PI * phase) * Math.exp(-5 * i / len) * Math.min(1, i / 20);
      out[0][(at + i) % n] += v * (1 - .5 * pan);
      out[1][(at + i) % n] += v * (.5 + .5 * pan);
    }
  }
  return sets;
}
function loopSource(ctx, channels) {
  const buffer = ctx.createBuffer(2, channels[0].length, ctx.sampleRate);
  buffer.copyToChannel(channels[0], 0);
  buffer.copyToChannel(channels[1], 1);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.loop = true;
  return source;
}
// Steady: bed 4 dB lower than Flow's level, half the crackle, each tick 6 dB softer.
const BED = {noise: .039, dust: .46, level: .44, sparseDust: .366, jazzLevel: .5, jazzDust: .8, jazzLowpass: 8000, highpass: 32, lowpass: 6000, flowLowpass: 1800, flowLevel: .7, flowAir: 9, sway: .4, swayHz: .045};
// The listening room: the music passes straight through and the bed sits under it.
// Loudness control lives in each song's master bus (engine/master.js).
export function createRoom(ctx) {
  const input = ctx.createGain(), output = ctx.createGain();
  input.connect(output);
  const bed = createBed(ctx), quiet = ctx.createGain();
  bed.connect(quiet).connect(output);
  // Flow is darker and quieter on top, so its bed is too.
  // The newer styles borrow a bed: dark and soft (Flow's) or Jazz Hop's crackle; else Steady's.
  const BORROW = {sleep: 'flow', nature: 'flow', chillhop: 'jazzhop'};
  function setStyle(kind) {
    const style = BORROW[kind] || kind, flow = style === 'flow', jazz = style === 'jazzhop', now = ctx.currentTime;
    bed.tone.frequency.setTargetAtTime(flow ? BED.flowLowpass : jazz ? BED.jazzLowpass : BED.lowpass, now, 1.5);
    bed.gain.setTargetAtTime(flow ? BED.flowLevel : jazz ? BED.jazzLevel : BED.level, now, 1.5);
    // A faint, even hiss stays on the very top, as in the reference.
    bed.air.gain.setTargetAtTime(flow ? BED.flowAir : 0, now, 1.5);
    // Jazz Hop keeps an audible record crackle; Steady gets half of it, softer.
    bed.dust[0].gain.setTargetAtTime(flow ? BED.dust : jazz ? BED.jazzDust : BED.sparseDust, now, 1.5);
    bed.dust[1].gain.setTargetAtTime(flow ? BED.dust : jazz ? BED.jazzDust : 0, now, 1.5);
  }
  // Silence the bed for the gap between songs after a manual change.
  function mute(on, seconds) {
    const now = ctx.currentTime;
    quiet.gain.cancelScheduledValues(now);
    quiet.gain.setValueAtTime(quiet.gain.value, now);
    quiet.gain.linearRampToValueAtTime(on ? 0 : 1, now + seconds);
  }
  return {input, output, setStyle, mute};
}
function createBed(ctx, seed = 0x5eed) {
  const rng = random(seed), out = ctx.createGain();
  const noise = loopSource(ctx, noiseLoop(ctx.sampleRate, rng)), dust = dustLoop(ctx.sampleRate, rng).map(set => loopSource(ctx, set));
  const highpass = ctx.createBiquadFilter(), lowpass = ctx.createBiquadFilter();
  highpass.type = 'highpass'; highpass.frequency.value = BED.highpass; highpass.Q.value = .5;
  lowpass.type = 'lowpass'; lowpass.frequency.value = BED.lowpass; lowpass.Q.value = .5;
  const noiseGain = ctx.createGain(), dustGains = dust.map(() => ctx.createGain()), sway = ctx.createOscillator(), depth = ctx.createGain();
  noiseGain.gain.value = BED.noise;
  dustGains.forEach(g => { g.gain.value = BED.dust; });
  // The hiss breathes slowly, a few dB either way.
  sway.frequency.value = BED.swayHz;
  depth.gain.value = BED.noise * BED.sway;
  sway.connect(depth).connect(noiseGain.gain);
  noise.connect(noiseGain).connect(highpass);
  dust.forEach((source, i) => source.connect(dustGains[i]).connect(highpass));
  const air = ctx.createBiquadFilter();
  air.type = 'highshelf'; air.frequency.value = 6500; air.gain.value = 0;
  highpass.connect(lowpass).connect(air).connect(out);
  out.tone = lowpass;
  out.air = air;
  out.dust = dustGains;
  noise.start(); dust.forEach(source => source.start()); sway.start();
  return out;
}
