// The band. Plays each bar of events with recorded instruments (upright piano, vibraphone,
// nylon guitar, finger bass, lo-fi drum hits, shaker) and soft synth layers, then mixes:
//   keys bus   keys, counter-lines, pads, arps, swells: wide, tape-saturated, ducked by the kick
//   lead bus   the melody and its harmony: dry and close, a little tape wow
//   main bus   bass, drums, ambience: bass and kick in the centre
//   send bus   a reverb send per part, with a short pre-delay
// Then the section filter and level from the song plan, and the master bus.
import {random} from './neural/random.js';
import {padEvents} from './pad.js';
import {createMaster} from './master.js';

export const SR = 44100;
// Release after the composed note length, in seconds. The clash audit uses the same values.
export const RELEASE = {keys: .35, felt: .3, glass: .5, bass: .15, sub: .12, pad: .35, drone: .6, counter: .35, arp: .3, swell: .02};
const TAIL = 8;
const LEVEL = {keys: 1.9, felt: 1.8, glass: 2.2, counter: 1.9, arp: 1.6, swell: 1.4, bass: .2, kick: .55, brush: .32, snare: .32, rim: .4, hat: .22,
  shaker: .5, pad: .5, drone: .2, sub: .5, texture: .028, rumble: .5, riser: .35};
// Per-style changes to the mix. Jazz Hop leans on a deep bass and a punchy kit.
const MIX = {jazzhop: {bass: 1.05, kick: .9, snare: 1.1, hat: 1.8, keys: 2.0, felt: 1.1}, steady: {hat: 1.3, brush: 1.5, kick: .85}};
// Sampled instruments other than piano sit at different levels.
const INST_GAIN = {piano: 1, vibes: 1.5, guitar: 2.2};
// How much of each part goes to the reverb: pads far back, the lead close.
const SEND = {keys: .3, counter: .45, pad: .9, arp: .55, swell: .7, felt: .15, glass: .3, snare: .2, brush: .2, rim: .15, hat: .05, shaker: .1, riser: .5, rumble: .2};
const KEYS_BUS = new Set(['keys', 'counter', 'arp', 'swell', 'pad']), LEAD_BUS = new Set(['felt', 'glass']);
// Typical composer gains: a hit above this (after humanising) uses the harder piano layer.
const SOFT = {keys: .118, felt: .135, glass: .06, counter: .06, arp: .05, swell: .1};
const ROOM = {
  steady: {scale: 1, feedback: .62, wet: 1.3, tapeDrive: 1.3, duck: .25},
  rainycafe: {scale: 1.1, feedback: .6, wet: 1.4, tapeDrive: 1.25, duck: 0},
  sleep: {scale: 2.4, feedback: .82, wet: 2.2, tapeDrive: 1.1, duck: 0},
  bossa: {scale: 1, feedback: .55, wet: 1.1, tapeDrive: 1.2, duck: .1},
  studypiano: {scale: 1.4, feedback: .66, wet: 1.5, tapeDrive: 1.1, duck: 0},
  chillhop: {scale: .8, feedback: .5, wet: 1, tapeDrive: 1.4, duck: .35},
  nature: {scale: 2, feedback: .75, wet: 1.8, tapeDrive: 1.1, duck: 0},
  jazzhop: {scale: .8, feedback: .5, wet: 1, tapeDrive: 1.45, duck: .35},
  flow: {scale: 2.4, feedback: .8, wet: 2, tapeDrive: 1.15, duck: 0}
};

export function createBank() { return {piano: {L: new Map(), H: new Map()}, vibes: new Map(), guitar: new Map(), bass: new Map(), drums: {}, ambience: new Map()}; }
export function addSample(bank, {kind, name, left, right}) {
  if (kind === 'piano') bank.piano[name.at(-1)].set(parseInt(name), {left, right, note: parseInt(name)});
  else if (kind === 'bass' || kind === 'vibes' || kind === 'guitar') bank[kind].set(+name, {left, right, note: +name});
  else if (kind === 'ambience') bank.ambience.set(name, [left, right || left]);
  else {
    const kind = name.replace(/\d+$/, '');
    (bank.drums[kind] ||= []).push(tidyHit(kind, left, right));
  }
}
// Drum one-shots carry hiss from their source loops. Keep the hit, fade the tail
// exponentially, end it where the fade reaches -40 dB, and take the hats' top above about 9 kHz.
const HIT = {kick: [.08, .07], snare: [.05, .06], rim: [.04, .05], hat: [.015, .03], shaker: [.04, .05]};
function tidyHit(kind, left, right) {
  const [hold, tau] = HIT[kind] || [.05, .06], channels = [left, right || left].map(c => Float32Array.from(c));
  const holdN = Math.round(hold * SR), cut = Math.min(channels[0].length, holdN + Math.round(tau * Math.log(100) * SR));
  const a = 1 - Math.exp(-2 * Math.PI * 9000 / SR);
  for (const c of channels) {
    let low = 0;
    for (let i = 0; i < cut; i++) {
      if (kind === 'hat') { low += a * (c[i] - low); c[i] = low; }
      if (i > holdN) c[i] *= Math.exp(-(i - holdN) / (tau * SR));
    }
  }
  return {left: channels[0].subarray(0, cut), right: right ? channels[1].subarray(0, cut) : null};
}
function nearest(map, pitch) {
  let best = null, distance = Infinity;
  for (const [note, sample] of map) {
    const d = Math.abs(note - pitch);
    if (d < distance) { distance = d; best = sample; }
  }
  return best;
}
const panGains = (gain, pan) => { const p = Math.max(-1, Math.min(1, pan)); return [gain * Math.sqrt((1 - p) * .5), gain * Math.sqrt((1 + p) * .5)]; };
// Linear-interpolated playback with an attack ramp and a linear release after the gate.
// `send` also feeds the reverb bus. `reverse` plays the sample backwards so it swells into its end.
function play(sample, L, R, start, ratio, gain, pan, gate, release, attack, send, reverse = false) {
  const a = sample.left, b = sample.right || sample.left;
  const n = Math.min(L.length - start, Math.floor((a.length - 2) / ratio), gate + release);
  if (n <= 0) return;
  const [gl, gr] = panGains(gain, pan), [SL, SR_, s] = send;
  for (let i = 0; i < n; i++) {
    const pos = (reverse ? n - 1 - i : i) * ratio, k = pos | 0, f = pos - k;
    let env = reverse ? (i / n) ** 2 : i < attack ? i / attack : 1;
    if (!reverse && i > gate) env *= 1 - (i - gate) / release;
    const l = (a[k] + (a[k + 1] - a[k]) * f) * gl * env, r = (b[k] + (b[k + 1] - b[k]) * f) * gr * env;
    L[start + i] += l; R[start + i] += r;
    if (s) { SL[start + i] += l * s; SR_[start + i] += r * s; }
  }
}
const sine = new Float32Array(4097);
for (let i = 0; i < sine.length; i++) sine[i] = Math.sin(2 * Math.PI * i / 4096);
const sin = c => { const p = (c - Math.floor(c)) * 4096, k = p | 0; return sine[k] + (p - k) * (sine[k + 1] - sine[k]); };
// Soft synth layers: a round root-and-fifth pad, Flow's sub drone that glides between roots,
// a low vinyl rumble swell and a rising noise for builds.
function synth(e, beatSec, L, R, start, send, rng) {
  const gate = Math.round(e.duration * beatSec * SR), release = Math.round((RELEASE[e.type] ?? .1) * SR);
  const n = Math.min(L.length - start, gate + release), gain = e.gain * LEVEL[e.type];
  const [gl, gr] = panGains(gain, e.pan), [SL, SR_, s] = send;
  if (e.type === 'rumble' || e.type === 'riser') {
    let lo = 0, lo2 = 0;
    for (let i = 0; i < n; i++) {
      const x = i / n, w = rng() * 2 - 1;
      // Rumble: very low filtered noise that rises and falls. Riser: noise that opens and grows.
      const c = e.type === 'rumble' ? .012 : .02 + .5 * x * x;
      lo += c * (w - lo); lo2 += c * (lo - lo2);
      const v = (e.type === 'rumble' ? lo2 * 6 * Math.sin(Math.PI * x) : (lo - lo2) * x * x * 3);
      L[start + i] += v * gl; R[start + i] += v * gr;
      if (s) { SL[start + i] += v * gl * s; SR_[start + i] += v * gr * s; }
    }
    return;
  }
  const attack = (e.type === 'drone' ? 1.2 : e.type === 'sub' ? .01 : .38) * SR, glide = e.type === 'sub' ? 1 : .9 * SR;
  const f1 = 440 * 2 ** ((e.pitch - 69) / 12), f0 = 440 * 2 ** (((e.from ?? e.pitch) - 69) / 12);
  let phase = 0;
  for (let i = 0; i < n; i++) {
    const t = i / SR, f = i < glide ? f0 * (f1 / f0) ** (i / glide) : f1;
    phase += f / SR;
    const v = e.type === 'sub' ? .9 * sin(phase) + .22 * sin(2 * phase)
      : e.type === 'drone'
      ? .7 * sin(phase) + .5 * sin(2 * phase) + .08 * sin(3 * phase)
      : .70 * sin(phase + .0018 * sin(t * .43)) + .24 * sin(phase * 1.0031) + .16 * sin(2.001 * phase);
    let env = Math.min(1, i / attack);
    if (i > gate) env *= 1 - (i - gate) / release;
    L[start + i] += v * gl * env; R[start + i] += v * gr * env;
    if (s) { SL[start + i] += v * gl * env * s; SR_[start + i] += v * gr * env * s; }
  }
}
// Ambience: distant rain, made here (seeded noise and droplets), looped under the music.
let rainLoop = null;
function rain() {
  if (rainLoop) return rainLoop;
  const n = Math.round(7.3 * SR), rng = random(0x7a1e), out = [new Float32Array(n), new Float32Array(n)];
  for (let ch = 0; ch < 2; ch++) {
    let b = 0, lp = 0;
    for (let i = 0; i < n; i++) {
      const w = rng() * 2 - 1; b = .995 * b + .05 * w; lp += .3 * (w - lp);
      out[ch][i] = .6 * b + .25 * lp;
    }
    for (let k = 0; k < 7.3 * 30; k++) {
      const at = Math.floor(rng() * n), len = 200 + Math.floor(rng() * 500), amp = .3 * rng() ** 3;
      for (let i = 0; i < len; i++) out[ch][(at + i) % n] += amp * (rng() * 2 - 1) * Math.exp(-6 * i / len);
    }
  }
  return rainLoop = out;
}

export function createRenderer(song, seed, bank, style) {
  const beatSec = 60 / song.bpm, frames = 4 * beatSec * SR, tail = Math.ceil(TAIL * SR), room = ROOM[style];
  const rng = random(seed ^ 0x51f15e), master = createMaster(style), level = {...LEVEL, ...MIX[style]};
  const backbeat = rng() < .5 ? 'snare' : 'rim', turn = {};
  // Buffers carried over bar lines: main, keys, lead and send buses, left and right.
  let carry = Array.from({length: 8}, () => new Float32Array(tail)), bar = 0, previous = [];
  const tapeSize = 8192, tape = Array.from({length: 4}, () => new Float32Array(tapeSize));
  let tapeW = 0, clock = 0, lastKick = -1e9, cut = 20000, gainNow = 1, texNow = {};
  const lp = [0, 0, 0, 0], rp = [0, 0, 0, 0];
  const drive = room.tapeDrive, norm = 1 / Math.tanh(drive);
  const delays = [.0341, .0473, .0719, .0893].map(sec => new Float32Array(Math.round(sec * room.scale * SR)));
  const indices = [0, 0, 0, 0], damp = [0, 0, 0, 0];
  const preSize = Math.round(.025 * SR), pre = [new Float32Array(preSize), new Float32Array(preSize)];
  let preW = 0;
  const jitter = (lo, hi) => (lo + (hi - lo) * rng()) * SR;
  function instrument(e) {
    const inst = e.inst || 'piano';
    if (inst !== 'piano' && bank[inst]?.size) return {map: bank[inst], gain: INST_GAIN[inst]};
    const hard = e.gain > (SOFT[e.type] ?? .1) * 1.08;
    return {map: (hard ? [bank.piano.H, bank.piano.L] : [bank.piano.L, bank.piano.H]).find(m => m.size), gain: 1};
  }
  function voice(e, bus) {
    const at = Math.round(e.beat * beatSec * SR);
    // Touch varies by about 15% either way; hats a little less, so the groove stays even.
    const vel = 1 + (rng() - .5) * (e.type === 'hat' || e.type === 'shaker' ? .12 : .3);
    const target = KEYS_BUS.has(e.type) ? bus.keys : LEAD_BUS.has(e.type) ? bus.lead : bus.main;
    const send = [bus.send[0], bus.send[1], SEND[e.type] ?? 0];
    if (e.type === 'pad' || e.type === 'drone' || e.type === 'sub' || e.type === 'rumble' || e.type === 'riser') return synth(e, beatSec, target[0], target[1], Math.max(0, at), send, rng);
    if (e.type === 'texture') return;
    if (['keys', 'felt', 'glass', 'counter', 'arp', 'swell'].includes(e.type)) {
      const {map, gain} = instrument({...e, gain: e.gain * vel}), sample = map && nearest(map, e.pitch);
      if (!sample) return;
      const loose = style === 'jazzhop' ? .006 : .012;
      const start = Math.max(0, at + (e.type === 'swell' ? 0 : Math.round(jitter(-loose, loose)) + (e.strum || 0)));
      play(sample, target[0], target[1], start, 2 ** ((e.pitch - sample.note) / 12), e.gain * vel * (level[e.type] ?? 1.8) * gain, e.pan,
        Math.round(e.duration * beatSec * SR), Math.round((e.releaseSec ?? RELEASE[e.type] ?? .3) * SR), Math.round(.006 * SR), send, e.type === 'swell');
    } else if (e.type === 'bass') {
      const pitch = e.pitch > 45 ? e.pitch - 12 : e.pitch, sample = nearest(bank.bass, pitch);
      // Samples still loading (a style switch in the first seconds): skip the note.
      if (!sample) return;
      play(sample, target[0], target[1], Math.max(0, at + Math.round(jitter(0, .012))), 2 ** ((pitch - sample.note) / 12),
        e.gain * vel * level.bass, 0, Math.round(e.duration * beatSec * SR), Math.round(RELEASE.bass * SR), Math.round(.008 * SR), send);
    } else if (['kick', 'brush', 'snare', 'rim', 'hat', 'shaker'].includes(e.type)) {
      const kind = e.type === 'brush' ? backbeat : e.type, hits = bank.drums[kind];
      if (!hits) return;
      turn[kind] = ((turn[kind] ?? -1) + 1 + Math.floor(rng() * 2)) % hits.length;
      // Drums sit a little behind the beat: a steady lateness with only a few ms of scatter,
      // so the groove stays locked while the keys stay loose. The kick stays in the centre.
      play(hits[turn[kind]], target[0], target[1], Math.max(0, at + Math.round(jitter(.008, .014))), 1, e.gain * vel * level[e.type],
        e.type === 'kick' ? 0 : e.pan, Infinity, 0, Math.round(.002 * SR), send);
    }
  }
  return function next() {
    const len = Math.round((bar + 1) * frames) - Math.round(bar * frames);
    const bufs = carry.map(c => { const b = new Float32Array(len + tail); b.set(c); return b; });
    const bus = {main: [bufs[0], bufs[1]], keys: [bufs[2], bufs[3]], lead: [bufs[4], bufs[5]], send: [bufs[6], bufs[7]]};
    const events = song.nextBar(), info = events.info;
    // Songs without a plan (tools, older paths) get their pads here, as before.
    const pads = info.arranged ? [] : style === 'flow' ? (info.chordStart ? padEvents(info, song.shift, song.bpm, previous, 8) : [])
      : style === 'steady' ? padEvents(info, song.shift, song.bpm, previous) : [];
    // A slight strum: chord notes struck together roll upward by about 12 ms each.
    const chord = events.filter(e => e.type === 'keys').sort((a, b) => a.beat - b.beat || a.pitch - b.pitch);
    chord.forEach((e, i) => { e.strum = Math.round(.012 * SR * (i % 4)); });
    for (const e of events.concat(pads)) voice(e, bus);
    previous = events;
    const kicks = events.filter(e => e.type === 'kick').map(e => clock + Math.round(e.beat * beatSec * SR)).sort((a, b) => a - b);
    // Ambience: each kind (rain, cafe, forest, water) fades between bars to its new level.
    // Recorded loops when loaded; rain falls back to the made-up loop until then.
    const texTargets = {};
    for (const e of events) if (e.type === 'texture') texTargets[e.inst || 'rain'] = e.gain * LEVEL.texture * (bank.ambience.has(e.inst) ? 1.6 : 1);
    const layers = [...new Set([...Object.keys(texTargets), ...Object.keys(texNow)])].map(kind => ({kind, from: texNow[kind] || 0, to: texTargets[kind] || 0,
      loop: bank.ambience.get(kind) || (kind === 'rain' || kind === 'softrain' ? rain() : null)})).filter(l => l.loop && (l.from || l.to));
    const cutTarget = info.filter ?? 20000, gainTarget = info.gain ?? 1, verb = room.wet * (info.reverb ?? 1);
    const stopAt = info.tapeStop != null ? Math.round(info.tapeStop * beatSec * SR) : Infinity;
    const [ml, mr] = bus.main, [kl, kr] = bus.keys, [ll, lr] = bus.lead, [sl, sr] = bus.send;
    let a0 = 0, a1 = 0, a2 = 0, b1 = 0, b2 = 0, ki = 0, duck = 1, d = 0;
    for (let i = 0; i < len; i++, clock++) {
      const x = i / len;
      // Section filter: cutoff and level glide to the plan's values over the bar.
      if (i % 32 === 0) {
        const fc = Math.min(18000, cut * (cutTarget / cut) ** x), w = 2 * Math.PI * fc / SR, al = Math.sin(w) / (2 * .707), cs = Math.cos(w), n0 = 1 + al;
        a0 = (1 - cs) / 2 / n0; a1 = (1 - cs) / n0; a2 = a0; b1 = -2 * cs / n0; b2 = (1 - al) / n0;
      }
      // Sidechain, tape wow and the tape stop change slowly: update them every 16 samples.
      if (i % 16 === 0) {
        while (ki < kicks.length && kicks[ki] <= clock + 15) lastKick = kicks[ki++];
        // Keys and pads dip for a moment under each kick.
        const since = Math.max(0, clock - lastKick) / SR;
        duck = room.duck ? 1 - room.duck * (1 - Math.exp(-since / .004)) * Math.exp(-since / .14) : 1;
        // A slowly wandering delay (wow and flutter); a tape stop pulls the pitch down.
        const t = clock / SR, slow = .0006 + .0003 * sin(.031 * t);
        d = (.004 + slow * sin(.45 * t) + .00008 * sin(6.1 * t)) * SR;
        if (i >= stopAt) d += ((i - stopAt) / SR) ** 2 * .9 * SR;
      }
      tape[0][tapeW] = kl[i]; tape[1][tapeW] = kr[i]; tape[2][tapeW] = ll[i]; tape[3][tapeW] = lr[i];
      const pos = tapeW - Math.min(d, tapeSize - 2) + tapeSize, k = Math.floor(pos), f = pos - k;
      const read = ch => tape[ch][k % tapeSize] + (tape[ch][(k + 1) % tapeSize] - tape[ch][k % tapeSize]) * f;
      tapeW = (tapeW + 1) % tapeSize;
      let L = ml[i] + duck * Math.tanh(drive * read(0)) * norm + read(2);
      let R = mr[i] + duck * Math.tanh(drive * read(1)) * norm + read(3);
      for (const l of layers) {
        const g = l.from + (l.to - l.from) * x, p = clock % l.loop[0].length;
        L += l.loop[0][p] * g; R += l.loop[1][p] * g;
      }
      // Reverb send, after a 25 ms pre-delay: four damped delay lines fed with mid and side.
      const inL = pre[0][preW], inR = pre[1][preW];
      pre[0][preW] = sl[i]; pre[1][preW] = sr[i]; preW = (preW + 1) % preSize;
      const da = damp[0], db = damp[1], dc = damp[2], dd = damp[3];
      const mid = .5 * (inL + inR), side = .5 * (inL - inR), sum = .5 * (da + db + dc + dd);
      for (let j = 0; j < 4; j++) {
        const line = delays[j], p = indices[j], v = line[p];
        line[p] = mid + (j % 2 ? side : -side) + room.feedback * (sum - damp[j]);
        damp[j] += .28 * (v - damp[j]); indices[j] = (p + 1) % line.length;
      }
      L += verb * (.125 * (da + dc) + .048 * db) * 2.2;
      R += verb * (.125 * (db + dd) + .048 * dc) * 2.2;
      const g = gainNow + (gainTarget - gainNow) * x;
      const yl = a0 * L + a1 * lp[0] + a2 * lp[1] - b1 * lp[2] - b2 * lp[3];
      lp[1] = lp[0]; lp[0] = L; lp[3] = lp[2]; lp[2] = yl;
      const yr = a0 * R + a1 * rp[0] + a2 * rp[1] - b1 * rp[2] - b2 * rp[3];
      rp[1] = rp[0]; rp[0] = R; rp[3] = rp[2]; rp[2] = yr;
      ml[i] = yl * g;
      mr[i] = yr * g;
    }
    cut = cutTarget; gainNow = gainTarget; texNow = texTargets;
    master(ml, mr, len);
    carry = bufs.map(b => b.slice(len));
    bar++;
    return {left: ml.subarray(0, len), right: mr.subarray(0, len), events};
  };
}
