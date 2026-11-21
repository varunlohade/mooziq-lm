// Silent offline render of one song through the site's own modules (stereo engine PCM),
// in headless Chrome. Usage: node tools/render.mjs <style id> <seed> <seconds> <out.wav>
import fs from 'node:fs';
import {serve} from './serve.mjs';
import {chromium} from 'playwright-core';
const [style = 'steady', seed = '1', seconds = '60', file = 'out/render.wav'] = process.argv.slice(2);
fs.mkdirSync(new URL('../out/', import.meta.url), {recursive:true});
const server = await serve();
const browser = await chromium.launch({channel: 'chrome', headless: true, args: ['--mute-audio']});
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${server.address().port}/model/model.json`);
const result = await page.evaluate(async ([style, seed, seconds, useLearned]) => {
  const composer = (await import('/engine/styles/index.js')).STYLES[style];
  const {PRESETS} = await import('/engine/plan.js');
  const {createRenderer, createBank, addSample} = await import('/engine/render.js');
  const {createArrangedSong} = await import('/engine/arranger.js');
  const {loadSamples, loadAmbience} = await import('/engine/loader.js');
  const [meta, bytes] = await Promise.all([fetch('/model/model.json').then(r => r.json()), fetch('/model/model.bin').then(r => r.arrayBuffer())]);
  const bank = createBank();
  await loadSamples('/', style, sample => addSample(bank, sample), () => {}, Promise.resolve());
  await loadAmbience('/', PRESETS[style].textures, sample => addSample(bank, sample));
  // The same planned, arranged song the site plays.
  // Fetch the production models unless explicitly rendering the preset fallback.
  const part = async name => ({meta: await fetch(`/model/learned/${name}.json`).then(r => r.json()), bytes: await fetch(`/model/learned/${name}.bin`).then(r => r.arrayBuffer())});
  const learned = useLearned ? {planner: await part('planner'), composer: await part('songlm_s')} : undefined;
  const song = createArrangedSong(composer, meta, bytes, seed, style, undefined, undefined, learned), render = createRenderer(song, seed, bank, style);
  const bars = [];
  const frames = Math.round(seconds * 44100), left = new Float32Array(frames), right = new Float32Array(frames);
  for (let at = 0; at < frames;) {
    const bar = render(), n = Math.min(bar.left.length, frames - at), info = bar.events.info;
    bars.push({at: at / 44100, section: info.sectionType, energy: info.energy, layers: [...new Set(bar.events.map(e => e.type))].length, candy: bar.events.filter(e => e.candy).length});
    left.set(bar.left.subarray(0, n), at); right.set(bar.right.subarray(0, n), at);
    at += n;
  }
  // Same final ceiling as engine/worklet.js.
  const KNEE = 10 ** (-4 / 20), TOP = 10 ** (-3.3 / 20);
  const ceiling = x => { const a = Math.abs(x); return a <= KNEE ? x : Math.sign(x) * (KNEE + (TOP - KNEE) * Math.tanh((a - KNEE) / (TOP - KNEE))); };
  for (let i = 0; i < frames; i++) { left[i] = ceiling(left[i]); right[i] = ceiling(right[i]); }
  const ctx = new OfflineAudioContext(2, frames, 44100), buffer = ctx.createBuffer(2, frames, 44100);
  buffer.copyToChannel(left, 0); buffer.copyToChannel(right, 1);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.start();
  const out = await ctx.startRendering(), l = out.getChannelData(0), r = out.getChannelData(1);
  const pcm = new Int16Array(frames * 2);
  let peak = 0;
  for (let i = 0; i < frames; i++) {
    peak = Math.max(peak, Math.abs(l[i]), Math.abs(r[i]));
    pcm[2 * i] = Math.max(-32768, Math.min(32767, Math.round(l[i] * 32767)));
    pcm[2 * i + 1] = Math.max(-32768, Math.min(32767, Math.round(r[i] * 32767)));
  }
  const bytesOut = new Uint8Array(pcm.buffer);
  let binary = '';
  for (let i = 0; i < bytesOut.length; i += 32768) binary += String.fromCharCode(...bytesOut.subarray(i, i + 32768));
  return {pcm: btoa(binary), peak, bpm: song.bpm, shift: song.shift, bars, plan: song.plan, development: song.development};
}, [style, +seed, +seconds, process.env.LEARNED !== '0']);
await browser.close();
server.close();
const data = Buffer.from(result.pcm, 'base64'), header = Buffer.alloc(44);
header.write('RIFF'); header.writeUInt32LE(36 + data.length, 4); header.write('WAVEfmt ', 8);
header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20); header.writeUInt16LE(2, 22);
header.writeUInt32LE(44100, 24); header.writeUInt32LE(176400, 28); header.writeUInt16LE(4, 32);
header.writeUInt16LE(16, 34); header.write('data', 36); header.writeUInt32LE(data.length, 40);
fs.writeFileSync(file, Buffer.concat([header, data]));
fs.writeFileSync(file.replace(/\.wav$/, '.json'), JSON.stringify({bpm: result.bpm, plan: result.plan, bars: result.bars}));
console.log(JSON.stringify({file, style, seed: +seed, bpm: result.bpm, shift: result.shift, peakDbfs: +(20 * Math.log10(result.peak)).toFixed(2)}));
