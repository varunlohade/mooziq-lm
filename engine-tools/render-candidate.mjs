// Silent offline clips of the real site (composers, samples, band, master) with any
// candidate model swapped in. Same seed, style and arrangement for every model.
// Usage: node engine-tools/render-candidate.mjs <out dir> <seconds> <seed> <spec> [spec ...]
//   spec: live | web:<model dir> | q:<path without .bin/.json>   (see engine-tools/candidates.mjs)
// Writes <out dir>/<NAME>-<style>-seed<seed>.wav for jazzhop, steady and flow (or STYLES=jazzhop,...).
import fs from 'node:fs';
import {serve} from './serve.mjs';
import path from 'node:path';
import {chromium} from 'playwright-core';
const [outDir, seconds = '45', seed = '1', ...specs] = process.argv.slice(2);
const root = new URL('..', import.meta.url).pathname;
fs.mkdirSync(outDir, {recursive: true});
// Only the two files of each chosen checkpoint are mounted.
const mounts = new Map([['/qload.js', path.resolve(root, 'models/qload.js')], ['/model-layout.js', path.resolve(root, 'models/model-layout.js')]]);
const candidates = specs.map((spec, n) => {
  if (spec === 'live') return {name:'LIVE', kind:'web', url:'/model/'};
  const web = spec.startsWith('web:');
  if (!web && !spec.startsWith('q:')) throw new Error('Expected live, web:<dir>, or q:<base>');
  const selected = path.resolve(spec.slice(web ? 4 : 2));
  const base = web ? path.join(selected, 'model') : selected;
  const url = '/cand/' + n + '/model';
  for (const ext of ['.json','.bin']) mounts.set(url + ext, fs.realpathSync(base + ext));
  return {name:path.basename(selected), kind:web ? 'web' : 'q', url:web ? '/cand/' + n + '/' : url};
});
const server = await serve(0, mounts);
const browser = await chromium.launch({channel: 'chrome', headless: true, args: ['--mute-audio']});
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${server.address().port}/model/model.json`);
const styles = (process.env.STYLES || 'jazzhop,steady,flow').split(',');
for (const cand of candidates) for (const style of styles) {
  const result = await page.evaluate(async ([cand, style, seed, seconds]) => {
    const composer = await import(`/engine/${{jazzhop: 'jazzhop/compose', flow: 'flow/compose', steady: 'steady/music'}[style]}.js`);
    const {createRenderer, createBank, addSample} = await import('/engine/render.js');
    const {loadSamples} = await import('/engine/loader.js');
    let meta, bytes;
    if (cand.kind === 'web') [meta, bytes] = await Promise.all([fetch(cand.url + 'model.json').then(r => r.json()), fetch(cand.url + 'model.bin').then(r => r.arrayBuffer())]);
    else {
      const {dequantToF16} = await import('/qload.js');
      const [metaQ, raw] = await Promise.all([fetch(cand.url + '.json').then(r => r.json()), fetch(cand.url + '.bin').then(r => r.arrayBuffer())]);
      ({meta, f16buf: bytes} = dequantToF16(metaQ, raw));
    }
    // The live composers prompt with meta.LOFI; models with style tokens get their style's token.
    meta = {...meta, LOFI: meta[{jazzhop: 'JAZZHOP', steady: 'STEADY', flow: 'FLOW'}[style]] ?? meta.LOFI};
    const bank = createBank();
    await loadSamples('/', style, sample => addSample(bank, sample), () => {});
    const song = composer.createSong(meta, bytes, seed), render = createRenderer(song, seed, bank, style);
    const frames = Math.round(seconds * 44100), left = new Float32Array(frames), right = new Float32Array(frames);
    for (let at = 0; at < frames;) {
      const bar = render(), n = Math.min(bar.left.length, frames - at);
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
    for (let i = 0; i < frames; i++) {
      pcm[2 * i] = Math.max(-32768, Math.min(32767, Math.round(l[i] * 32767)));
      pcm[2 * i + 1] = Math.max(-32768, Math.min(32767, Math.round(r[i] * 32767)));
    }
    const u8 = new Uint8Array(pcm.buffer);
    let binary = '';
    for (let i = 0; i < u8.length; i += 32768) binary += String.fromCharCode(...u8.subarray(i, i + 32768));
    return {pcm: btoa(binary), bpm: song.bpm, shift: song.shift};
  }, [cand, style, +seed, +seconds]);
  const data = Buffer.from(result.pcm, 'base64'), header = Buffer.alloc(44);
  header.write('RIFF'); header.writeUInt32LE(36 + data.length, 4); header.write('WAVEfmt ', 8);
  header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20); header.writeUInt16LE(2, 22);
  header.writeUInt32LE(44100, 24); header.writeUInt32LE(176400, 28); header.writeUInt16LE(4, 32);
  header.writeUInt16LE(16, 34); header.write('data', 36); header.writeUInt32LE(data.length, 40);
  const file = path.join(outDir, `${cand.name}-${style}-seed${seed}.wav`);
  fs.writeFileSync(file, Buffer.concat([header, data]));
  console.log(JSON.stringify({file, model: cand.name, style, bpm: result.bpm, shift: result.shift}));
}
await browser.close();
server.close();
