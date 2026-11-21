// Quantized student -> qload.js -> shipped kernel. Compare logits to numpy dequant reference; time one bar (10 tokens).
import fs from 'fs';
import { loadModel } from '../../site/public/engine/neural/transformer.js';
import { dequantToF16 } from '../web/qload.js';
const [bin, js, refp] = process.argv.slice(2);
const b = fs.readFileSync(bin), metaQ = JSON.parse(fs.readFileSync(js));
const t0 = performance.now(); const { meta, f16buf } = dequantToF16(metaQ, b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); const s = loadModel(meta, f16buf); const tl = performance.now() - t0;
s.reset(); let lg; for (const t of [meta.BOS, meta.JAZZHOP, meta.MIN, meta.BAR, meta.CH0 + 2 * meta.NQ + 1]) lg = s.step(t);
const ref = JSON.parse(fs.readFileSync(refp)); let md = 0; for (let i = 0; i < lg.length; i++) md = Math.max(md, Math.abs(lg[i] - ref[i]));
const times = []; for (let bar = 0; bar < 15; bar++) { const t1 = performance.now(); s.step(meta.BAR); s.step(meta.CH0 + 2 * meta.NQ + 1); for (let k = 0; k < 8; k++) s.step(meta.M0 + 2 + 12); times.push(performance.now() - t1); }
times.sort((a, c) => a - c);
console.log(JSON.stringify({ bin, bytes: b.length, loadMs: +tl.toFixed(0), maxAbsDiffVsNumpy: md, barMsMedian: +times[7].toFixed(1), barMsMax: +times[14].toFixed(1) }));
