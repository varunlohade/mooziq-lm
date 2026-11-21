// Load web/NAME/model.{bin,json} with the SHIPPED kernel (site/public/engine/neural/transformer.js, read-only) and
// compare next-token logits to a float64 numpy reference written by parity_ref.py. Also time 10 tokens (one bar).
import fs from 'fs';
import { loadModel } from '../../site/public/engine/neural/transformer.js';
const dir = process.argv[2], style = process.argv[3] || 'JAZZHOP';
const meta = JSON.parse(fs.readFileSync(`${dir}/model.json`)); const b = fs.readFileSync(`${dir}/model.bin`);
const s = loadModel(meta, b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); s.reset();
const seq = [meta.BOS, meta[style], meta.MIN, meta.BAR, meta.CH0 + 2 * meta.NQ + 1];
let lg; for (const t of seq) lg = s.step(t);
const ref = JSON.parse(fs.readFileSync(`${dir}/parity_ref.json`));
let md = 0; for (let i = 0; i < lg.length; i++) md = Math.max(md, Math.abs(lg[i] - ref[i]));
const t0 = performance.now(); for (let k = 0; k < 10; k++) s.step(meta.M0 + 2 + 12); const ms = performance.now() - t0;
console.log(JSON.stringify({ dir, V: lg.length, maxAbsDiff: md, msPer10Tokens: +ms.toFixed(1) }));
