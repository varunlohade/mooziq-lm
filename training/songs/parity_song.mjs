// Song models (planner / songlm) in a copy of the shipped kernel whose only change is slot = (p-3) % meta.SLOT.
// Checks logits against songs/parity_ref_song.py and times one songlm bar (20 tokens) or one planner section (5 tokens).
import fs from 'fs';
import { loadModel } from '../web/transformer_slot.js';
const [base, refp] = process.argv.slice(2);
const meta = JSON.parse(fs.readFileSync(base + '.json')), b = fs.readFileSync(base + '.bin');
const s = loadModel(meta, b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); s.reset();
const seq = meta.kind === 'planner' ? [meta.BOS, meta.STY0, meta.MIN, meta.TYPE0, meta.LEN0 + 1] : [meta.BOS, meta.STY0, meta.MIN, meta.BAR, meta.SEC0 + 5, meta.ACT0 + 15, meta.CH0 + 11];
let lg; for (const t of seq) lg = s.step(t);
const ref = JSON.parse(fs.readFileSync(refp)); let md = 0; for (let i = 0; i < lg.length; i++) md = Math.max(md, Math.abs(lg[i] - ref[i]));
const per = meta.SLOT, times = [];
for (let r = 0; r < 9; r++) { const t0 = performance.now(); for (let k = 0; k < per; k++) s.step(meta.kind === 'planner' ? meta.TYPE0 : (k === 0 ? meta.BAR : meta.M0 + 2)); times.push(performance.now() - t0); }
times.sort((a, c) => a - c);
console.log(JSON.stringify({ base, params: meta.params, bytes: b.length, maxAbsDiff: md, msPerUnitMedian: +times[4].toFixed(1), unit: meta.kind === 'planner' ? 'section (5 tokens)' : 'bar (20 tokens)' }));
