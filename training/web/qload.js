import {validateLayout} from './model-layout.js';
// Load a tinylofi-q8 / tinylofi-q4 model and hand it to the SHIPPED kernel (engine/neural/transformer.js) unchanged.
// It dequantizes once at load: weight = q * scale (one fp16 scale per `group` input elements), then re-packs as fp16
// in the layout loadModel() already reads. Format: see FORMAT.md next to this file.
const f16 = (u) => { const s = u >> 15, e = (u >> 10) & 31, m = u & 1023;
  const v = e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1 + m / 1024) * 2 ** (e - 15); return s ? -v : v; };
const toF16 = (() => { const f = new Float32Array(1), i = new Uint32Array(f.buffer);
  return (v) => { f[0] = v; const x = i[0], s = (x >>> 16) & 0x8000; let e = ((x >>> 23) & 255) - 112, m = x & 0x7fffff;
    if (e <= 0) { if (e < -10) return s; m |= 0x800000; const sh = 14 - e; let r = m >> sh; if ((m >> (sh - 1)) & 1) r++; return s | r; }
    if (e >= 31) return s | 0x7c00; let r = s | (e << 10) | (m >> 13); if (m & 0x1000) r++; return r; }; })();
export function dequantToF16(metaQ, buf) {
  validateLayout(metaQ, buf.byteLength, true);
  const u8 = new Uint8Array(buf), dv = new DataView(buf), tensors = {}; let total = 0;
  for (const [k, t] of Object.entries(metaQ.tensors)) { const n = t.shape.reduce((a, b) => a * b, 1); tensors[k] = { shape: t.shape, off: total }; total += n; }
  const out = new Uint16Array(total);
  for (const [k, t] of Object.entries(metaQ.tensors)) {
    const n = t.shape.reduce((a, b) => a * b, 1), o = tensors[k].off;
    if (t.dtype === 'f16') { for (let i = 0; i < n; i++) out[o + i] = dv.getUint16(t.off + 2 * i, true); continue; }
    const g = t.group;
    for (let i = 0; i < n; i++) {
      const sc = f16(dv.getUint16(t.scale_off + 2 * Math.floor(i / g), true));
      let q;
      if (t.dtype === 'int8') q = (u8[t.off + i] << 24) >> 24;
      else { const b = u8[t.off + (i >> 1)]; q = ((i & 1) ? (b >> 4) : (b & 15)) - 8; }
      out[o + i] = toF16(q * sc);
    }
  }
  const { tensors: _drop, ...rest } = metaQ;
  return { meta: { ...rest, tensors }, f16buf: out.buffer };
}
