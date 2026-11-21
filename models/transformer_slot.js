// Browser/Node inference for the tiny lo-fi transformer. Plain Float32Array math, KV cache, no libraries.
// Decoded float32 weights are shared by every song; each song keeps its own KV cache.
const decoded = new WeakMap();
export function loadModel(meta, f16buf) {
  const W = decoded.get(f16buf) || decode(meta, f16buf);
  decoded.set(f16buf, W);
  return session(meta, W);
}
function decode(meta, f16buf) {
  const h = new Uint16Array(f16buf), W = {};
  const f16 = (u) => { const s = u >> 15, e = (u >> 10) & 31, m = u & 1023;
    const v = e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1 + m / 1024) * 2 ** (e - 15); return s ? -v : v; };
  for (const [k, t] of Object.entries(meta.tensors)) {
    const n = t.shape.reduce((a, b) => a * b, 1), a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = f16(h[t.off + i]); W[k] = { a, shape: t.shape };
  }
  return W;
}
function session(meta, W) {
  const { D, L, H } = meta, hd = D / H;
  let K, Vc, n;
  const mv = (w, b, x) => { const [o, i] = w.shape, y = new Float32Array(o);
    for (let r = 0; r < o; r++) { let s = b.a[r]; const off = r * i; for (let c = 0; c < i; c++) s += w.a[off + c] * x[c]; y[r] = s; } return y; };
  const ln = (x, g, b) => { let m = 0; for (const v of x) m += v; m /= x.length; let v2 = 0; for (const v of x) v2 += (v - m) ** 2; v2 = 1 / Math.sqrt(v2 / x.length + 1e-5);
    return x.map((v, i) => (v - m) * v2 * g.a[i] + b.a[i]); };
  const gelu = (x) => x.map((v) => .5 * v * (1 + Math.tanh(.7978845608 * (v + .044715 * v * v * v))));
  return {
    reset() { K = [...Array(L)].map(() => []); Vc = [...Array(L)].map(() => []); n = 0; },
    step(tok) {
      const p = n, slot = Math.max(0, p - 3) % (meta.SLOT || 10); let x = new Float32Array(D);
      for (let i = 0; i < D; i++) x[i] = W["tok.weight"].a[tok * D + i] + W["pos.weight"].a[p * D + i] + W["slot.weight"].a[slot * D + i];
      for (let l = 0; l < L; l++) {
        const pre = `blocks.${l}.`, hN = ln(x, W[pre + "ln1.weight"], W[pre + "ln1.bias"]);
        const qkv = mv(W[pre + "qkv.weight"], W[pre + "qkv.bias"], hN);
        K[l].push(qkv.slice(D, 2 * D)); Vc[l].push(qkv.slice(2 * D)); const y = new Float32Array(D), T = K[l].length;
        for (let hh = 0; hh < H; hh++) {
          const att = new Float32Array(T); let mx = -1e30;
          for (let t = 0; t < T; t++) { let s = 0; for (let d = 0; d < hd; d++) s += qkv[hh * hd + d] * K[l][t][hh * hd + d]; att[t] = s / Math.sqrt(hd); mx = Math.max(mx, att[t]); }
          let z = 0; for (let t = 0; t < T; t++) { att[t] = Math.exp(att[t] - mx); z += att[t]; }
          for (let t = 0; t < T; t++) for (let d = 0; d < hd; d++) y[hh * hd + d] += att[t] / z * Vc[l][t][hh * hd + d];
        }
        const pr = mv(W[pre + "proj.weight"], W[pre + "proj.bias"], y); for (let i = 0; i < D; i++) x[i] += pr[i];
        const h2 = ln(x, W[pre + "ln2.weight"], W[pre + "ln2.bias"]);
        const o = mv(W[pre + "out.weight"], W[pre + "out.bias"], gelu(mv(W[pre + "fc.weight"], W[pre + "fc.bias"], h2))); for (let i = 0; i < D; i++) x[i] += o[i];
      }
      n++; return mv(W["head.weight"], W["head.bias"], ln(x, W["lnf.weight"], W["lnf.bias"]));
    },
  };
}
