// The browser release supports small decoder-only checkpoints (up to 32M weights).
// Validate the complete layout before allocating or expanding any tensor.
export function validateLayout(meta, byteLength, quantized = false) {
  const positive = x => Number.isSafeInteger(x) && x > 0;
  const {D, H, L, T, V, SLOT = 10} = meta;
  if (![D,H,L,T,V,SLOT].every(positive) || D % H || D > 1024 || L > 64 ||
      V > 8192 || T > 4096 || SLOT > T || T * L * D > 8_000_000) {
    throw new RangeError('Unsupported model dimensions');
  }
  const expected = new Map([
    ['tok.weight',[V,D]], ['pos.weight',[T,D]], ['slot.weight',[SLOT,D]],
    ['head.weight',[V,D]], ['head.bias',[V]], ['lnf.weight',[D]], ['lnf.bias',[D]]
  ]);
  for (let i = 0; i < L; i++) {
    for (const [part, shape] of Object.entries({
      ln1:[D], ln2:[D], qkv:[3*D,D], proj:[D,D], fc:[4*D,D], out:[D,4*D]
    })) {
      expected.set('blocks.' + i + '.' + part + '.weight', shape);
      expected.set('blocks.' + i + '.' + part + '.bias', [shape[0]]);
    }
  }
  if (!meta.tensors || Object.keys(meta.tensors).length !== expected.size) throw new Error('Unexpected tensor count');
  function span(offset, bytes) {
    if (!Number.isSafeInteger(offset) || offset < 0 || offset > byteLength - bytes) {
      throw new RangeError('Tensor outside weight file');
    }
  }
  let total = 0;
  for (const [name, shape] of expected) {
    const tensor = meta.tensors[name];
    if (!Array.isArray(tensor?.shape) || tensor.shape.length !== shape.length ||
        !shape.every((n, i) => tensor.shape[i] === n)) throw new Error('Invalid tensor shape for ' + name);
    const n = shape.reduce((a,b) => a*b);
    total += n;
    if (total > 32_000_000) throw new RangeError('Checkpoint exceeds browser weight budget');
    if (!quantized) { span(tensor.off * 2, n * 2); if (!Number.isSafeInteger(tensor.off)) throw new Error('Invalid fp16 offset'); }
    else if (tensor.dtype === 'f16') span(tensor.off, n * 2);
    else {
      if (!['int4','int8'].includes(tensor.dtype) || !positive(tensor.group)) throw new Error('Invalid quantization');
      const groups = Math.ceil(n / tensor.group);
      if (tensor.nscale !== groups) throw new RangeError('Invalid scale count');
      span(tensor.off, tensor.dtype === 'int4' ? Math.ceil(n / 2) : n);
      span(tensor.scale_off, groups * 2);
    }
  }
}
