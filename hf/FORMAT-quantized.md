# Quantized student format (tinylofi-q8 / tinylofi-q4)

Two files per model: `NAME.qB.bin` (weights) and `NAME.qB.json` (index). `B` is 8 or 4.

## Network and tokens
The network is the shipped one: the same tensor names, shapes and math as `transformer.js`.
- `D`, `L`, `H`, `V`, `T` are in the JSON. `V` is 101: the shipped 99 tokens plus `JAZZHOP` = 99 and `FLOW` = 100.
- `STEADY` is the old `LOFI` = 2, and `FOLK` = 3 (public-domain tunes).
- A song starts with `[BOS, style, MAJ|MIN]`. The style is `JAZZHOP`, `STEADY` (= `LOFI`) or `FLOW`.
- Each bar is 10 tokens: `BAR`, chord (`CH0 + root*NQ + quality`), then 8 eighth-note melody tokens. These are unchanged.

## JSON
- `format`: `"tinylofi-q8"` or `"tinylofi-q4"`.
- `bits`, `group`, `D`, `L`, `H`, `V`, `T`: model settings.
- `PAD BOS LOFI STEADY FOLK JAZZHOP FLOW MAJ MIN BAR CH0 M0 NQ LO HI`: token ids.
- `tensors`: `name -> {shape, dtype, off, [group, scale_off, nscale]}`. Offsets are bytes from the start of the .bin. Every tensor starts 4-byte aligned.

## Tensor encodings (little endian, row-major)
- `f16`: IEEE half floats. Biases and LayerNorm weights use this encoding.
- `int8`: one signed byte `q` in [-128, 127] per weight.
- `int4`: two weights per byte. The low nibble holds the even index and the high nibble holds the odd index. Each nibble stores `q + 8`, with `q` in [-8, 7].
- Scales: `nscale` fp16 values at `scale_off`. Element `i` of the flattened tensor uses scale `floor(i / group)`. In other words, there is one scale per `group` consecutive input elements of a row. If `group` equals the row length, the tensor has one scale per row.
- The weight is `q * scale`. The scheme is symmetric, with no zero point.

All 2-D tensors are quantized: `tok`, `pos`, `slot`, `qkv`, `proj`, `fc`, `out` and `head`. Everything else is `f16`.

## Loading in the browser
`qload.js` exports `dequantToF16(metaQ, arrayBuffer)`. It returns `{meta, f16buf}` in the shipped fp16 layout, so `loadModel(meta, f16buf)` from the shipped kernel runs unchanged. The model dequantizes once at load. RAM use then matches an fp32 model of the same size, because the kernel decodes to Float32Array.

`training/bars/parity_q.mjs` (in the GitHub repo) checks the output against a float64 numpy reference (`parity_ref_q.py`) and times one bar.
