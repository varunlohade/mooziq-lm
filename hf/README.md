---
license: mit
tags:
  - music-generation
  - lofi
  - symbolic-music
  - transformer
  - onnx-free
  - browser
---

# Mooziq LM

Mooziq LM is a family of tiny transformers that write lo-fi music as symbols (chords, melodies, bass lines and song plans) fast enough to run live in a web browser with plain JavaScript. It powers [Unlimited Lofi Music](https://lofi.varunlohade.com), an endless lo-fi playlist written on your device.

- **Demo:** https://lofi.varunlohade.com
- **Code, engine and training pipeline:** https://github.com/varunlohade/mooziq-lm
- **Licence:** MIT

## Models in this repository

| Folder | What it writes | Params | File | Used by the live site |
|---|---|---|---|---|
| [live](live) | Production composer and planner for all eight styles | 506,768 and 114,802 | 1.01 and 0.23 MB fp16 | yes |
| `B483K` | One chord plus an 8-step melody per bar. Style tokens: Jazz Hop, Steady, Flow, Folk | 483,653 | 0.97 MB fp16 | fallback |
| `S8-int8` | Same task; distilled student | 4.84M | 4.9 MB int8 | no |
| `S4-int4` | Same task; distilled student | 8.75M | 4.7 MB int4 | no |
| `planner` | A song plan: sections with type, length, energy, active tracks and transition; 9 styles | 114,802 | 0.23 MB fp16 | original checkpoint |
| `song-composer` | Chord, 8 lead steps and 8 bass steps per bar, conditioned on the plan; 9 styles | 506,768 | 1.0 MB fp16 | original checkpoint |

The live folder pairs each binary with its JSON layout. The older folders have the weights (`.bin`), a JSON with the tensor layout and vocabulary, and a `meta.json`. The token formats are in `FORMAT-quantized.md` and `FORMAT-song-models.md`.

## Architecture

The models are decoder-only transformers with learned token, position and slot embeddings, pre-LayerNorm blocks, GELU MLPs and a KV cache. Weights are raw fp16 (or int8/int4 with per-row or per-group scales). There is no ONNX and no runtime library: `transformer.js` (about 50 lines) runs them in a browser or Node.

```js
import {loadModel} from './transformer.js';
const meta = await (await fetch('B483K/model.json')).json();
const bytes = await (await fetch('B483K/model.bin')).arrayBuffer();
const model = loadModel(meta, bytes);
model.reset();
let logits;
for (const t of [meta.BOS, meta.JAZZHOP, meta.MIN, meta.BAR]) logits = model.step(t);
// logits: one score per token. Sample a chord token (CH0 + root * NQ + quality), then 8 melody tokens.
```

- `planner` and `song-composer` use a slot period from `meta.SLOT`. Use `transformer_slot.js` for them.
- The quantised students load through `qload.js` (`dequantToF16`), then run on the same kernel.

## Intended use

- Generative background music: lo-fi, ambient and calm instrumental styles, in real time on consumer devices.
- Research and teaching on tiny symbolic music models and constrained decoding.

The models are meant to run inside constrained decoding. The live site masks chords to diatonic sevenths, keeps melody notes on the beat to chord tones or the ninth, and blocks semitone clashes with the harmony. Sampled freely, the output is noticeably rougher.

## Training data

The models learn symbols only. Full details are in [DATA.md](https://github.com/varunlohade/mooziq-lm/blob/main/DATA.md).

- **Public-domain melodies** from the music21 corpus (classical works and traditional folk tunes): 3,515 tunes, 43,463 bars.
- **ACE-Step 1.5 generations** (open model, MIT): 1,612 instrumental clips for the bar models and 587 full instrumental songs for the planner and song composer.
  - Transcribed to chords and melodies with basic-pitch (Apache-2.0) and librosa (ISC); Demucs (MIT) stems for full songs.
  - Some prompts named well-known artists as style references. The models learn only chords and melodies transcribed from ACE-Step's own output.
- **No closed-model outputs** (such as text or music from Claude or GPT) are in the training data. For the current release, locally supplied reference playlists informed aggregate harmonic statistics used for fine-tuning. The weights and code are available without those recordings.
- **Distillation:** the students learned from samples of a 5-model ensemble trained on the same data.

## Evaluation

| Model | Held-out loss (ACE) | Held-out loss (public domain) |
|---|---|---|
| B483K | 1.326 | 1.133 |
| S8-int8 | 1.304 | 1.038 |
| S4-int4 | 1.301 | 1.025 |

| Model | Validation loss | Unigram baseline |
|---|---|---|
| planner | 0.629 | 1.516 |
| song-composer | 1.832 | 2.544 |

- **Browser kernel parity** with a float64 numpy reference: at most 7e-7 maximum logit difference.
- **Historical browser benchmarks** (B483K):
  - writing a whole song takes about 0.3 s;
  - 210 songs (12.8 hours) ran with no memory growth;
  - a hidden-tab run of 31 minutes had 0 audio underruns.

## Limitations

- These are small models. In the live product, rules and constrained decoding carry much of the musical quality.
- Vocabulary: diatonic seventh chords, 4/4 time, eighth-note grid, one melody and one bass line.
- The training melodies are machine transcriptions of generated audio, so they are noisy.
- The newer styles have little data (about 68 songs each).
- The planner's track labels for drumless styles were noisy (it sometimes marks drums). Mask drums at decode time for those styles.
- No listening study has been done.

## Citation

```
@software{mooziq_lm_2026,
  author = {Varun Lohade},
  title = {Mooziq LM: tiny transformers for live lo-fi music in the browser},
  year = {2026},
  url = {https://github.com/varunlohade/mooziq-lm}
}
```
