# Mooziq LM

**An AI that writes an endless lo-fi playlist, live, in your browser.**

Mooziq LM powers [Unlimited Lofi Music](https://lofi.varunlohade.com). A 483K-parameter transformer runs on your device and writes the chords and melody of every song as it plays. A small planner lays out each song's form. Recorded instruments play it, and a small mixing chain makes it sound like a record. Nothing streams from a server. No two songs are the same, and the music never stops.

- **Live demo:** https://lofi.varunlohade.com
- **Model:** 483,653 parameters, 0.97 MB (fp16), plain JavaScript inference with no libraries.
- **Speed:** about 0.3 s to write a whole song on a laptop.
- **Styles:** Jazz Hop, Flow, Steady, Rainy Café, Sleep, Bossa, Study Piano, Chillhop, Nature.
- **Licence:** MIT (code and model weights). Samples keep their own licences (see [CREDITS.md](CREDITS.md)).

## How it works

1. **A song plan.** A small stochastic planner (weighted choices per style) gives each song a plan: sections (intro, themes, break, bridge, outro), how long each one lasts, how intense it is, which instruments play, and how sections join. One gentle peak, then a settled ending.
2. **The model writes the music.** Mooziq LM B483K writes one chord and an eight-step melody per bar, conditioned on the style. It writes the whole song before the first note plays, so the main motif can come back later in the song: restated, moved up a step, cut to a fragment, or carried into the second theme.
3. **Calm rules.** Decoding is constrained, so the music stays background-friendly: diatonic seventh chords, chord tones or the ninth on the beat, no melody note a semitone away from the harmony, a limited number of notes per bar, gentle tempos (50 to 88 BPM by style), and a narrow energy range.
4. **Playback.** A Web Worker renders each bar to audio from real recorded samples: piano, bass, vibraphone, nylon guitar, drums, plus rain, café, forest and stream ambience. An AudioWorklet mixes two decks with crossfades. A master chain per style (filters, compressor and limiter) holds every song at about -19 LUFS.

The site is static: HTML, CSS and ES modules, with a strict Content-Security-Policy and no build step.

## Numbers

Measured on the shipped site (commit `e8a23e8`):

| Check | Result |
|---|---|
| Song write time (210 songs) | median 307 ms, p95 441 ms |
| 24/7 soak: 210 songs, 12.8 h of music | 210 of 210 distinct songs; heap 5.5 MB at song 1 and 5.9 MB at song 210 |
| Loudness over the soak | median -19.2 LUFS (range -20.5 to -17.8) |
| Hidden background tab, 31 min, Chrome | 0 underruns, 0 silent gaps |
| 4x CPU throttle, every style | 0 underruns; manual changes keep a 1 s gap |
| Melody against harmony (clash audit) | 0% semitone clashes in 8 styles, at most 0.2% in Study Piano |
| Site size | 4.9 MB, of which the model is 0.97 MB |

## Run it locally

You need Node.js 20 or later. Optionally install Google Chrome to run the browser checks.

```sh
npm install                 # playwright-core, for the browser checks only
node engine-tools/serve.mjs # serves site/public with the production headers
```

Open the printed address and press Play.

To deploy, point a static host at `site/`. `site/vercel.json` holds the headers used in production.

## Engine tools

Every tool runs from the repository root and writes results to `out/`. None of them play sound.

| Tool | What it does |
|---|---|
| `node engine-tools/render.mjs jazzhop 7 60 out/jazzhop.wav` | Renders a song offline in headless Chrome (style, seed, seconds) |
| `node engine-tools/audit.mjs 11 4242` | Clash audit of whole songs for each seed |
| `node engine-tools/form-metrics.mjs arranged 10` | Form metrics: changes, layers, motif development, endings |
| `node --expose-gc engine-tools/soak.mjs 210` | Writes and renders many songs in Node: memory, loudness, distinctness |
| `node engine-tools/check.mjs jazzhop 60 4 2` | Live playback in muted headless Chrome: underruns, gaps, CSP |
| `node engine-tools/hidden.mjs jazzhop 1860` | The same, as a hidden background tab |
| `node engine-tools/picker.mjs` | Style picker: keyboard, 1 s gap, remembered style, phone layout |
| `node engine-tools/candidates.mjs live q:models/S8-int8/S8.q8` | Compares model candidates inside the real composers |

`engine-tools/build-samples.sh`, `build-samples-2.sh`, `slice-drums.py` and `build-ambience.py` rebuild `site/public/samples/` from the original sources listed in CREDITS.md.

## Models

See [`models/`](models). Each folder holds the weights, a JSON file with the vocabulary and tensor layout, and a `meta.json`.

| Model | Params | Size | Used by the site |
|---|---|---|---|
| `B483K`: bar model (chord + 8-step melody) | 483,653 | 0.97 MB fp16 | yes |
| `S8-int8`: distilled student | 4.84M | 4.9 MB int8 | no |
| `S4-int4`: distilled student | 8.75M | 4.7 MB int4 | no |
| `planner`: song planner, 9 styles | 114,802 | 0.23 MB fp16 | no (candidate) |
| `song-composer`: chord, lead and bass per bar, 9 styles | 506,768 | 1.0 MB fp16 | no (candidate) |

The planner and the song composer need one kernel change: the slot period comes from `meta.SLOT`. See `models/transformer_slot.js`.

## Retrain

`training/` has the full pipeline as it ran, with paths made relative to a work folder. See [`training/README.md`](training/README.md). In short:

1. Generate instrumental lo-fi audio with ACE-Step 1.5 (MIT) on a CUDA GPU.
2. Transcribe it to chords and melodies (basic-pitch, librosa; Demucs stems for full songs).
3. Add public-domain melodies from the music21 corpus.
4. Train the bar model, an ensemble and distilled students, or the planner and song composer.
5. Export fp16 (or int8 and int4) weights, then check the logits against a numpy reference in Node.

The whole pipeline cost about $11 of rented GPU time. [DATA.md](DATA.md) explains where every piece of data came from.

## Limitations

- The model is small, and the rules do much of the work. It writes pleasant, calm background music, not memorable songs.
- It knows only diatonic seventh chords, 4/4 time and eighth-note melodies.
- The training melodies are machine transcriptions of generated audio, so they are noisy.
- The newer styles have little training data (about 68 songs each, for the planner and song composer).
- The ambience loops repeat every 28 to 45 seconds.
- Background playback depends on the browser. Phones may pause audio to save power.
- The learned planner and song composer are not live yet: they write songs too slowly for a 1 s gap on slow devices.
- No listening study has been done. Quality claims rest on automatic checks.

## Credits and licence

Code and model weights: MIT, © 2026 Varun Lohade (see [LICENSE](LICENSE)). The samples and ambience recordings keep their own licences, listed with full attribution in [CREDITS.md](CREDITS.md).

## Jazz Hop: fine-tuned song models

The live Jazz Hop style is planned and composed by a fine-tuned song planner and multi-track composer (`models/jazzhop-finetuned/`, ~1.2 MB total), picked in a blind listening test. The other eight styles use the rule-based song planner with the bar-level model writing the chords and melody. See `site/public/engine/learned.js`.

## Licence

Code and models: MIT (see LICENSE). The audio samples in `site/public/samples/` are not covered by the MIT licence; they keep their own licences, listed in [CREDITS.md](CREDITS.md).
