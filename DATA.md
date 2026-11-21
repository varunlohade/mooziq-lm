# Mooziq LM: training data provenance

Mooziq LM models learn **symbols, not audio**: chords, eighth-note melodies and bass lines, plus song sections. Every training example came from one of two sources:

1. Public-domain melodies from the music21 corpus.
2. Instrumental audio generated with the open model ACE-Step 1.5, then transcribed to symbols.

**No text or music written by a closed model (such as Claude or GPT) is training data.** An AI coding assistant helped write the code and the ACE-Step text prompts. The prompts are inputs to ACE-Step; they are not training data.

## 1. Public-domain melodies (style token `FOLK`)

- **Source:** the corpus that ships with [music21](https://github.com/cuthbertLab/music21) (toolkit licence: BSD-3-Clause).
- **Subsets used:** Bach, Mozart, Beethoven, Haydn, Robert and Clara Schumann, Chopin, Handel, Schubert, Joplin, Corelli, C. P. E. Bach, Weber, Verdi, plus the folk collections O'Neill's 1850, Ryan's Mammoth Collection, the Essen Folksong Collection, Aird's Airs and the Nottingham dataset.
- **Copyright status:** the compositions are public-domain works (pre-1920 composers and traditional tunes). The machine-readable encodings come bundled with music21.
- **Processing:** `training/round1/prep_pd_corpus.py` keeps the first part of each piece, transposes it to C major or A minor, and writes a text encoding. `prep_pd.py` keeps 4/4 tunes (pairing 2/4 bars) and labels each bar with the best-fitting diatonic seventh chord. That gives 3,515 tunes and 43,463 bars.

## 2. ACE-Step 1.5 generations (style tokens for the lo-fi styles)

- **Generator:** [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5), an open text-to-music model released under the MIT licence, run on rented CUDA GPUs. Every clip is instrumental (`[Instrumental]`, no vocals requested), with fixed seeds, BPM and key.
- **Prompts:** plain style descriptions, such as "calm lofi hip hop beat, soft Rhodes seventh chords, gentle dusty drums". Some of the original Jazz Hop prompts used well-known artist names as style descriptors; the published scripts replace them with plain genre descriptions (for example "mellow Japanese jazz hip hop", "off-grid swung jazz hop").
- **Transcription:**
  - [basic-pitch](https://github.com/spotify/basic-pitch) (Apache-2.0) for notes.
  - [librosa](https://librosa.org) (ISC) for beats and onsets.
  - For full songs, [Demucs](https://github.com/facebookresearch/demucs) `htdemucs` (MIT) stems first.
  - The key is detected from the audio.
  - Vocal-stem energy was logged for every song. The song with the most was regenerated and its vocal stem transcribed with [Whisper](https://github.com/openai/whisper) (MIT) (`vcheck.py`). A song with real lyrics would have been dropped (via `drop.txt`).

| Model | Data |
|---|---|
| Round 1 (prototype) | 39 clips, 797 bars |
| **B483K** (shipped) and the bar-model ensemble | 1,612 clips of 90 s, giving 39,277 bars (Jazz Hop 888 clips, Steady 404, Flow 320), plus the round-1 clips and the public-domain bars |
| **S8 / S4** students | distilled from samples of a 5-model ensemble (B483K, a 25M teacher and three baselines), all trained on the data above |
| **Planner and song composer** | 587 full songs of 150-240 s (Jazz Hop 99, Steady 44, Flow 36; Rainy Café, Sleep, Bossa, Study Piano, Chillhop and Nature 68 each). Sections come from stem energy and chroma novelty on a 4-bar grid; active tracks come from stem energy |

## Reference playlists used in the live release

The fine-tune uses the symbolic song corpus above. Separately, `train-reference-harmony.py` estimates chord-root movement and recurrence from multiple local playlists. It gives each playlist equal weight, evaluates by holding out whole playlists and exports aggregate probabilities. The composer fine-tune can weight examples using these statistics. The register detector holds out whole transcribed songs before evaluating injected octave dips. Reference tracks and per-track analysis are kept local.

## 3. Recorded samples (playback only, not training data)

The piano, bass, vibraphone, guitar, drums and ambience the site plays are CC0, public-domain or CC BY recordings, listed in [CREDITS.md](CREDITS.md). None of them were used to train a model.

## What is not included

- **No raw audio.** The repository has none of the ACE-Step clips, Demucs stems or the original sample libraries. The scripts in `training/` regenerate the clips and the transcriptions from fixed seeds.
- **No transcribed datasets.** The bar and song JSON files are not included; the scripts rebuild them.
- **No teacher or ensemble checkpoints.** The 25M and 85M teachers and the larger baselines are not included. The released models are listed in `models/`.
- The reference playlists used for the current release are not distributed.

## Tools used in training

PyTorch (BSD-3-Clause), NumPy (BSD-3-Clause), music21 (BSD-3-Clause), ACE-Step 1.5 (MIT), basic-pitch (Apache-2.0), librosa (ISC), Demucs (MIT) and Whisper (MIT, used only for the vocal check). Inference in the browser uses no third-party code.
