# Training pipeline (read-only copies)

These are the scripts that produced the Mooziq LM models, as they ran. Only the paths changed: they are now relative to a work folder (`${WORK:-work}` in the shell scripts; Python scripts expect to run from that folder). They were written for one rented CUDA GPU (an RTX 6000-class card) and a Mac for export and checks. Expect to adapt them. See [DATA.md](../DATA.md) for where every piece of data came from.

## `round1/`: the first small model
1. `prep_pd_corpus.py`: music21 corpus to `melodies.txt` (public-domain melodies in a compact text format).
2. `prep_pd.py`: melodies to bars (chord label plus eight melody steps).
3. `gen_ace.py`: instrumental clips with ACE-Step 1.5 (fixed seeds, BPM and key).
4. `transcribe.py`: clips to bars with basic-pitch and librosa.
5. `train.py` (with `common.py`): train; `export_web.py`: write fp16 `model.bin` and `model.json`.

## `bars/`: the shipped bar model (B483K), the ensemble and the students
1. `gen_pod.py` (ACE-Step clips), `trans_pod.py` (transcription), `build_data.py` (merge into one bar dataset).
2. `final.sh`, then `final2.sh`: teachers and baselines (`trainer.py`, `train_pod.py`, `lm.py`), a teacher sample pool (`sample_pool.py`), distilled int8 and int4 students (`distill.py`), held-out loss and music audit (`audit.py`, `ens_eval.py`).
3. `export_web.py`: npz to the browser format; `parity.mjs` / `parity_q.mjs` with `parity_ref*.py`: check the browser kernel's logits against a float64 numpy reference.

## `songs/`: the planner and the song composer
1. `setup.sh`: ACE-Step 1.5, basic-pitch and Demucs.
2. `run.sh`, `run2.sh`, `run3b.sh`: generate full songs (`gen_song.py`, `gen_new.py`), separate stems (`stems_pod.py`), transcribe sections and tracks (`song_trans.py`), build the dataset (`build_songs.py`), check vocals (`vcheck.py`), then train (`songmodels.py planner ...`, `songmodels.py songlm ...`).
3. `parity_song.mjs` with `parity_ref_song.py`: kernel parity. The format is in `FORMAT.md`.

## `web/`
Formats (`FORMAT.md`), the int8/int4 loader (`qload.js`) and the kernel with a configurable slot period (`transformer_slot.js`).
