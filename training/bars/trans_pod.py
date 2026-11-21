# Watch clips; basic-pitch (Apache-2.0) -> notes -> per-bar chord + 8 eighth-note melody tokens (C major / A minor relative).
# Writes out/notes/<name>.json (raw notes) and out/bars/<name>.json (style-labelled bars). Loops until GEN_DONE and all done.
import os, sys, json, glob, time, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4"); os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from multiprocessing import Pool
QUAL = {"maj7": [0, 4, 7, 11], "m7": [0, 3, 7, 10], "7": [0, 4, 7, 10], "m7b5": [0, 3, 6, 10], "sus": [0, 5, 7, 10], "maj": [0, 4, 7], "m": [0, 3, 7]}
QMAP = {"maj7": 0, "m7": 1, "7": 2, "m7b5": 3, "sus": 4, "maj": 0, "m": 1}
TPL = []
for r in range(12):
    for q, iv in QUAL.items():
        v = np.zeros(12); v[[(r + i) % 12 for i in iv]] = 1; TPL.append((r, q, v / np.linalg.norm(v)))
def chord_of(prof, bass_pc):
    best = None
    for r, q, v in TPL:
        s = float(prof @ v) / (np.linalg.norm(prof) + 1e-9) + (.12 if bass_pc == r else 0) + (.03 if len(QUAL[q]) == 4 else 0)
        if best is None or s > best[0]: best = (s, r, q)
    return best
_M = None
def model():
    global _M
    if _M is None:
        import onnxruntime as ort
        _IS = ort.InferenceSession
        def IS(path, sess_options=None, providers=None, **kw):
            so = sess_options or ort.SessionOptions(); so.intra_op_num_threads = int(os.environ.get("ORT_T", "1")); so.inter_op_num_threads = 1
            return _IS(path, sess_options=so, providers=providers, **kw)
        ort.InferenceSession = IS
        from basic_pitch.inference import Model
        from basic_pitch import ICASSP_2022_MODEL_PATH
        _M = Model(ICASSP_2022_MODEL_PATH)
    return _M
def run(wav, meta):
    import librosa
    from basic_pitch.inference import predict
    bpm = meta["bpm"]; name = meta["name"]
    y, sr = librosa.load(wav, sr=22050, mono=True)
    grid = "beat"
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm=bpm, tightness=400, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    if abs(2 * tempo - bpm) <= 6:
        beats = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2])); tempo *= 2
    if abs(tempo - 2 * bpm) <= 6: beats = beats[::2]; tempo /= 2
    if abs(tempo - bpm) > 6 or len(beats) < 16:            # drumless (Flow) or lost tracker: fixed grid at the prompt BPM
        on = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        t0 = float(on[0]) if len(on) else 0.; beats = np.arange(t0, len(y) / sr, 60. / bpm); grid = "fixed"; tempo = bpm
    _, _, notes = predict(wav, model(), onset_threshold=.5, frame_threshold=.3, minimum_note_length=80)
    notes = [(round(float(s), 3), round(float(e), 3), int(p), round(float(a), 3)) for s, e, p, a, _ in notes]
    json.dump(dict(name=name, style=meta["style"], key=meta["key"], bpm=bpm, notes=notes), open(f"out/notes/{name}.json", "w"))
    w = np.zeros(12)
    for s_, e_, p_, a_ in notes: w[p_ % 12] += (e_ - s_) * a_
    DIA = np.zeros(12); DIA[[0, 2, 4, 5, 7, 9, 11]] = 1
    shift = max(range(12), key=lambda k: (np.roll(w, k) @ DIA, -min(k, 12 - k)))
    if shift > 6: shift -= 12
    notes = [(s, e, p + shift, a) for s, e, p, a in notes]
    low = np.array([s for s, e, p, a in notes if p < 52])
    def score_phase(ph):
        db = beats[ph::4]; return sum(np.min(np.abs(low - t)) < .07 for t in db) if len(low) else 0
    ph = max(range(4), key=score_phase); db = beats[ph::4]
    bars = []
    for i in range(len(db) - 1):
        t0, t1 = db[i], db[i + 1]; step = (t1 - t0) / 8
        prof = np.zeros(12); bass = {}
        for s, e, p, a in notes:
            ov = max(0, min(e, t1) - max(s, t0))
            if ov <= 0: continue
            prof[p % 12] += ov * a * (1.5 if p < 60 else 1)
            if p < 55: bass[p % 12] = bass.get(p % 12, 0) + ov * a
        if prof.sum() < 1e-3: continue
        bpc = max(bass, key=bass.get) if bass else None
        sc, r, q = chord_of(prof, bpc)
        mel = [-1] * 8
        cand = [(s, e, p, a) for s, e, p, a in notes if p >= 62 and t0 - step / 2 <= s < t1 - step / 2 and a > .38 and e - s > .6 * step]
        for k in range(8):
            ins = [c for c in cand if abs(c[0] - (t0 + k * step)) < step / 2]
            if ins:
                s, e, p, a = max(ins, key=lambda c: (c[2], c[3]))
                while p > 84: p -= 12
                while p < 60: p += 12
                n = max(1, int(round((e - s) / step))); mel[k] = int(p)
                for j in range(k + 1, min(8, k + n)):
                    if mel[j] == -1: mel[j] = -2
        bars.append([int(r), QMAP[q], mel])
    roots = [b_[0] for b_ in bars]
    mode = "minor" if roots.count(9) + .5 * roots.count(2) > roots.count(0) + .5 * roots.count(5) else "major"
    return dict(name=name, style=meta["style"], key=meta["key"], mode=mode, shift=shift, bpm=bpm, tempo=round(tempo, 1), grid=grid, bars=bars)
def job(name):
    t0 = time.time()
    try:
        meta = json.load(open(f"clips/{name}.json"))
        r = run(f"clips/{name}.wav", meta); r["secs"] = round(time.time() - t0, 1)
        json.dump(r, open(f"out/bars/{name}.json", "w")); return f"{name} {meta['style']} {r['grid']} {len(r['bars'])} bars {r['secs']}s"
    except Exception as e:
        json.dump(dict(name=name, error=repr(e)[:300]), open(f"out/bars/{name}.json", "w")); return f"{name} ERR {e!r}"[:300]
if __name__ == "__main__":
    os.makedirs("out/notes", exist_ok=True); os.makedirs("out/bars", exist_ok=True)
    pool = Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 24); pending = set()
    while True:
        names = [os.path.basename(f)[:-5] for f in sorted(glob.glob("clips/*.json"))]
        todo = [n for n in names if n not in pending and not os.path.exists(f"out/bars/{n}.json") and os.path.exists(f"clips/{n}.wav")]
        if not todo:
            if os.path.exists("GEN_DONE") and not pending: break
            time.sleep(5); pending = {n for n in pending if not os.path.exists(f"out/bars/{n}.json")}; continue
        pending |= set(todo)
        for msg in pool.imap_unordered(job, todo): print(msg, flush=True)
        pending = {n for n in pending if not os.path.exists(f"out/bars/{n}.json")}
    print("trans done", flush=True)
