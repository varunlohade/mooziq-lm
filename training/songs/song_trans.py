# Stems -> multi-track symbolic song. basic-pitch (Apache-2.0) on bass and other; drum onsets classified by band energy.
# Segmentation: ACE-Step's LM output carries no section timestamps, so sections come from stem energy + chroma novelty on a 4-bar grid.
import os, sys, json, glob, time, numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from multiprocessing import Pool
QUAL = {"maj7": [0, 4, 7, 11], "m7": [0, 3, 7, 10], "7": [0, 4, 7, 10], "m7b5": [0, 3, 6, 10], "sus": [0, 5, 7, 10], "maj": [0, 4, 7], "m": [0, 3, 7]}
QMAP = {"maj7": 0, "m7": 1, "7": 2, "m7b5": 3, "sus": 4, "maj": 0, "m": 1}
TPL = []
for r in range(12):
    for q, iv in QUAL.items():
        v = np.zeros(12); v[[(r + i) % 12 for i in iv]] = 1; TPL.append((r, q, v / np.linalg.norm(v)))
TRACKS = ["drums", "bass", "keys", "lead"]
_M = None
def model():
    global _M
    if _M is None:
        import onnxruntime as ort
        _IS = ort.InferenceSession
        def IS(path, sess_options=None, providers=None, **kw):
            so = sess_options or ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
            return _IS(path, sess_options=so, providers=providers, **kw)
        ort.InferenceSession = IS
        from basic_pitch.inference import Model
        from basic_pitch import ICASSP_2022_MODEL_PATH
        _M = Model(ICASSP_2022_MODEL_PATH)
    return _M
def notes_of(path, **kw):
    from basic_pitch.inference import predict
    _, _, n = predict(path, model(), **kw)
    return [(round(float(s), 3), round(float(e), 3), int(p), round(float(a), 3)) for s, e, p, a, _ in n]
def chord_of(prof, bass_pc):
    best = None
    for r, q, v in TPL:
        s = float(prof @ v) / (np.linalg.norm(prof) + 1e-9) + (.12 if bass_pc == r else 0) + (.03 if len(QUAL[q]) == 4 else 0)
        if best is None or s > best[0]: best = (s, r, q)
    return best
def run(name):
    import librosa
    meta = json.load(open(f"songs/{name}.json")); bpm = meta["bpm"]; sd = f"stems/{name}"
    st = {k: librosa.load(f"{sd}/{k}.wav", sr=22050)[0] for k in ("drums", "bass", "other", "vocals")}
    mix = st["drums"] + st["bass"] + st["other"]; sr = 22050
    # beat grid (drums stem first, then mix, then a fixed grid at the prompt BPM)
    grid = None
    for src, lab in ((st["drums"], "beat:drums"), (mix, "beat:mix")):
        tempo, beats = librosa.beat.beat_track(y=src, sr=sr, start_bpm=bpm, tightness=400, units="time"); tempo = float(np.atleast_1d(tempo)[0])
        if abs(2 * tempo - bpm) <= 6: beats = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2])); tempo *= 2
        if abs(tempo - 2 * bpm) <= 6: beats = beats[::2]; tempo /= 2
        if abs(tempo - bpm) <= 6 and len(beats) > 32: grid = lab; break
    if grid is None:
        on = librosa.onset.onset_detect(y=mix, sr=sr, units="time"); t0 = float(on[0]) if len(on) else 0.
        beats = np.arange(t0, len(mix) / sr, 60. / bpm); grid = "fixed"; tempo = bpm
    else:   # ACE songs keep a steady tempo: extend the tracked grid (median period, tracked phase) over the whole song
        per = float(np.median(np.diff(beats))); st0 = beats[0] - per * np.floor(beats[0] / per)
        beats = np.arange(st0, len(mix) / sr - per / 2, per); grid += "+extended"
    bn = notes_of(f"{sd}/bass.wav", onset_threshold=.5, frame_threshold=.3, minimum_note_length=80, maximum_frequency=400)
    on_ = notes_of(f"{sd}/other.wav", onset_threshold=.5, frame_threshold=.3, minimum_note_length=80)
    w = np.zeros(12)
    for s_, e_, p_, a_ in bn + on_: w[p_ % 12] += (e_ - s_) * a_
    DIA = np.zeros(12); DIA[[0, 2, 4, 5, 7, 9, 11]] = 1
    shift = max(range(12), key=lambda k: (np.roll(w, k) @ DIA, -min(k, 12 - k)))
    if shift > 6: shift -= 12
    bn = [(s, e, p + shift, a) for s, e, p, a in bn]; on_ = [(s, e, p + shift, a) for s, e, p, a in on_]
    low = np.array([s for s, e, p, a in bn])
    ph = max(range(4), key=lambda ph: sum(np.min(np.abs(low - t)) < .07 for t in beats[ph::4]) if len(low) else 0); db = beats[ph::4]
    # drum hits: onsets on the drum stem, band energies -> kick / snare / hat
    D = np.abs(librosa.stft(st["drums"], n_fft=1024, hop_length=256)); fr = librosa.fft_frequencies(sr=sr, n_fft=1024)
    bands = [D[fr < 150].sum(0), D[(fr >= 150) & (fr < 4000)].sum(0), D[fr >= 6000].sum(0)]
    ons = librosa.onset.onset_detect(y=st["drums"], sr=sr, hop_length=256, units="frames", backtrack=False)
    p90 = [np.percentile(b[ons], 90) + 1e-9 if len(ons) else 1. for b in bands]
    hits = []
    for f in ons:
        e = [bands[i][f:f + 3].max() / p90[i] for i in range(3)]
        lab = [i for i in range(3) if e[i] > .45 and e[i] >= .5 * max(e)]
        if lab: hits.append((f * 256 / sr, lab))
    rms = {k: librosa.feature.rms(y=v, hop_length=512)[0] for k, v in st.items()}; tr = librosa.times_like(rms["drums"], sr=sr, hop_length=512)
    chroma = librosa.feature.chroma_cqt(y=st["other"] + st["bass"], sr=sr, hop_length=512)
    def seg(x, t0, t1): i0, i1 = np.searchsorted(tr, t0), max(np.searchsorted(tr, t0) + 1, np.searchsorted(tr, t1)); return x[..., i0:i1]
    bars = []
    for i in range(len(db) - 1):
        t0, t1 = db[i], db[i + 1]; step = (t1 - t0) / 8
        prof = np.zeros(12); bass = {}
        for s, e, p, a in on_ + bn:
            ov = max(0, min(e, t1) - max(s, t0))
            if ov > 0: prof[p % 12] += ov * a * (1.5 if p < 60 else 1)
        for s, e, p, a in bn:
            ov = max(0, min(e, t1) - max(s, t0))
            if ov > 0: bass[p % 12] = bass.get(p % 12, 0) + ov * a
        r, q = (chord_of(prof, max(bass, key=bass.get) if bass else None)[1:]) if prof.sum() > 1e-3 else (0, "sus")
        def line(ns, lo, hi, minp, pick):
            mel = [-1] * 8; cand = [(s, e, p, a) for s, e, p, a in ns if t0 - step / 2 <= s < t1 - step / 2 and a > .3 and e - s > .5 * step and p >= minp]
            for k in range(8):
                ins = [c for c in cand if abs(c[0] - (t0 + k * step)) < step / 2]
                if ins:
                    s, e, p, a = pick(ins)
                    while p > hi: p -= 12
                    while p < lo: p += 12
                    mel[k] = int(p)
                    for j in range(k + 1, min(8, k + max(1, int(round((e - s) / step))))):
                        if mel[j] == -1: mel[j] = -2
            return mel
        lead = line(on_, 60, 84, 62, lambda c: max(c, key=lambda z: (z[2], z[3])))
        bassl = line(bn, 28, 55, 0, lambda c: min(c, key=lambda z: (z[2], -z[3])))
        dr = [0] * 8
        for t, lab in hits:
            k = int(round((t - t0) / step))
            if 0 <= k < 8 and t0 - step / 2 <= t < t1 - step / 2:
                for l in lab: dr[k] |= 1 << l
        e = {k: float(seg(v, t0, t1).mean()) if seg(v, t0, t1).size else 0. for k, v in rms.items()}
        bars.append(dict(chord=[int(r), QMAP[q]], lead=lead, bass=bassl, drums=dr, e=e, chroma=[round(float(c), 3) for c in seg(chroma, t0, t1).mean(1)]))
    if len(bars) < 16: return None
    # activity per bar
    E = {k: np.array([b["e"][k] for b in bars]) for k in ("drums", "bass", "other")}
    top = max(np.percentile(E[k], 95) for k in E) + 1e-9
    def act(k, b, i):
        p95 = np.percentile(E[k], 95)
        return bool(p95 > .08 * top and E[k][i] > .2 * p95)
    for i, b in enumerate(bars):
        a = [act("drums", b, i) and sum(x > 0 for x in b["drums"]) >= 2, act("bass", b, i) and any(x >= 0 for x in b["bass"]), act("other", b, i), sum(x >= 0 for x in b["lead"]) >= 1 and act("other", b, i)]
        b["active"] = [int(x) for x in a]; b["energy"] = float(sum(b["e"][k] for k in ("drums", "bass", "other")))
    # sections: novelty on a 4-bar grid from activity + energies + chroma
    F_ = np.array([b["active"] + [b["e"]["drums"] / top, b["e"]["bass"] / top, b["e"]["other"] / top] + [c * .5 for c in b["chroma"]] for b in bars])
    n = len(bars); cand = list(range(4, n - 3, 4)); nov = {}
    for c in cand: nov[c] = float(np.linalg.norm(F_[max(0, c - 4):c].mean(0) - F_[c:c + 4].mean(0)))
    thr = np.percentile(list(nov.values()), 55) if nov else 9e9
    bnd = [0] + [c for c in cand if nov[c] >= thr and nov[c] >= max(nov.get(c - 4, 0), nov.get(c + 4, 0))] + [n]
    secs = []
    for s0, s1 in zip(bnd[:-1], bnd[1:]):
        ab = np.array([bars[i]["active"] for i in range(s0, s1)]); en = float(np.mean([bars[i]["energy"] for i in range(s0, s1)]))
        secs.append(dict(start=s0, len=s1 - s0, active=[int(x) for x in (ab.mean(0) >= .5)], e=en))
    es = sorted(s["e"] for s in secs)
    for s in secs: s["energy"] = int(min(3, np.searchsorted(es, s["e"], side="right") * 4 // (len(es) + 1)))
    maxact = max(sum(s["active"]) for s in secs); hasdr = any(s["active"][0] for s in secs)
    for i, s in enumerate(secs):
        na = sum(s["active"])
        if i == 0 and (s["energy"] <= 1 or na < maxact): t = "intro"
        elif i == len(secs) - 1 and len(secs) > 2 and (s["energy"] <= 1 or na < maxact): t = "outro"
        elif hasdr and not s["active"][0]: t = "break"
        elif s["energy"] >= 3 and na == maxact: t = "hook"
        elif i + 1 < len(secs) and secs[i + 1]["energy"] > s["energy"] and secs[i + 1]["energy"] >= 2: t = "build"
        else: t = "groove"
        s["type"] = t
        pv = secs[i - 1] if i else None
        s["transition"] = "start" if pv is None else ("drop" if s["active"][0] and not pv["active"][0] else "up" if s["energy"] > pv["energy"] else "down" if s["energy"] < pv["energy"] else "same")
    roots = [b["chord"][0] for b in bars]
    mode = "minor" if roots.count(9) + .5 * roots.count(2) > roots.count(0) + .5 * roots.count(5) else "major"
    return dict(name=name, style=meta["style"], prompt_key=meta["key"], prompt_bpm=bpm, duration=meta["duration"], tempo=round(tempo, 1), grid=grid, shift=shift, mode=mode,
                segmentation="stem-energy+chroma novelty on a 4-bar grid (ACE LM gives no section timestamps)", tracks=TRACKS,
                vocals_rms=float(np.sqrt((st["vocals"] ** 2).mean())), other_rms=float(np.sqrt((st["other"] ** 2).mean())),
                sections=[{k: s[k] for k in ("type", "start", "len", "energy", "active", "transition")} for s in secs],
                bars=[{k: b[k] for k in ("chord", "lead", "bass", "drums", "active")} | {"energy": round(b["energy"], 5)} for b in bars])
def job(name):
    t0 = time.time()
    try:
        r = run(name)
        json.dump(r or dict(name=name, error="too short"), open(f"sym/{name}.json", "w"))
        return f"{name} {'ok' if r else 'short'} {len(r['bars']) if r else 0} bars {len(r['sections']) if r else 0} secs {time.time() - t0:.1f}s"
    except Exception as e:
        json.dump(dict(name=name, error=repr(e)[:300]), open(f"sym/{name}.json", "w")); return f"{name} ERR {e!r}"[:300]
if __name__ == "__main__":
    os.makedirs("sym", exist_ok=True); pool = Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
    while True:
        todo = [os.path.basename(d) for d in sorted(glob.glob("stems/*")) if os.path.exists(f"{d}/done") and not os.path.exists(f"sym/{os.path.basename(d)}.json")]
        if not todo:
            if os.path.exists("STEMS_DONE"): break
            time.sleep(5); continue
        for msg in pool.imap_unordered(job, todo): print(msg, flush=True)
    print("trans done", flush=True)
