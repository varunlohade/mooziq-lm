# Distil the teacher: ACE-Step audio -> notes (basic-pitch, Apache-2.0) -> per-bar chord + 8th-note melody, in C major / A minor.
# Output: data/ace_bars.json  [{name, mode, bpm, bars: [[chord_root_pc, quality, [8 melody tokens]], ...]}]
# melody token: -1 rest, -2 hold, else MIDI pitch (already transposed to C/Am)
import os, sys, json, glob, numpy as np, librosa
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
HERE = os.path.dirname(os.path.abspath(__file__))
PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
QUAL = {"maj7": [0, 4, 7, 11], "m7": [0, 3, 7, 10], "7": [0, 4, 7, 10], "m7b5": [0, 3, 6, 10], "sus": [0, 5, 7, 10], "maj": [0, 4, 7], "m": [0, 3, 7]}
QMAP = {"maj7": 0, "m7": 1, "7": 2, "m7b5": 3, "sus": 4, "maj": 0, "m": 1}     # triads fold into 7th qualities (lo-fi voicer colours them)
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
def meta_for(wav):
    js = wav[:-4] + ".json"
    if os.path.exists(js): j = json.load(open(js)); return j["key"], j["bpm"]
    return None
def run(wav, key, bpm):
    tonic, mode = key.split()[0], key.split()[1]
    shift = (0 - PC[tonic]) % 12 if mode == "major" else (9 - PC[tonic]) % 12      # -> C major or A minor
    if shift > 6: shift -= 12
    y, sr = librosa.load(wav, sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm=bpm, tightness=400, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    if abs(2 * tempo - bpm) <= 6:                                    # tracker locked to half time: add the off-beats
        beats = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2])); tempo *= 2
    if abs(tempo - bpm) > 6 and abs(tempo - 2 * bpm) > 6: return None, f"tempo {tempo:.1f} vs {bpm}"
    if abs(tempo - 2 * bpm) <= 6: beats = beats[::2]
    cache = os.path.join(HERE, "data/notes_" + os.path.basename(wav)[:-4] + ".json")
    if os.path.exists(cache): notes = json.load(open(cache))
    else:
        _, _, notes = predict(wav, ICASSP_2022_MODEL_PATH, onset_threshold=.5, frame_threshold=.3, minimum_note_length=80)
        notes = [(float(s), float(e), int(p), float(a), 0) for s, e, p, a, _ in notes]; json.dump(notes, open(cache, "w"))
    w = np.zeros(12)
    for s_, e_, p_, a_, _ in notes: w[p_ % 12] += (e_ - s_) * a_
    DIA = np.zeros(12); DIA[[0, 2, 4, 5, 7, 9, 11]] = 1
    shift = max(range(12), key=lambda k: (np.roll(w, k) @ DIA, -min(k, 12 - k)))
    if shift > 6: shift -= 12
    notes = [(s, e, p + shift, a) for s, e, p, a, _ in notes]
    # downbeat phase: where low notes (bass/kick-pitched) and new chords start
    low = np.array([s for s, e, p, a in notes if p < 52])
    def score_phase(ph):
        db = beats[ph::4]; return sum(np.min(np.abs(low - t)) < .07 for t in db) if len(low) else 0
    ph = max(range(4), key=score_phase)
    db = beats[ph::4]
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
        # melody: skyline of notes >= C4 whose onset falls in this bar, quantised to 8ths
        mel = [-1] * 8
        cand = [(s, e, p, a) for s, e, p, a in notes if p >= 62 and t0 - step / 2 <= s < t1 - step / 2 and a > .38 and e - s > .6 * step]
        for k in range(8):
            ins = [c for c in cand if abs(c[0] - (t0 + k * step)) < step / 2]
            if ins:
                s, e, p, a = max(ins, key=lambda c: (c[2], c[3]))
                while p > 84: p -= 12
                while p < 60: p += 12
                n = max(1, int(round((e - s) / step)))
                mel[k] = int(p)
                for j in range(k + 1, min(8, k + n)):
                    if mel[j] == -1: mel[j] = -2
        bars.append([int(r), QMAP[q], mel])
    roots = [b_[0] for b_ in bars]
    mode = "minor" if roots.count(9) + .5 * roots.count(2) > roots.count(0) + .5 * roots.count(5) else "major"
    return dict(mode=mode, shift=shift, bpm=bpm, tempo=round(tempo, 1), bars=bars), "ok"
if __name__ == "__main__":
    items = []
    for w in sorted(glob.glob(os.path.join(HERE, "data/ace/*.wav"))):
        m = meta_for(w)
        if m: items.append((w, *m))
    # earlier ACE-Step 1.5 renders on this Mac (open model, known prompts/keys), also teacher data
    ACE = os.environ.get("ACE_EXTRA_DIR", os.path.join(HERE, "data/ace_extra")) + "/"
    for f, k, b in [("focus-A-minimal-loop.wav", "F major", 70), ("focus-B-piano-ambient.wav", "D minor", 66),
                    ("focus-C-guitar-jazzhop.wav", "G major", 72), ("focus-D-minimal-noplanner.wav", "F major", 70)]:
        if os.path.exists(ACE + f): items.append((ACE + f, k, b))
    out = []
    for w, k, b in items:
        r, msg = run(w, k, b)
        name = os.path.basename(w)
        if r: r["name"] = name; out.append(r)
        print(name, k, b, msg, len(r["bars"]) if r else 0, flush=True)
        if r:
            NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]; Q = ["maj7", "m7", "7", "m7b5", "sus"]
            print("   ", " ".join(NAMES[c] + Q[q] for c, q, _ in r["bars"][:16]))
            print("   ", " | ".join("".join("." if t == -1 else "-" if t == -2 else "x" for t in m) for _, _, m in r["bars"][4:8]))
    json.dump(out, open(os.path.join(HERE, "data/ace_bars.json"), "w"))
    print("clips", len(out), "bars", sum(len(o["bars"]) for o in out))
