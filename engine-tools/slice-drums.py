# Slices one-shot drum hits out of WaivOps Lo-Fi Drums loops by onset detection and spectral class.
# Usage: uv run --with numpy python -I engine-tools/slice-drums.py <loops dir> <labels dir> <out dir>
# The loops were streamed from the dataset archive (18 of 10,000); labels come from its JSON archive.
import sys, json, glob, os, wave, numpy as np
src, labels_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
labels = {}
for f in glob.glob(labels_dir + '/*bpm*.json'):
    d = json.load(open(f)); k = list(d)[0]; labels[k] = set(d[k]['pitch'])
def read(path):
    w = wave.open(path); n = w.getnframes(); ch = w.getnchannels(); sw = w.getsampwidth(); sr = w.getframerate()
    x = np.frombuffer(w.readframes(n), dtype='<i2' if sw == 2 else '<i4').reshape(-1, ch).astype(np.float32)
    return x / (32768 if sw == 2 else 2**31), sr
cands = []
for path in sorted(glob.glob(src + '/*.wav')):
    key = os.path.basename(path)[:-4]; x, sr = read(path); m = x.mean(1)
    hop = 256; frames = len(m) // hop; win = np.hanning(1024)
    spec = np.array([np.abs(np.fft.rfft(np.pad(m[i*hop:i*hop+1024], (0, max(0, 1024 - len(m[i*hop:i*hop+1024])))) * win)) for i in range(frames)])
    f = np.fft.rfftfreq(1024, 1 / sr)
    flux = np.maximum(0, np.diff(np.log1p(spec * 50), axis=0)).sum(1)
    thr = np.median(flux) + 2.5 * flux.std()
    on = [i + 1 for i in range(1, len(flux) - 1) if flux[i] > thr and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]]
    onsets = []
    for o in on:
        if not onsets or o - onsets[-1] > 8: onsets.append(o)
    for j, o in enumerate(onsets):
        start = o * hop - 128; nxt = onsets[j + 1] * hop - 128 if j + 1 < len(onsets) else len(m)
        gap = (nxt - start) / sr
        if gap < .18: continue
        seg = spec[o:o + 6].mean(0)
        tot = seg.sum() + 1e-9
        cands.append(dict(key=key, pitches=sorted(labels.get(key, [])), start=int(max(0, start)), end=int(min(nxt, start + int(.9 * sr))), gap=gap,
                          low=seg[f < 150].sum() / tot, high=seg[f >= 5000].sum() / tot, cent=(seg * f).sum() / tot))
def pick(kind, cond, score, n):
    for i, d in enumerate(sorted([d for d in cands if cond(d)], key=score, reverse=True)[:n]):
        x, sr = read(os.path.join(src, d['key'] + '.wav'))
        seg = x[d['start']:d['end']].copy(); fade = min(len(seg) // 3, int(.04 * sr))
        seg[-fade:] *= np.linspace(1, 0, fade)[:, None]; seg[:32] *= np.linspace(0, 1, 32)[:, None]
        seg *= .7 / (np.abs(seg).max() + 1e-9)
        with wave.open(f'{out}/{kind}{i + 1}.wav', 'wb') as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes((seg * 32767).astype('<i2').tobytes())
        print(kind, i + 1, d['key'], d['pitches'])
alone = lambda d: set(d['pitches']) <= {35, 36, 37, 38, 39}
pick('kick', lambda d: alone(d) and d['low'] > .45, lambda d: d['low'] + .3 * min(d['gap'], .6), 3)
pick('snare', lambda d: alone(d) and 38 in d['pitches'] and d['low'] < .15 and 900 < d['cent'] < 4000, lambda d: min(d['gap'], .6) - d['low'], 2)
pick('rim', lambda d: alone(d) and 37 in d['pitches'] and d['low'] < .15 and d['cent'] > 900, lambda d: min(d['gap'], .5) - d['low'], 2)
pick('hat', lambda d: not alone(d) and d['low'] < .06 and d['high'] > .35, lambda d: d['high'] - d['low'], 4)
