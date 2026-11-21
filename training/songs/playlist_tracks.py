"""Locate persistent sound changes in local continuous mixes; boundaries are estimates."""
import subprocess

import numpy as np


def track_ranges(path, duration):
    # Four-second feature blocks. Compare the half-minute on either side so a
    # fill or chord change alone is unlikely to be called a track boundary.
    rate, nfft, seconds = 8000, 2048, 4
    freq = np.fft.rfftfreq(nfft, 1 / rate)
    pitch = np.rint(69 + 12 * np.log2(np.maximum(freq, 1) / 440)).astype(int) % 12
    edges = np.geomspace(45, 3950, 17)
    bands = [(freq >= lo) & (freq < hi) for lo,hi in zip(edges,edges[1:])]
    chroma = [(pitch == k) & (freq >= 100) & (freq < 1800) for k in range(12)]
    window = np.hanning(nfft)
    rows = []
    with subprocess.Popen(['ffmpeg','-v','error','-i',str(path),'-ac','1','-ar',str(rate),
                           '-f','f32le','-'], stdout=subprocess.PIPE) as process:
        while data := process.stdout.read(rate * seconds * 4):
            x = np.frombuffer(data, np.float32)
            if len(x) < nfft:
                break
            frames = np.lib.stride_tricks.sliding_window_view(x,nfft)[::512]
            mag = np.abs(np.fft.rfft(frames * window))
            timbre = np.log(np.maximum([mag[:,m].mean() for m in bands],1e-5))
            harmony = np.array([mag[:,m].mean() for m in chroma])
            harmony /= max(harmony.sum(),1e-8)
            rows.append(np.r_[timbre-timbre.mean(), harmony, timbre.mean()])
        if process.wait():
            raise RuntimeError(f'Could not decode {path}')
    features = np.array(rows)
    features = (features-features.mean(0)) / np.maximum(features.std(0),.0001)
    radius = 7
    novelty = np.zeros(len(rows))
    for i in range(radius,len(rows)-radius):
        a, b = features[i-radius:i], features[i:i+radius]
        novelty[i] = np.linalg.norm(a.mean(0)-b.mean(0))
    # A conservative peak threshold, with at least a minute between changes.
    centre = np.median(novelty)
    threshold = centre + 2.5*np.median(np.abs(novelty-centre))
    cuts = [0, duration]
    for i in np.argsort(novelty)[::-1]:
        if novelty[i] < threshold:
            break
        t = int(i)*seconds
        if min(abs(t-v) for v in cuts) >= 60:
            cuts.append(t)
    cuts.sort()
    return list(zip(cuts,cuts[1:]))
