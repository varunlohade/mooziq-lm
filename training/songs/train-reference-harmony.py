"""Fit a transposition-invariant chord movement and phrase recurrence model.
Usage: python3 tools/train-reference-harmony.py OUTPUT.json MIX1.mp3 MIX2.mp3 --report TRACKS.json
Use --browser-module public/engine/reference-harmony.js to export the fitted weights for playback.
Requires NumPy, ffmpeg and ffprobe. Each input is a distinct reference mix.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from playlist_tracks import track_ranges

RATE, HOP, FFT = 11025, 256, 4096
LAGS = [1, 2, 4]  # two-bar decisions: 2, 4 and 8 bars back


def roots(path, start, duration):
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(path),
                                  "-t", str(duration), "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"])
    audio = np.frombuffer(raw, np.float32)
    if len(audio) < RATE * 32:
        return [], None
    frames = np.lib.stride_tricks.sliding_window_view(audio, FFT)[::HOP]
    mag = np.abs(np.fft.rfft(frames * np.hanning(FFT))).astype(np.float32)
    # Estimate the pulse from onset autocorrelation in a relaxed lo-fi tempo range.
    onset = np.maximum(np.diff(np.log1p(mag[:, 20:1100]), axis=0), 0).mean(axis=1)
    onset -= onset.mean()
    lo, hi = [round(60 * RATE / (bpm * HOP)) for bpm in (110, 55)]
    corr = np.correlate(onset, onset, 'full')[len(onset)-1:]
    lag = lo + np.argmax(corr[lo:hi+1])
    if corr[lag] < .08 * corr[0] or corr[0] < 1e-8:
        return [], None
    # Refine the period to sub-frame precision to limit phase drift.
    left, mid, right = corr[lag-1:lag+2]
    period = float(lag)
    if mid > left and mid > right:
        period += .5 * (left-right) / (left-2*mid+right)
    freq = np.fft.rfftfreq(FFT, 1/RATE)
    pc = np.rint(69 + 12*np.log2(np.maximum(1, freq)/440)).astype(int) % 12
    def chroma(low, high):
        # Local spectral peaks suppress energy bleeding into adjacent notes.
        peaks = mag[:, 1:-1] * ((mag[:, 1:-1] >= mag[:, :-2]) & (mag[:, 1:-1] > mag[:, 2:]))
        f = freq[1:-1]
        return np.stack([peaks[:, (pc[1:-1] == k) & (f >= low) & (f < high)].sum(1)
                         for k in range(12)], axis=1)
    mid, bass = chroma(100, 1800), chroma(45, 220)
    # Pick the bar phase with the strongest harmonic boundaries. These are
    # estimated roots from a mix, so store only aggregate movement statistics.
    def bars(phase):
        bounds = np.round(np.arange(phase, len(mag), 4*period)).astype(int)
        def profile(a):
            x = np.array([a[i:j].mean(0) for i,j in zip(bounds,bounds[1:])])
            return x / np.maximum(1e-8, x.sum(axis=1, keepdims=True))
        return profile(mid), profile(bass)
    profiles = [bars(p * period / 2) for p in range(8)]
    c, b = max(profiles, key=lambda cb: np.abs(np.diff(cb[0],axis=0)).sum(1).mean())
    score = c + .55*b + .7*np.roll(c,-7,axis=1) + .65*np.maximum(np.roll(c,-3,axis=1),np.roll(c,-4,axis=1))
    # One harmonic decision covers two bars in the browser composer as well.
    n = len(score)//2*2
    return score[:n].reshape(-1,2,12).mean(1).argmax(1).tolist(), 60*RATE/(HOP*period)


def examples(sequences):
    return np.asarray([[seq[i], *(seq[i-lag] for lag in LAGS)]
                       for seq in sequences for i in range(max(LAGS),len(seq))])


def likelihood(rows, moves):
    target, prev = rows[:,0], rows[:,1:]
    return np.column_stack([moves[(target-prev[:,0])%12], target[:,None] == prev])


def fit(train, weights):
    # Learn a mixture of interval motion and literal returns to earlier roots.
    mix = np.full(4, .25)
    moves = np.full(12, 1/12)
    intervals = (train[:,0]-train[:,1])%12
    for _ in range(100):
        responsibility = likelihood(train,moves)*mix
        responsibility /= responsibility.sum(1,keepdims=True)
        responsibility *= weights[:,None]
        mix = responsibility.sum(0) + 1
        mix /= mix.sum()
        moves = np.bincount(intervals,weights=responsibility[:,0],minlength=12) + 1
        moves /= moves.sum()
    return moves, mix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('references',nargs='+',type=Path)
    parser.add_argument('--browser-module', type=Path, help='Also export a static browser module')
    parser.add_argument('--ranges', type=Path, help='Optional previously measured boundaries by filename')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    cached = json.loads(args.ranges.read_text()) if args.ranges else {}
    groups, report = [], []
    for path in args.references:
        duration = float(subprocess.check_output(['ffprobe','-v','error','-show_entries',
                         'format=duration','-of','csv=p=0',str(path)]))
        ranges = cached.get(path.name) or track_ranges(path,duration)
        tracks = []
        for number,(start,end) in enumerate(ranges,1):
            # Analyse the track in bounded chunks. No observation or lagged
            # training example straddles an estimated transition between tracks.
            count = max(1,int(np.ceil((end-start)/96)))
            length = (end-start)/count
            sequences, tempi = [], []
            for i in range(count):
                seq, bpm = roots(path,start+i*length,length)
                if len(seq) > max(LAGS):
                    sequences.append(seq)
                    tempi.append(bpm)
            data = examples(sequences)
            row = dict(playlist=path.name, track=number, start_s=round(start,2), end_s=round(end,2),
                       timing="estimated", chunks=count, analysed_chunks=len(sequences), decisions=len(data))
            if len(data):
                tracks.append(data)
                row.update(tempo_bpm=round(float(np.median(tempi)),1),
                           root_return=dict(zip(['2_bars','4_bars','8_bars'],
                                            (data[:,0,None]==data[:,1:]).mean(0).round(3).tolist())))
            report.append(row)
        if not tracks:
            parser.error('No usable tracks in '+path.name)
        groups.append(tracks)
        print(path.name,len(ranges),'estimated tracks;',len(tracks),'usable',flush=True)

    def pool(mixes):
        rows, weights = [], []
        for tracks in mixes:
            for track in tracks:
                rows.append(track)
                weights.extend([1/(len(tracks)*len(track))]*len(track))
        weights = np.asarray(weights)
        return np.concatenate(rows), weights/weights.mean()

    if len(groups)<2:
        parser.error('Need at least two playlists for leave-one-playlist-out validation')
    scores = []
    for i,path in enumerate(args.references):
        train,w = pool(groups[:i]+groups[i+1:])
        valid,vw = pool([groups[i]])
        moves,mix = fit(train,w)
        loss = np.average(-np.log(likelihood(valid,moves)@mix),weights=vw)
        counts = np.bincount((train[:,0]-train[:,1])%12,weights=w,minlength=12)+1
        baseline = np.average(-np.log((counts/counts.sum())[(valid[:,0]-valid[:,1])%12]),weights=vw)
        scores.append(dict(held_out=path.name,loss=float(loss),interval_only=float(baseline)))
        print(path.name,'held out:',loss,'interval-only:',baseline,flush=True)
    args.report.write_text(json.dumps(dict(tracks=report, validation=scores),indent=2))
    if np.mean([s['loss']-s['interval_only'] for s in scores]) >= 0:
        raise ValueError('Phrase model did not improve held-out playlist prediction')
    observations, weights = pool(groups)
    moves, mix = fit(observations,weights)
    model = {'lags': LAGS, 'mixture': mix.round(6).tolist(), 'intervals': moves.round(6).tolist()}
    payload = json.dumps(model, indent=2)
    args.output.write_text(payload+'\n')
    if args.browser_module:
        args.browser_module.write_text('// Generated by tools/train-reference-harmony.py.\nexport default '+payload+';\n')
