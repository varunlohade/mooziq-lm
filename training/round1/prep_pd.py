# Public-domain melodies (melodies.txt from prep_pd_corpus.py, music21 corpus) -> same bar format as the teacher data.
# Keeps 4/4 (and paired 2/4) tunes, puts them on an 8th grid, labels each bar with the diatonic 7th chord that fits best.
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "melodies.txt")  # written by prep_pd_corpus.py
DUR = {"1": 1, "2": 2, "3": 3, "4": 4, "6": 6, "8": 8, "c": 12, "g": 16}      # 16ths
DIA = [(0, 0), (2, 1), (4, 1), (5, 0), (7, 2), (9, 1), (11, 3)]               # Cmaj7 Dm7 Em7 Fmaj7 G7 Am7 Bm7b5
IV = {0: [0, 4, 7, 11], 1: [0, 3, 7, 10], 2: [0, 4, 7, 10], 3: [0, 3, 6, 10]}
def bars_of(line):
    out = []
    for b in line.split("|"):
        notes, i = [], 0
        while i + 1 < len(b):
            d = DUR.get(b[i + 1])
            if d: notes.append((None if b[i] == "r" else 48 + ord(b[i]) - 65, d))
            i += 2
        out.append(notes)
    return out
def to_grid(notes):                      # 16 sixteenths -> 8 eighth steps
    mel, pos = [-1] * 8, 0
    for p, d in notes:
        if pos % 2 == 0 and pos // 2 < 8:
            k = pos // 2
            if p is None: mel[k] = -1
            else:
                mel[k] = p
                for j in range(k + 1, min(8, (pos + d + 1) // 2)): mel[j] = -2
        pos += d
    return mel
def label(mel, prev):
    best = None
    for r, q in DIA:
        tones = {(r + i) % 12 for i in IV[q]}
        s = 0
        for k, p in enumerate(mel):
            if p >= 0: s += (2 if k % 4 == 0 else 1.2 if k % 2 == 0 else .6) * (1 if p % 12 in tones else -.6) + (.5 if p % 12 == r and k == 0 else 0)
        s -= .8 if (r, q) == prev else 0
        if best is None or s > best[0]: best = (s, r, q)
    return best[1], best[2]
if __name__ == "__main__":
    pieces = []
    for line in open(SRC):
        bs = bars_of(line.strip())
        lens = [sum(d for _, d in b) for b in bs]
        if len(bs) < 8: continue
        full = [l for l in lens[1:-1]]
        if not full: continue
        if sum(l == 16 for l in full) / len(full) > .9: body = [b for b, l in zip(bs, lens) if l == 16]
        elif sum(l == 8 for l in full) / len(full) > .9:
            b8 = [b for b, l in zip(bs, lens) if l == 8]; body = [b8[i] + b8[i + 1] for i in range(0, len(b8) - 1, 2)]
        else: continue
        grid = [to_grid(b) for b in body]
        pitched = [p for g in grid for p in g if p >= 0]
        if len(grid) < 6 or not pitched: continue
        mode = "minor" if pitched[-1] % 12 == 9 else "major"
        prev, bars = None, []
        for g in grid:
            r, q = label(g, prev); prev = (r, q); bars.append([r, q, g])
        pieces.append(dict(mode=mode, bars=bars))
    json.dump(pieces, open(os.path.join(HERE, "data/pd_bars.json"), "w"))
    print("pieces", len(pieces), "bars", sum(len(p["bars"]) for p in pieces))
