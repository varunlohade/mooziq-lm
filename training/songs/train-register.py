"""Train an isolated octave-drop detector on transcribed song JSON files.
Usage: python3 tools/train-register.py OUTPUT.js CORPUS.json [CORPUS.json ...]
Each observed three-onset phrase is paired with the same phrase with its
middle onset lowered one octave. Hold out complete songs before making pairs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

MAX_DEPTH = 24
MAX_GAP = 8  # eighth-note slots


def depth(left, middle, right):
    return min(MAX_DEPTH, max(0, min(left, right) - middle))


def counts(songs, part):
    clean, lowered = [[0] * (MAX_DEPTH + 1) for _ in range(2)]
    for song in songs:
        if song["style"] == "bossa" or (part == "bass" and song["style"] == "study_piano"):
            continue
        line = [(i, p) for i, p in enumerate(p for b in song["bars"] for p in b[part]) if p >= 0]
        for (a, left), (b, middle), (c, right) in zip(line, line[1:], line[2:]):
            if max(b-a, c-b) <= MAX_GAP:
                clean[depth(left, middle, right)] += 1
                lowered[depth(left, middle-12, right)] += 1
    return clean, lowered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("corpus", nargs="+", type=Path)
    args = parser.parse_args()
    songs = {s["name"]: s for f in args.corpus for s in json.loads(f.read_text()) if s["sections"] and s["style"] != "bossa"}
    train, test = [], []
    for name, song in songs.items():
        bucket = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 5
        (test if bucket == 0 else train).append(song)
    print(f"{len(train)} training songs, {len(test)} held-out songs")
    weights = {}
    for part in ("lead", "bass"):
        clean, lowered = counts(train, part)
        weights[part] = [round(math.log((b+1)/(a+1)), 4) for a,b in zip(clean, lowered)]
        clean, lowered = counts(test, part)
        # Equal training class sizes. Require 12:1 evidence of corruption.
        flagged = [d for d,w in enumerate(weights[part]) if d > 6 and w >= math.log(12)]
        for label, hist in [("original flagged", clean), ("injected detected", lowered)]:
            hits = sum(hist[d] for d in flagged)
            print(f"{part} {label}: {hits}/{sum(hist)} ({hits/sum(hist):.2%})")
    args.output.write_text("// Log likelihood ratios fitted by tools/train-register.py. Index by dip in semitones.\n"
                           + "export default " + json.dumps(weights, indent=None) + ";\n")


if __name__ == "__main__":
    main()
