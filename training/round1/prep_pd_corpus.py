# Melody text format, one piece per line:
#   each note = pitch char + duration char, bars split by '|'
#   pitch: 'r' rest, else chr(ord('A') + (midi - 48)) for midi 48..95 (C3..B6), transposed to C major / A minor
#   duration in 16ths: 1 2 3 4 6 8 c(12) g(16)
import sys, json, multiprocessing as mp
from music21 import corpus, note, chord, interval, pitch
DUR = {0.25: "1", 0.5: "2", 0.75: "3", 1.0: "4", 1.5: "6", 2.0: "8", 3.0: "c", 4.0: "g"}
GROUPS = ["bach", "mozart", "beethoven", "haydn", "schumann_robert", "schumann_clara", "chopin", "handel", "schubert",
          "joplin", "corelli", "cpebach", "weber", "verdi", "oneills1850", "ryansMammoth", "essenFolksong", "airdsAirs", "nottingham-dataset"]
def enc(path):
    try:
        s = corpus.parse(path)
        works = s.scores if hasattr(s, "scores") and len(getattr(s, "scores", [])) > 1 else [s]
        out = []
        for w in works[:400]:
            part = w.parts[0] if w.parts else w
            k = w.analyze("key")
            tonic = "C" if k.mode == "major" else "A"
            iv = interval.Interval(k.tonic, pitch.Pitch(tonic))
            toks, bar = [], None
            for el in part.recurse().notesAndRests:
                ql = float(el.quarterLength); d = DUR.get(round(ql * 4) / 4)
                if d is None: continue
                m = el.measureNumber
                if bar is not None and m != bar: toks.append("|")
                bar = m
                if isinstance(el, note.Rest): toks.append("r" + d); continue
                p = el.pitches[-1] if isinstance(el, chord.Chord) else el.pitch
                midi = p.transpose(iv).midi
                while midi < 55: midi += 12
                while midi > 88: midi -= 12
                toks.append(chr(ord("A") + midi - 48) + d)
            if len(toks) > 16: out.append("".join(toks))
        return path.split("/corpus/")[-1].split("/")[0], out
    except Exception as e:
        return None, []
if __name__ == "__main__":
    paths = [str(p) for p in corpus.getPaths() if str(p).split("/corpus/")[-1].split("/")[0] in GROUPS]
    print("files", len(paths), flush=True)
    lines, by = [], {}
    with mp.Pool(10) as pool:
        for i, (g, out) in enumerate(pool.imap_unordered(enc, paths, chunksize=4)):
            if g: by[g] = by.get(g, 0) + len(out); lines += out
            if i % 200 == 0: print(i, len(lines), flush=True)
    open("melodies.txt", "w").write("\n".join(lines) + "\n")
    print(json.dumps(by), "pieces", len(lines), "chars", sum(map(len, lines)))
