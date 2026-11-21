"""Normalize the full-song transcription corpus to the station's instrument policy."""
import argparse
import json
from pathlib import Path


def repair(song):
    if song["style"] not in {"flow", "sleep", "study_piano", "nature"}:
        return
    solo = song["style"] == "study_piano"
    for bar in song["bars"]:
        bar["drums"] = [0] * len(bar["drums"])
        bar["active"][0] = 0
        if solo:
            bar["bass"] = [-1] * len(bar["bass"])
            bar["active"][1] = 0

    sections = song["sections"]
    for section in sections:
        section["active"][0] = 0
        if solo:
            section["active"][1] = 0
    # Reapply the transcriber's section rules with corrected activity. In
    # particular, "break" (drums leave) and "drop" (drums return) cannot occur.
    fullest = max(sum(section["active"]) for section in sections)
    for i, section in enumerate(sections):
        energy = section["energy"]
        layers = sum(section["active"])
        sparse = energy <= 1 or layers < fullest
        if i == 0 and sparse:
            kind = "intro"
        elif i == len(sections) - 1 and len(sections) > 2 and sparse:
            kind = "outro"
        elif energy >= 3 and layers == fullest:
            kind = "hook"
        elif i + 1 < len(sections) and sections[i+1]["energy"] >= max(2, energy+1):
            kind = "build"
        else:
            kind = "groove"
        section["type"] = kind
        previous = sections[i-1]["energy"]
        section["transition"] = ("start" if i == 0 else "up" if energy > previous
                                 else "down" if energy < previous else "same")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    songs = json.loads(args.source.read_text())
    for song in songs:
        if song["sections"]:
            repair(song)
    args.destination.write_text(json.dumps(songs, separators=(",", ":")))
