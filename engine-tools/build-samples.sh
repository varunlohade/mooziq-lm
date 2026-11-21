#!/bin/sh
# Builds public/samples/ from the original sample sets (see CREDITS.md for sources).
# Usage: engine-tools/build-samples.sh <UprightPianoKW samples dir> <FingerBassYR finger dir> <sliced drums dir>
# The drum one-shots come from engine-tools/slice-drums.py (WaivOps Lo-Fi Drums loops), run first.
set -e
cd "$(dirname "$0")/../site"
PIANO=$1 BASS=$2 DRUMS=$3 OUT=public/samples
rm -rf $OUT && mkdir -p $OUT/piano $OUT/bass $OUT/drums
enc() { ffmpeg -nostdin -v error -y "$@"; }
# Piano: notes every 3 semitones from A2 to C6, two velocity layers, 7 s with a 1.5 s fade.
for n in A2:45 C3:48 D#3:51 F#3:54 A3:57 C4:60 D#4:63 F#4:66 A4:69 C5:72 D#5:75 F#5:78 A5:81 C6:84; do
  name=${n%:*} midi=${n#*:}
  for v in L H; do
    f="$PIANO/${name}v$v.flac"; [ -f "$f" ] || continue
    enc -i "$f" -t 7 -af "afade=t=out:st=5.5:d=1.5" -ar 44100 -ac 2 -c:a aac -b:a 96k "$OUT/piano/$midi$v.m4a"
  done
done
# Bass: every 3 semitones from E1 to D#2, mono.
for n in E:28 G:31 A#:34 C#:37 D#:39; do
  enc -i "$BASS/${n%:*}.flac" -t 3 -af "afade=t=out:st=2.4:d=0.6" -ar 44100 -ac 1 -c:a aac -b:a 64k "$OUT/bass/${n#*:}.m4a"
done
for f in "$DRUMS"/*.wav; do
  enc -i "$f" -ar 44100 -ac 2 -c:a aac -b:a 96k "$OUT/drums/$(basename "$f" .wav).m4a"
done
list() { ls "$OUT/$1" | sed 's/\.m4a$//' | sort -n | sed 's/.*/"&"/' | paste -sd, -; }
printf '{\n"piano": [%s],\n"bass": [%s],\n"drums": [%s]\n}\n' "$(list piano)" "$(list bass)" "$(list drums)" > $OUT/index.json
du -sh $OUT
