#!/bin/sh
# Adds the second-tier instruments to public/samples/ (loaded after the first sound):
# vibraphone and shaker from VCSL (CC0) and nylon guitar from FreePats (CC0). See CREDITS.md.
# Usage: engine-tools/build-samples-2.sh <VCSL download dir> <SpanishClassicalGuitar samples dir>
set -e
cd "$(dirname "$0")/../site"
VCSL=$1 GUITAR=$2 OUT=public/samples
mkdir -p $OUT/vibes $OUT/guitar
enc() { ffmpeg -nostdin -v error -y "$@"; }
# VCSL names the vibraphone one octave low: its "A2" sounds at MIDI 57.
for n in F2:53 A2:57 C3:60 E3:64 G3:67 B3:71 D4:74 F4:77 A4:81 C5:84 E5:88; do
  f=$(ls "$VCSL"/Vibes_soft_${n%:*}_v1_*Main.wav | head -1)
  enc -i "$f" -t 4 -af "afade=t=out:st=3:d=1" -ar 44100 -ac 2 -c:a aac -b:a 80k "$OUT/vibes/${n#*:}.m4a"
done
# Nylon guitar, every 3 semitones across the lead range (FreePats names: C4 = MIDI 60).
for n in E3:52 G3:55 A#3:58 C#4:61 E4:64 G4:67 A#4:70 C#5:73 E5:76; do
  f="$GUITAR/${n%:*}.flac"
  [ -f "$f" ] || { echo "missing $f"; continue; }
  enc -i "$f" -t 3 -af "afade=t=out:st=2.2:d=0.8" -ar 44100 -ac 2 -c:a aac -b:a 80k "$OUT/guitar/${n#*:}.m4a"
done
k=1
for f in "$VCSL"/Mid_ShakerDouble_Down_rr1.wav "$VCSL"/Mid_ShakerDouble_Up_rr1.wav "$VCSL"/Mid_ShakerHighFaster_Down_rr1.wav "$VCSL"/Mid_ShakerHighFaster_Up_rr2.wav; do
  enc -i "$f" -ar 44100 -ac 2 -c:a aac -b:a 96k "$OUT/drums/shaker$k.m4a"; k=$((k+1))
done
list() { ls "$OUT/$1" | sed 's/\.m4a$//' | sort -n | sed 's/.*/"&"/' | paste -sd, -; }
printf '{\n"piano": [%s],\n"bass": [%s],\n"drums": [%s],\n"vibes": [%s],\n"guitar": [%s]\n}\n' "$(list piano)" "$(list bass)" "$(list drums)" "$(list vibes)" "$(list guitar)" > $OUT/index.json
du -sh $OUT $OUT/vibes $OUT/guitar
