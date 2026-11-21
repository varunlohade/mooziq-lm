#!/bin/sh
# Generation + stems + transcription, then models. Markers in xfer/.
mkdir -p "${WORK:-work}" && cd "${WORK:-work}"; C=code; DL=$1; NW=${2:-3}
python3 -c "import torch; from demucs.pretrained import get_model; get_model('htdemucs')" > demucs_dl.log 2>&1
for i in $(seq 0 $((NW-1))); do B=4 NB=75 nohup python3 $C/gen_song.py $i $NW $DL > gen$i.log 2>&1 & sleep 2; done
nohup python3 $C/stems_pod.py > stems.log 2>&1 &
bp/bin/python -c "import librosa, numpy as np; y=np.random.randn(22050*20).astype('float32'); librosa.beat.beat_track(y=y, sr=22050); librosa.onset.onset_detect(y=y, sr=22050)" > numba_warm.log 2>&1
nohup bp/bin/python $C/song_trans.py 10 > trans.log 2>&1 &
(while true; do python3 $C/build_songs.py > build.log 2>&1; sleep 90; done) &
while pgrep -f 'gen_son[g]' > /dev/null; do sleep 10; done; touch GEN_DONE
while pgrep -f 'stems_po[d]' > /dev/null; do sleep 5; done; touch STEMS_DONE
while pgrep -f 'song_tran[s]' > /dev/null; do sleep 5; done
python3 $C/build_songs.py > build.log 2>&1; date -u +%H:%M:%S > xfer/stage_data.done
DROP=.2 EVERY=50 PATIENCE=12 python3 $C/songmodels.py planner planner 64 2 4 6 > planner.out 2>&1 &
DROP=.3 EVERY=50 PATIENCE=12 python3 $C/songmodels.py planner planner_d3 96 3 4 6 > planner_d3.out 2>&1 &
DROP=.2 EVERY=100 python3 $C/songmodels.py songlm songlm 160 6 5 12 > songlm.out 2>&1 &
DROP=.15 EVERY=100 python3 $C/songmodels.py songlm songlm_s 96 4 4 10 > songlm_s.out 2>&1 &
DROP=.3 EVERY=100 python3 $C/songmodels.py songlm songlm_l 256 6 8 12 > songlm_l.out 2>&1 &
wait; date -u +%H:%M:%S > xfer/stage_models.done
