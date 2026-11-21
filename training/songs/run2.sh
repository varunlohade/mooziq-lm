#!/bin/sh
# Job 3 wave B: the 6 new styles, then the 9-style planner + composer. Replaces run.sh's tail (run.sh is killed first).
mkdir -p "${WORK:-work}" && cd "${WORK:-work}"; C=code; DL=$1
for i in 0 1 2; do B=4 NB=60 nohup python3 $C/gen_new.py $i 3 $DL > genN$i.log 2>&1 & sleep 2; done
sleep 200; pkill -f 'gen_son[g]'
while pgrep -f 'gen_ne[w]' > /dev/null; do sleep 10; done; touch GEN_DONE
while pgrep -f 'stems_po[d]' > /dev/null; do sleep 5; done; touch STEMS_DONE
while pgrep -f 'song_tran[s]' > /dev/null; do sleep 5; done
python3 $C/build_songs.py > build.log 2>&1; date -u +%H:%M:%S > xfer/stage_data.done
DROP=.2 EVERY=50 PATIENCE=12 python3 $C/songmodels.py planner planner 64 2 4 4 > planner.out 2>&1 &
DROP=.3 EVERY=50 PATIENCE=12 python3 $C/songmodels.py planner planner_d3 96 3 4 4 > planner_d3.out 2>&1 &
DROP=.2 EVERY=100 python3 $C/songmodels.py songlm songlm 160 6 5 8 > songlm.out 2>&1 &
DROP=.15 EVERY=100 python3 $C/songmodels.py songlm songlm_s 96 4 4 7 > songlm_s.out 2>&1 &
DROP=.3 EVERY=100 python3 $C/songmodels.py songlm songlm_l 256 6 8 8 > songlm_l.out 2>&1 &
wait; date -u +%H:%M:%S > xfer/stage_models.done
