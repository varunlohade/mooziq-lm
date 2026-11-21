#!/bin/sh
# Stage B-D on the final data. Each stage writes xfer/stage_*.done so the Mac can pull after every stage.
mkdir -p "${WORK:-work}" && cd "${WORK:-work}"; C=code
pkill -f 'build_dat[a]' ; pkill -f 'while true; do python3 code/build_dat[a]'
touch GEN_DONE
while pgrep -f 'trans_po[d]' > /dev/null; do sleep 5; done
python3 $C/build_data.py > xfer/data/final_stats.txt; cp xfer/data/ace2_bars.json xfer/data/final_ace_bars.json
date -u +%H:%M:%S > xfer/stage_data.done
# Stage B: teachers + non-distilled baselines (same data, same split)
DROP=.3 LR=2e-4 EVERY=50 PATIENCE=12 python3 $C/trainer.py T85 768 12 12 25 > T85.out 2>&1 &
DROP=.3 LR=3e-4 EVERY=50 PATIENCE=12 python3 $C/trainer.py T25 512 8 8 20 > T25.out 2>&1 &
DROP=.25 EVERY=100 python3 $C/trainer.py B2p5M 160 8 5 15 > B2p5M.out 2>&1 &
DROP=.15 EVERY=100 python3 $C/trainer.py B483K 96 4 4 15 > B483K.out 2>&1 &
DROP=.25 EVERY=100 python3 $C/trainer.py ND4p8M 256 6 8 15 > ND4p8M.out 2>&1 &
DROP=.25 EVERY=100 python3 $C/trainer.py ND8p7M 320 7 10 15 > ND8p7M.out 2>&1 &
wait
TEACH=$(python3 -c "import json; b={n: json.load(open(f'xfer/ckpt/{n}.json'))['best'] for n in ('T85','T25')}; print(min(b, key=lambda n: b[n]['ace'] + .25 * b[n]['pd']))")
echo $TEACH > xfer/teacher.txt; date -u +%H:%M:%S > xfer/stage_teacher.done
# Stage C: teacher pool + distilled students
python3 $C/sample_pool.py $TEACH 60000 10 > pool.out 2>&1
QAT_MIN=10 python3 $C/distill.py S8 256 6 8 8 0 20 $TEACH > S8.out 2>&1 &
QAT_MIN=10 python3 $C/distill.py S4 320 7 10 4 64 20 $TEACH > S4.out 2>&1 &
wait
date -u +%H:%M:%S > xfer/stage_students.done
# Stage D: held-out loss + music audit for everything
TAG=final python3 $C/audit.py 192 npz:B483K npz:B2p5M npz:ND4p8M npz:ND8p7M pt:$TEACH q:S8 q:S4 > audit.out 2>&1
date -u +%H:%M:%S > xfer/stage_audit.done
