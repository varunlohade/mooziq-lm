#!/bin/sh
# Second pod: retrain the 5 ensemble members, ensemble-distil int8/int4 students, audit. Stage markers in xfer/.
mkdir -p "${WORK:-work}" && cd "${WORK:-work}"; C=code; E=B483K,T25,B2p5M,ND4p8M,ND8p7M
DROP=.3 LR=3e-4 EVERY=50 PATIENCE=12 python3 $C/trainer.py T25 512 8 8 14 > T25.out 2>&1 &
DROP=.25 EVERY=100 python3 $C/trainer.py B2p5M 160 8 5 12 > B2p5M.out 2>&1 &
DROP=.15 EVERY=100 python3 $C/trainer.py B483K 96 4 4 12 > B483K.out 2>&1 &
DROP=.25 EVERY=100 python3 $C/trainer.py ND4p8M 256 6 8 12 > ND4p8M.out 2>&1 &
DROP=.25 EVERY=100 python3 $C/trainer.py ND8p7M 320 7 10 12 > ND8p7M.out 2>&1 &
wait
python3 $C/ens_eval.py $E > xfer/ckpt/E5_eval.txt 2>&1
date -u +%H:%M:%S > xfer/stage_members.done
POOL=E5 python3 $C/sample_pool.py $E 32768 6 > xfer/ckpt/pool.log 2>&1
date -u +%H:%M:%S > xfer/stage_pool.done
POOL=E5 QAT_MIN=6 python3 $C/distill.py S8 256 6 8 8 0 14 $E > S8.out 2>&1 &
POOL=E5 QAT_MIN=6 python3 $C/distill.py S4 320 7 10 4 64 14 $E > S4.out 2>&1 &
wait
date -u +%H:%M:%S > xfer/stage_students.done
TAG=final python3 $C/audit.py 128 npz:B483K npz:B2p5M npz:ND4p8M npz:ND8p7M pt:T25 ens:$E q:S8 q:S4 > xfer/ckpt/audit.log 2>&1
date -u +%H:%M:%S > xfer/stage_audit.done
