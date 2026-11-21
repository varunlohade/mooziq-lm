# Sample a pool of 16-bar sequences from the teacher (slot-masked, temperature 1) for distillation.
# Usage: sample_pool.py TEACHER N [MAX_MINUTES]  -- stops early at the time limit, keeps what it has.
import sys, os, json, random, time, torch
sys.path.insert(0, "code"); from lm import *
TEACH, N = sys.argv[1], int(sys.argv[2]); MAXMIN = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9; O = "xfer/ckpt"
te = load_models(TEACH); smp = sample; TAGN = os.environ.get("POOL", TEACH)
r = random.Random(11); t0 = time.time(); out = []
while sum(len(x) for x in out) < N and time.time() - t0 < MAXMIN * 60:
    sts = r.choices(["jazzhop", "steady", "flow"], [.55, .25, .2], k=512); mos = [r.choice(["major", "minor"]) for _ in range(512)]
    out.append(smp(te, sts, mos, 1.0, bs=512)); print("pool", sum(len(x) for x in out), round(time.time() - t0), "s", flush=True)
x = torch.cat(out); torch.save(x, f"syn_{TAGN}.pt"); print("pool", tuple(x.shape), round(time.time() - t0), "s")
