# Plain (non-distilled) trainer on lm.py: used for the teacher and for the final-data baselines.
# Usage: trainer.py NAME D L H MAX_MINUTES  (env DROP NOISE LR WD MELW BS EVERY PATIENCE)
import sys, os, json, math, time, random, numpy as np, torch
sys.path.insert(0, "code"); from lm import *
NAME, D, L, H, MAXMIN = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
E = lambda k, d: float(os.environ.get(k, d))
DROP, NOISE, LR, WD, MELW, BS = E("DROP", .2), E("NOISE", 0), E("LR", 1e-3), E("WD", .1), E("MELW", .5), int(E("BS", 128))
EVERY, PATIENCE = int(E("EVERY", 200)), int(E("PATIENCE", 10))
SEED = int(E("SEED", 0)); torch.manual_seed(SEED); random.seed(SEED)
tr, val, pd_tr, pd_val = load_data(); EV = make_eval(val, pd_val); bat = Batcher(tr, pd_tr, MELW, NOISE)
O = "xfer/ckpt"; os.makedirs(O, exist_ok=True); log = open(f"{O}/{NAME}.log", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
m = GPT(D, L, H, DROP).cuda(); NP = nparams(m)
P("data train clips", len(tr), "val clips", len(val), "train bars", sum(len(a["bars"]) for a in tr), "val bars", sum(len(a["bars"]) for a in val), "pd", len(pd_tr), len(pd_val))
P("params", NP, "D L H", D, L, H, "drop", DROP, "noise", NOISE, "lr", LR, "wd", WD, "melw", MELW, "bs", BS)
dec = [p for n, p in m.named_parameters() if p.dim() == 2]; nodec = [p for n, p in m.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([dict(params=dec, weight_decay=WD), dict(params=nodec, weight_decay=0)], lr=LR, betas=(.9, .98))
t0, step, best, bad, hist = time.time(), 0, 9e9, 0, []
while time.time() - t0 < MAXMIN * 60:
    lr = LR * min(1, (step + 1) / 300) * max(.1, .5 * (1 + math.cos(math.pi * min(1, step / 15000))))
    for g in opt.param_groups: g["lr"] = lr
    x, y, w = bat(BS)
    with torch.autocast("cuda", dtype=torch.bfloat16): loss = ce(m(x.cuda()), y.cuda(), w.cuda())
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1); opt.step(); step += 1
    if step % EVERY == 0:
        r = evaluate(m, EV); r.update(step=step, train=round(loss.item(), 4), secs=round(time.time() - t0)); hist.append(r)
        P(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
        crit = r["ace"] + .25 * r["pd"]
        if crit < best - 1e-3:
            best = crit; bad = 0; torch.save(m.state_dict(), f"{O}/{NAME}.pt")
        else:
            bad += 1
            if bad >= PATIENCE: P("early stop"); break
bi = min(range(len(hist)), key=lambda i: hist[i]["ace"] + .25 * hist[i]["pd"])
sd = torch.load(f"{O}/{NAME}.pt", weights_only=True)
if NP < 20e6: np.savez(f"{O}/{NAME}.npz", **{k: v.cpu().numpy().astype(np.float16) for k, v in sd.items()})
json.dump(dict(name=NAME, params=NP, D=D, L=L, H=H, V=V, T=T, drop=DROP, noise=NOISE, lr=LR, wd=WD, melw=MELW, seed=SEED, steps=step, best=hist[bi], hist=hist, tokens=TOKENS),
          open(f"{O}/{NAME}.json", "w"), indent=1)
P("done", NAME, "best", json.dumps(hist[bi]))
