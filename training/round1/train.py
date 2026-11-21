# Train the tiny chord+melody transformer: public-domain tunes (FOLK style token) + ACE-Step lo-fi transcriptions (LOFI style token).
import os, sys, json, math, time, random, numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import *
torch.manual_seed(0); random.seed(0)
MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 10
pd = json.load(open(os.path.join(HERE, "data/pd_bars.json")))
ace = json.load(open(os.path.join(HERE, "data/ace_bars.json")))
random.shuffle(ace); nval = max(2, len(ace) // 8)
ace_val, ace_tr = ace[:nval], ace[nval:]
pd_val, pd_tr = pd[:150], pd[150:]
print("pd", len(pd_tr), "ace", len(ace_tr), "ace bars", sum(len(a["bars"]) for a in ace_tr), flush=True)
def window(p, style):
    b = p["bars"]; s = random.randint(0, max(0, len(b) - BARS)); seq = encode(style, p["mode"], b[s:s + BARS])
    return seq + [PAD] * (T - len(seq))
MELW = .25          # teacher melodies are noisy transcriptions: learn lo-fi HARMONY from them at full weight, melody only lightly
def batch(B, pdset, aceset, frac=.5):
    rows, ws = [], []
    for _ in range(B):
        if random.random() < frac:
            r = window(random.choice(aceset), LOFI); rows.append(r)
            ws.append([MELW if t >= M0 else 1. for t in r[1:]])
        else: r = window(random.choice(pdset), FOLK); rows.append(r); ws.append([1.] * (len(r) - 1))
    x = torch.tensor(rows); return x[:, :-1], x[:, 1:], torch.tensor(ws)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
m = GPT().to(dev); nparam = sum(p.numel() for p in m.parameters()); print("params", nparam, "V", V, "T", T, dev, flush=True)
opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=.1)
def loss_of(x, y, w, only_chords=False):
    lg = m(x.to(dev)); y = y.to(dev); w = w.to(dev) * (y != PAD)
    if only_chords: w = w * ((y >= CH0) & (y < M0))
    l = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="none") * w.reshape(-1)
    return l.sum() / w.sum()
@torch.no_grad()
def evals():
    m.eval(); r = []
    for f in (0, 1):
        r.append(sum(loss_of(*batch(64, pd_val, ace_val, f), only_chords=bool(f)).item() for _ in range(8)) / 8)   # val_ace = chord NLL on held-out lo-fi
    m.train(); return r
t0, step, best = time.time(), 0, 9e9
budget = MINUTES * 60
while time.time() - t0 < budget:
    frac_t = (time.time() - t0) / budget
    lr = 2e-3 * min(1, step / 300) * (.5 * (1 + math.cos(math.pi * frac_t)) * .95 + .05)
    for g in opt.param_groups: g["lr"] = lr
    acefrac = .1 + .15 * frac_t                     # curriculum: grammar from PD tunes first, lo-fi style later
    loss = loss_of(*batch(96, pd_tr, ace_tr, acefrac))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1); opt.step(); step += 1
    if step % 400 == 0:
        vp, va = evals(); print(f"step {step} {time.time()-t0:.0f}s train {loss.item():.3f} val_pd {vp:.3f} val_ace {va:.3f}", flush=True)
        if vp + va < best:
            best = vp + va; np.savez(os.path.join(HERE, "model.npz"), **{k: v.detach().cpu().numpy().astype(np.float16) for k, v in m.state_dict().items()})
json.dump(dict(params=nparam, V=V, T=T, D=D, L=L, H=H, best_val_ace=best, steps=step, minutes=MINUTES), open(os.path.join(HERE, "model_meta.json"), "w"))
print("done best val_ace", best, "bytes", os.path.getsize(os.path.join(HERE, "model.npz")))
