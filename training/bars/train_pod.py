# Style-conditioned chord + 8-melody-token transformer. Same layout/state-dict names as the shipped kernel.
# Usage: train_pod.py NAME D L H MAX_MINUTES   -> xfer/ckpt/NAME.{npz,log,json}
import os, sys, json, math, time, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
NAME, D, L, H, MAXMIN = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
DROP = float(os.environ.get("DROP", ".15")); MELW = float(os.environ.get("MELW", ".5")); WD = float(os.environ.get("WD", ".1")); NOISE = float(os.environ.get("NOISE", "0"))
LR, BS, WARM = float(os.environ.get("LR", "1e-3")), 128, 300
PAD, BOS, LOFI, FOLK, MAJ, MIN, BAR = range(7)
NQ = 5; CH0 = 7; M0 = CH0 + 12 * NQ; LO, HI = 57, 86; V0 = M0 + 2 + (HI - LO + 1)   # 99 = shipped vocab
JAZZHOP, FLOW = V0, V0 + 1; V = V0 + 2                                               # new style tokens appended; STEADY reuses LOFI=2
STYLE_TOK = {"jazzhop": JAZZHOP, "steady": LOFI, "flow": FLOW, "folk": FOLK}
BARS = 16; T = 3 + 10 * BARS
def m_tok(x):
    if x == -1: return M0
    if x == -2: return M0 + 1
    while x < LO: x += 12
    while x > HI: x -= 12
    return M0 + 2 + x - LO
def encode(style, mode, bars):
    s = [BOS, STYLE_TOK[style], MAJ if mode == "major" else MIN]
    for r, q, mel in bars: s += [BAR, CH0 + r * NQ + q] + [m_tok(x) for x in mel]
    return s
class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.ln1 = nn.LayerNorm(D); s.qkv = nn.Linear(D, 3 * D); s.proj = nn.Linear(D, D)
        s.ln2 = nn.LayerNorm(D); s.fc = nn.Linear(D, 4 * D); s.out = nn.Linear(4 * D, D)
    def forward(s, x):
        B, t, _ = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(D, 2)
        q, k, v = [z.view(B, t, H, D // H).transpose(1, 2) for z in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + F.dropout(s.proj(y.transpose(1, 2).reshape(B, t, D)), DROP, s.training)
        return x + F.dropout(s.out(F.gelu(s.fc(s.ln2(x)), approximate="tanh")), DROP, s.training)
class GPT(nn.Module):
    def __init__(s):
        super().__init__(); s.tok = nn.Embedding(V, D); s.pos = nn.Embedding(T, D); s.slot = nn.Embedding(10, D)
        s.blocks = nn.ModuleList(Block() for _ in range(L)); s.lnf = nn.LayerNorm(D); s.head = nn.Linear(D, V)
    def forward(s, idx):
        t = idx.shape[1]; ar = torch.arange(t, device=idx.device)
        x = F.dropout(s.tok(idx) + s.pos(ar) + s.slot((ar - 3).clamp(min=0) % 10), DROP, s.training)
        for b in s.blocks: x = b(x)
        return s.head(s.lnf(x))
torch.manual_seed(0); random.seed(0)
pd = json.load(open("data/pd_bars.json")); ace = json.load(open("data/ace2_bars.json"))   # ace items carry "style"
ace = [a for a in ace if len(a["bars"]) >= 8]
rs = random.Random(1); rs.shuffle(ace); rs.shuffle(pd)
# held-out: 10% of clips per style (whole clips), 150 PD pieces
val, tr = [], []
for st in ("jazzhop", "steady", "flow"):
    c = [a for a in ace if a["style"] == st]; n = max(2, len(c) // 10) if len(c) >= 4 else 0; val += c[:n]; tr += c[n:]
pd_val, pd_tr = pd[:150], pd[150:]
log = open(f"xfer/ckpt/{NAME}.log", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
P("train clips", {st: sum(a["style"] == st for a in tr) for st in STYLE_TOK}, "val clips", {st: sum(a["style"] == st for a in val) for st in STYLE_TOK},
  "train bars", sum(len(a["bars"]) for a in tr), "val bars", sum(len(a["bars"]) for a in val), "pd", len(pd_tr), len(pd_val))
def window(p, style, r=random):
    b = p["bars"]; s = r.randint(0, max(0, len(b) - BARS)); seq = encode(style, p["mode"], b[s:s + BARS])
    return seq + [PAD] * (T - len(seq))
def rows_w(rows, ace_flags):
    return [[(MELW if (t >= M0 and t < V0 and fa) else 1.) for t in r[1:]] for r, fa in zip(rows, ace_flags)]
def batch(B, frac):
    rows, fl = [], []
    for _ in range(B):
        if random.random() < frac: a = random.choice(tr); rows.append(window(a, a["style"])); fl.append(True)
        else: rows.append(window(random.choice(pd_tr), "folk")); fl.append(False)
    x = torch.tensor(rows); xi = x[:, :-1].clone()
    if NOISE > 0:   # input-token noise: swap some chord/melody inputs for random ones of the same kind (targets stay clean)
        r = torch.rand(xi.shape) < NOISE
        ch = r & (xi >= CH0) & (xi < M0); me = r & (xi >= M0) & (xi < V0)
        xi[ch] = torch.randint(CH0, M0, (int(ch.sum()),)); xi[me] = torch.randint(M0, V0, (int(me.sum()),))
    return xi, x[:, 1:], torch.tensor(rows_w(rows, fl))
# fixed held-out eval sets (deterministic windows)
er = random.Random(5)
def fixed(items, styl, n):
    rows = [window(it, styl(it), er) for it in (items * (n // max(1, len(items)) + 1))[:n]]
    x = torch.tensor(rows); return x[:, :-1], x[:, 1:]
EV = {"ace": fixed(val, lambda a: a["style"], 512), "pd": fixed(pd_val, lambda a: "folk", 256)}
for st in ("jazzhop", "steady", "flow"):
    if any(a["style"] == st for a in val): EV[st] = fixed([a for a in val if a["style"] == st], lambda a: a["style"], 192)
dev = "cuda"
m = GPT().to(dev); nparam = sum(p.numel() for p in m.parameters()); P("params", nparam, "D L H", D, L, H, "V", V, "T", T, "drop", DROP, "melw", MELW, "noise", NOISE, "wd", WD, "lr", LR)
opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD, betas=(.9, .98))
def loss_of(x, y, w, mask=None):
    lg = m(x.to(dev)); y = y.to(dev); w = w.to(dev) * (y != PAD)
    if mask is not None: w = w * mask(y)
    l = F.cross_entropy(lg.reshape(-1, V).float(), y.reshape(-1), reduction="none") * w.reshape(-1)
    return l.sum() / w.sum()
chord_mask = lambda y: ((y >= CH0) & (y < M0)).float()
mel_mask = lambda y: ((y >= M0) & (y < V0)).float()
@torch.no_grad()
def evals():
    m.eval(); r = {}
    for k, (x, y) in EV.items():
        ones = torch.ones_like(y, dtype=torch.float)
        r[k] = sum(loss_of(x[i:i + 128], y[i:i + 128], ones[i:i + 128]).item() * len(x[i:i + 128]) for i in range(0, len(x), 128)) / len(x)
    x, y = EV["ace"]; ones = torch.ones_like(y, dtype=torch.float)
    r["ace_chord"] = loss_of(x, y, ones, chord_mask).item(); r["ace_mel"] = loss_of(x, y, ones, mel_mask).item()
    m.train(); return r
t0, step, best, bad, hist = time.time(), 0, 9e9, 0, []
PATIENCE = 10; EVERY = 200
while time.time() - t0 < MAXMIN * 60:
    lr = LR * min(1, (step + 1) / WARM) * max(.1, .5 * (1 + math.cos(math.pi * min(1, step / 12000))))
    for g in opt.param_groups: g["lr"] = lr
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = loss_of(*batch(BS, .75))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1); opt.step(); step += 1
    if step % EVERY == 0:
        r = evals(); r.update(step=step, train=round(loss.item(), 4), secs=round(time.time() - t0)); hist.append(r)
        P(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
        crit = r["ace"] + .25 * r["pd"]
        if crit < best - 1e-3:
            best = crit; bad = 0
            np.savez(f"xfer/ckpt/{NAME}.npz", **{k: v.detach().cpu().numpy().astype(np.float16) for k, v in m.state_dict().items()})
            P("  saved best", round(best, 4))
        else:
            bad += 1
            if bad >= PATIENCE: P("early stop"); break
bi = min(range(len(hist)), key=lambda i: hist[i]["ace"] + .25 * hist[i]["pd"])
json.dump(dict(name=NAME, params=nparam, D=D, L=L, H=H, V=V, T=T, drop=DROP, melw=MELW, noise=NOISE, wd=WD, lr=LR, steps=step, best=hist[bi], hist=hist,
               tokens=dict(PAD=PAD, BOS=BOS, LOFI=LOFI, STEADY=LOFI, FOLK=FOLK, JAZZHOP=JAZZHOP, FLOW=FLOW, MAJ=MAJ, MIN=MIN, BAR=BAR, CH0=CH0, M0=M0, NQ=NQ, LO=LO, HI=HI)),
          open(f"xfer/ckpt/{NAME}.json", "w"), indent=1)
P("done", NAME, "best", json.dumps(hist[bi]), "npz bytes", os.path.getsize(f"xfer/ckpt/{NAME}.npz"))
