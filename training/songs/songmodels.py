# Planner LM (song -> section tokens) and Song LM (bar-level chord + lead + bass, conditioned on section and active-track tokens).
# Usage: songmodels.py planner NAME D L H MINUTES | songmodels.py songlm NAME D L H MINUTES
import os, sys, json, math, time, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
STY = ["jazzhop", "steady", "flow", "rainy_cafe", "sleep", "bossa", "study_piano", "chillhop", "nature"]; TYPES = ["intro", "groove", "build", "hook", "break", "outro"]; TRANS = ["start", "up", "down", "same", "drop"]
LENS = [4, 8, 12, 16, 24]
def len_tok(n): return 0 if n <= 5 else 1 if n <= 9 else 2 if n <= 13 else 3 if n <= 19 else 4
def act_id(a): return a[0] + 2 * a[1] + 4 * a[2] + 8 * a[3]
# ---- planner vocab
P = dict(PAD=0, BOS=1, EOS=2, STY0=3, MAJ=12, MIN=13, TYPE0=14, LEN0=20, EN0=25, ACT0=29, TR0=45); PV = 50; PSLOT = 5; PMAX = 24; PT = 3 + PSLOT * PMAX + 1
def plan_encode(song):
    s = [P["BOS"], P["STY0"] + STY.index(song["style"]), P["MAJ"] if song["mode"] == "major" else P["MIN"]]
    for x in song["sections"][:PMAX]:
        s += [P["TYPE0"] + TYPES.index(x["type"]), P["LEN0"] + len_tok(x["len"]), P["EN0"] + x["energy"], P["ACT0"] + act_id(x["active"]), P["TR0"] + TRANS.index(x["transition"])]
    return s + [P["EOS"]]
def plan_decode(seq):
    out = []; s = list(map(int, seq))[3:]
    for i in range(0, len(s) - 4, 5):
        if s[i] == P["EOS"] or not (P["TYPE0"] <= s[i] < P["LEN0"]): break
        a = s[i + 3] - P["ACT0"]
        out.append(dict(type=TYPES[s[i] - P["TYPE0"]], len=LENS[s[i + 1] - P["LEN0"]], energy=s[i + 2] - P["EN0"], active=[a & 1, a >> 1 & 1, a >> 2 & 1, a >> 3 & 1], transition=TRANS[s[i + 4] - P["TR0"]]))
    return out
def plan_str(secs):
    return " ".join(f"{x['type']}{x['len']}:e{x['energy']}:{''.join(c for c, b in zip('DBKL', x['active']) if b) or '-'}:{x['transition']}" for x in secs)
def plan_mask(dev):
    mk = torch.full((PSLOT, PV), float("-inf"), device=dev)
    mk[0, P["TYPE0"]:P["LEN0"]] = 0; mk[0, P["EOS"]] = 0; mk[1, P["LEN0"]:P["EN0"]] = 0; mk[2, P["EN0"]:P["ACT0"]] = 0; mk[3, P["ACT0"]:P["TR0"]] = 0; mk[4, P["TR0"]:PV] = 0
    return mk
# ---- song LM vocab
S = dict(PAD=0, BOS=1, STY0=2, MAJ=11, MIN=12, BAR=13, SEC0=14, ACT0=38, CH0=54, NQ=5, M0=114, LO=57, HI=86, B0=146, BLO=28, BHI=55); SV = 176; SSLOT = 20; SBARS = 12; ST = 3 + SSLOT * SBARS
def lt(x):
    if x == -1: return S["M0"]
    if x == -2: return S["M0"] + 1
    while x < S["LO"]: x += 12
    while x > S["HI"]: x -= 12
    return S["M0"] + 2 + x - S["LO"]
def bt(x):
    if x == -1: return S["B0"]
    if x == -2: return S["B0"] + 1
    while x < S["BLO"]: x += 12
    while x > S["BHI"]: x -= 12
    return S["B0"] + 2 + x - S["BLO"]
def song_bars(song):
    sec_of = {}
    for x in song["sections"]:
        for i in range(x["start"], x["start"] + x["len"]): sec_of[i] = x
    out = []
    for i, b in enumerate(song["bars"]):
        x = sec_of.get(i, song["sections"][-1])
        out.append([S["BAR"], S["SEC0"] + TYPES.index(x["type"]) * 4 + x["energy"], S["ACT0"] + act_id(x["active"]), S["CH0"] + b["chord"][0] * S["NQ"] + b["chord"][1]] + [lt(v) for v in b["lead"]] + [bt(v) for v in b["bass"]])
    return out
def song_window(song, bars_tok, r=random):
    s0 = r.randint(0, max(0, len(bars_tok) - SBARS)); seq = [S["BOS"], S["STY0"] + STY.index(song["style"]), S["MAJ"] if song["mode"] == "major" else S["MIN"]]
    for b in bars_tok[s0:s0 + SBARS]: seq += b
    return seq + [S["PAD"]] * (ST - len(seq))
def song_weights(y):
    w = torch.zeros(y.shape); w[(y >= S["CH0"]) & (y < S["M0"])] = 1.; w[(y >= S["M0"]) & (y < S["B0"])] = .5; w[y >= S["B0"]] = 1.; return w
# ---- model (same tensors and math as the shipped kernel; slot period is a parameter)
class Block(nn.Module):
    def __init__(s, D, H, drop):
        super().__init__(); s.H, s.drop = H, drop; s.ln1 = nn.LayerNorm(D); s.qkv = nn.Linear(D, 3 * D); s.proj = nn.Linear(D, D); s.ln2 = nn.LayerNorm(D); s.fc = nn.Linear(D, 4 * D); s.out = nn.Linear(4 * D, D)
    def forward(s, x):
        B, t, D = x.shape; q, k, v = s.qkv(s.ln1(x)).split(D, 2); q, k, v = [z.view(B, t, s.H, D // s.H).transpose(1, 2) for z in (q, k, v)]
        x = x + F.dropout(s.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, t, D)), s.drop, s.training)
        return x + F.dropout(s.out(F.gelu(s.fc(s.ln2(x)), approximate="tanh")), s.drop, s.training)
class GPT(nn.Module):
    def __init__(s, V, T, SLOT, D, L, H, drop):
        super().__init__(); s.SLOT, s.drop = SLOT, drop; s.tok = nn.Embedding(V, D); s.pos = nn.Embedding(T, D); s.slot = nn.Embedding(SLOT, D)
        s.blocks = nn.ModuleList(Block(D, H, drop) for _ in range(L)); s.lnf = nn.LayerNorm(D); s.head = nn.Linear(D, V)
    def forward(s, idx):
        t = idx.shape[1]; ar = torch.arange(t, device=idx.device)
        x = F.dropout(s.tok(idx) + s.pos(ar) + s.slot((ar - 3).clamp(min=0) % s.SLOT), s.drop, s.training)
        for b in s.blocks: x = b(x)
        return s.head(s.lnf(x))
def split(songs):
    rs = random.Random(1); songs = sorted(songs, key=lambda z: z["name"]); rs.shuffle(songs); tr, va = [], []
    for st in STY:
        c = [z for z in songs if z["style"] == st]; n = max(2, len(c) // 10) if len(c) >= 6 else 0; va += c[:n]; tr += c[n:]
    return tr, va
def export(m, meta, path):
    sd = {k: v.detach().float().cpu().numpy().astype(np.float16) for k, v in m.state_dict().items()}; off = 0; idx = {}; buf = []
    for k, a in sd.items(): idx[k] = dict(shape=list(a.shape), off=off); off += a.size; buf.append(a.ravel())
    np.concatenate(buf).tofile(path + ".bin"); json.dump(dict(tensors=idx, **meta), open(path + ".json", "w")); return os.path.getsize(path + ".bin")
if __name__ == "__main__":
    KIND, NAME, D, L, H, MAXMIN = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
    DROP = float(os.environ.get("DROP", ".2")); LR = float(os.environ.get("LR", "1e-3")); EVERY = int(os.environ.get("EVERY", "100")); PAT = int(os.environ.get("PATIENCE", "10"))
    torch.manual_seed(0); random.seed(0); dev = "cuda"; O = "xfer/models"; os.makedirs(O, exist_ok=True); log = open(f"{O}/{NAME}.log", "w")
    def Pr(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
    songs = [s for s in json.load(open("xfer/songs/songs_dataset.json")) if s.get("sections")]; tr, va = split(songs)
    Pr("songs train", len(tr), "val", len(va), {st: sum(z["style"] == st for z in tr) for st in STY})
    if KIND == "planner":
        V, T, SLOT = PV, PT, PSLOT
        enc = lambda z: (plan_encode(z) + [0] * PT)[:PT]
        def batch(B, items):
            x = torch.tensor([enc(random.choice(items)) for _ in range(B)]); y = x[:, 1:]; w = (y != 0).float(); return x[:, :-1], y, w
        EVX = torch.tensor([enc(z) for z in va]); EV = [(EVX[:, :-1], EVX[:, 1:], (EVX[:, 1:] != 0).float())]
    else:
        V, T, SLOT = SV, ST, SSLOT
        trb = [(z, song_bars(z)) for z in tr]; vab = [(z, song_bars(z)) for z in va]
        def batch(B, items):
            x = torch.tensor([song_window(*random.choice(items)) for _ in range(B)]); y = x[:, 1:]; return x[:, :-1], y, song_weights(y)
        rows = []
        for z, bt_ in vab:
            for s0 in range(0, max(1, len(bt_) - SBARS + 1), SBARS // 2):
                seq = [S["BOS"], S["STY0"] + STY.index(z["style"]), S["MAJ"] if z["mode"] == "major" else S["MIN"]]
                for b in bt_[s0:s0 + SBARS]: seq += b
                rows.append(seq + [0] * (ST - len(seq)))
        EVX = torch.tensor(rows); y = EVX[:, 1:]; EV = [(EVX[:, :-1], y, song_weights(y))]
        Pr("train bars", sum(len(b) for _, b in trb), "val bars", sum(len(b) for _, b in vab), "val windows", len(rows))
        items_tr = trb
    items = tr if KIND == "planner" else items_tr
    m = GPT(V, T, SLOT, D, L, H, DROP).to(dev); NP = sum(p.numel() for p in m.parameters()); Pr("params", NP, "D L H", D, L, H, "V", V, "T", T, "SLOT", SLOT, "drop", DROP, "lr", LR)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=.1, betas=(.9, .98))
    @torch.no_grad()
    def evaluate():
        m.eval(); r = {}
        for x, y, w in EV:
            tot = n = 0.; parts = {}
            for i in range(0, len(x), 128):
                xb, yb, wb = x[i:i + 128].to(dev), y[i:i + 128].to(dev), w[i:i + 128].to(dev)
                l = F.cross_entropy(m(xb).float().reshape(-1, V), yb.reshape(-1), reduction="none").reshape(yb.shape)
                tot += (l * wb).sum().item(); n += wb.sum().item()
                if KIND == "songlm":
                    for nm, mm in (("chord", (yb >= S["CH0"]) & (yb < S["M0"])), ("lead", (yb >= S["M0"]) & (yb < S["B0"])), ("bass", yb >= S["B0"])):
                        a, c = parts.get(nm, (0., 0)); parts[nm] = (a + (l * mm).sum().item(), c + mm.sum().item())
            r["val"] = tot / n
            for nm, (a, c) in parts.items(): r["val_" + nm] = a / max(1, c)
        m.train(); return r
    t0, step, best, bad, hist = time.time(), 0, 9e9, 0, []
    while time.time() - t0 < MAXMIN * 60:
        lr = LR * min(1, (step + 1) / 200) * max(.1, .5 * (1 + math.cos(math.pi * min(1, step / 10000))))
        for g in opt.param_groups: g["lr"] = lr
        x, y, w = batch(64, items); x, y, w = x.to(dev), y.to(dev), w.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = m(x)
        loss = (F.cross_entropy(lg.float().reshape(-1, V), y.reshape(-1), reduction="none") * w.reshape(-1)).sum() / w.sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1); opt.step(); step += 1
        if step % EVERY == 0:
            r = evaluate(); r.update(step=step, train=round(loss.item(), 4), secs=round(time.time() - t0)); hist.append(r); Pr(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}))
            if r["val"] < best - 1e-3: best = r["val"]; bad = 0; torch.save(m.state_dict(), f"{O}/{NAME}.pt")
            else:
                bad += 1
                if bad >= PAT: Pr("early stop"); break
    m.load_state_dict(torch.load(f"{O}/{NAME}.pt", weights_only=True)); bi = min(range(len(hist)), key=lambda i: hist[i]["val"])
    # baseline for promotion: style-conditional unigram per slot (add-0.5 smoothing), same targets and weights as "val"
    if KIND == "planner": TR = torch.tensor([enc(z) for z in tr])
    else:
        rr = []
        for z, bt_ in items_tr:
            for s0 in range(0, max(1, len(bt_) - SBARS + 1), SBARS // 2):
                seq = [S["BOS"], S["STY0"] + STY.index(z["style"]), S["MAJ"] if z["mode"] == "major" else S["MIN"]]
                for b in bt_[s0:s0 + SBARS]: seq += b
                rr.append(seq + [0] * (ST - len(seq)))
        TR = torch.tensor(rr)
    def slots(n): return torch.tensor([p if p < 3 else 3 + (p - 3) % SLOT for p in range(1, n + 1)])
    sl = slots(TR.shape[1] - 1); C = torch.full((len(STY), SLOT + 3, V), .5)
    yt = TR[:, 1:]; stt = TR[:, 1] - (P["STY0"] if KIND == "planner" else S["STY0"])
    for i in range(len(TR)):
        mk_ = yt[i] != 0; C[stt[i], sl[mk_], yt[i][mk_]] += 1
    LP = torch.log(C / C.sum(-1, keepdim=True))
    x, y, w = EV[0]; sv = x[:, 1] - (P["STY0"] if KIND == "planner" else S["STY0"]); slv = slots(y.shape[1])
    nll = -LP[sv[:, None].expand_as(y), slv[None].expand_as(y), y]
    base = float((nll * w).sum() / w.sum())
    meta = dict(kind=KIND, D=D, L=L, H=H, V=V, T=T, SLOT=SLOT, params=NP, **(P if KIND == "planner" else S), STYLES=STY, TYPES=TYPES, TRANS=TRANS, LENS=LENS, TRACKS=["drums", "bass", "keys", "lead"])
    nb = export(m, meta, f"{O}/{NAME}")
    res = dict(name=NAME, best=hist[bi], baseline_unigram_val=base, promoted=hist[bi]["val"] < base, hist=hist, params=NP, bytes=nb)
    m.eval(); mk_dev = dev
    if KIND == "planner":
        mk = plan_mask(dev); samples = {}; stats = {}
        def st_of(plans):
            n = [len(p) for p in plans]; L_ = [sum(x["len"] for x in p) for p in plans]
            ch = [sum(p[i]["active"] != p[i - 1]["active"] for i in range(1, len(p))) / max(1, len(p) - 1) for p in plans if len(p) > 1]
            ty = {t: sum(x["type"] == t for p in plans for x in p) / max(1, sum(n)) for t in TYPES}
            return dict(n=len(plans), sections_mean=float(np.mean(n)), sections_sd=float(np.std(n)), bars_mean=float(np.mean(L_)), bars_sd=float(np.std(L_)), active_change_rate=float(np.mean(ch)) if ch else 0., type_share=ty)
        for st in STY:
            seqs = []
            with torch.no_grad():
                for k in range(100):
                    x = torch.tensor([[P["BOS"], P["STY0"] + STY.index(st), P["MAJ"] if k % 2 else P["MIN"]]], device=dev)
                    for p in range(3, PT):
                        lg = m(x)[0, -1].float() + mk[(p - 3) % PSLOT]; t = torch.multinomial(F.softmax(lg, -1), 1)
                        x = torch.cat([x, t[None]], 1)
                        if t.item() == P["EOS"]: break
                    seqs.append(plan_decode(x[0].tolist()))
            samples[st] = [plan_str(p) for p in seqs[:10]]
            real = [[dict(x, len=LENS[len_tok(x["len"])]) for x in z["sections"]] for z in songs if z["style"] == st] or [[]]
            stats[st] = dict(sampled=st_of(seqs), real=st_of(real))
        res.update(samples=samples, stats=stats)
    else:
        # conditioning check: sample chord/lead/bass for held-out windows with the real BAR/SEC/ACT tokens fed in
        cnt = {"bass_on": [0, 0], "bass_off": [0, 0], "lead_on": [0, 0], "lead_off": [0, 0]}
        with torch.no_grad():
            for x0 in EVX[:96]:
                seq = x0.tolist(); x = torch.tensor([seq[:3]], device=dev)
                for p in range(3, ST):
                    sl = (p - 3) % SSLOT
                    if seq[p] == 0: break
                    if sl < 3: t = seq[p]
                    else:
                        lg = m(x)[0, -1].float(); lo, hi = (S["CH0"], S["M0"]) if sl == 3 else (S["M0"], S["B0"]) if sl < 12 else (S["B0"], SV)
                        mm = torch.full_like(lg, float("-inf")); mm[lo:hi] = 0; t = torch.multinomial(F.softmax(lg + mm, -1), 1).item()
                    x = torch.cat([x, torch.tensor([[t]], device=dev)], 1)
                s = x[0].tolist()[3:]
                for i in range(0, len(s) - 19, 20):
                    a = s[i + 2] - S["ACT0"]; lead = s[i + 4:i + 12]; bass = s[i + 12:i + 20]
                    k = "bass_on" if a >> 1 & 1 else "bass_off"; cnt[k][0] += sum(v >= S["B0"] + 2 for v in bass); cnt[k][1] += 8
                    k = "lead_on" if a >> 3 & 1 else "lead_off"; cnt[k][0] += sum(S["M0"] + 2 <= v < S["B0"] for v in lead); cnt[k][1] += 8
        res["note_rate_given_active"] = {k: v[0] / max(1, v[1]) for k, v in cnt.items()}
        real = {"bass_on": [0, 0], "bass_off": [0, 0], "lead_on": [0, 0], "lead_off": [0, 0]}
        for x0 in EVX[:96]:
            s = [v for v in x0.tolist()[3:]]
            for i in range(0, len(s) - 19, 20):
                if s[i] != S["BAR"]: break
                a = s[i + 2] - S["ACT0"]; lead = s[i + 4:i + 12]; bass = s[i + 12:i + 20]
                k = "bass_on" if a >> 1 & 1 else "bass_off"; real[k][0] += sum(v >= S["B0"] + 2 for v in bass); real[k][1] += 8
                k = "lead_on" if a >> 3 & 1 else "lead_off"; real[k][0] += sum(S["M0"] + 2 <= v < S["B0"] for v in lead); real[k][1] += 8
        res["note_rate_given_active_real"] = {k: v[0] / max(1, v[1]) for k, v in real.items()}
    json.dump(res, open(f"{O}/{NAME}.result.json", "w"), indent=1); Pr("done", NAME, json.dumps(hist[bi]), "baseline", round(base, 4), "bytes", nb)
