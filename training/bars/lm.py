# Shared pieces for stage 2 (teacher, distilled students, audit, quantized export). Same vocab/layout as train_pod.py and the shipped kernel.
import os, json, math, time, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils import parametrize
PAD, BOS, LOFI, FOLK, MAJ, MIN, BAR = range(7)
NQ = 5; CH0 = 7; M0 = CH0 + 12 * NQ; LO, HI = 57, 86; V0 = M0 + 2 + (HI - LO + 1)
JAZZHOP, FLOW = V0, V0 + 1; V = V0 + 2
STYLE_TOK = {"jazzhop": JAZZHOP, "steady": LOFI, "flow": FLOW, "folk": FOLK}
BARS = 16; T = 3 + 10 * BARS
QIV = [[0, 4, 7, 11], [0, 3, 7, 10], [0, 4, 7, 10], [0, 3, 6, 10], [0, 5, 7, 10]]
TOKENS = dict(PAD=PAD, BOS=BOS, LOFI=LOFI, STEADY=LOFI, FOLK=FOLK, JAZZHOP=JAZZHOP, FLOW=FLOW, MAJ=MAJ, MIN=MIN, BAR=BAR, CH0=CH0, M0=M0, NQ=NQ, LO=LO, HI=HI)
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
    def __init__(s, D, H, drop):
        super().__init__(); s.D, s.H, s.drop = D, H, drop
        s.ln1 = nn.LayerNorm(D); s.qkv = nn.Linear(D, 3 * D); s.proj = nn.Linear(D, D)
        s.ln2 = nn.LayerNorm(D); s.fc = nn.Linear(D, 4 * D); s.out = nn.Linear(4 * D, D)
    def forward(s, x):
        B, t, D = x.shape; H = s.H
        q, k, v = s.qkv(s.ln1(x)).split(D, 2)
        q, k, v = [z.view(B, t, H, D // H).transpose(1, 2) for z in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + F.dropout(s.proj(y.transpose(1, 2).reshape(B, t, D)), s.drop, s.training)
        return x + F.dropout(s.out(F.gelu(s.fc(s.ln2(x)), approximate="tanh")), s.drop, s.training)
class GPT(nn.Module):
    def __init__(s, D, L, H, drop=.1):
        super().__init__(); s.D, s.L, s.H, s.drop = D, L, H, drop
        s.tok = nn.Embedding(V, D); s.pos = nn.Embedding(T, D); s.slot = nn.Embedding(10, D)
        s.blocks = nn.ModuleList(Block(D, H, drop) for _ in range(L)); s.lnf = nn.LayerNorm(D); s.head = nn.Linear(D, V)
    def forward(s, idx):
        t = idx.shape[1]; ar = torch.arange(t, device=idx.device)
        x = F.dropout(s.tok(idx) + s.pos(ar) + s.slot((ar - 3).clamp(min=0) % 10), s.drop, s.training)
        for b in s.blocks: x = b(x)
        return s.head(s.lnf(x))
class Ens(nn.Module):
    def __init__(s, ms): super().__init__(); s.ms = nn.ModuleList(ms)
    def forward(s, x): return torch.logsumexp(torch.stack([F.log_softmax(m(x).float(), -1) for m in s.ms]), 0) - math.log(len(s.ms))
def nparams(m): return sum(p.numel() for p in m.parameters())
# ---------------- data (same split procedure as train_pod.py)
def load_data(ace_path="data/ace2_bars.json", pd_path="data/pd_bars.json"):
    pd = json.load(open(pd_path)); ace = json.load(open(ace_path)); ace = [a for a in ace if len(a["bars"]) >= 8]
    rs = random.Random(1); rs.shuffle(ace); rs.shuffle(pd)
    val, tr = [], []
    for st in ("jazzhop", "steady", "flow"):
        c = [a for a in ace if a["style"] == st]; n = max(2, len(c) // 10) if len(c) >= 4 else 0; val += c[:n]; tr += c[n:]
    return tr, val, pd[150:], pd[:150]
def window(p, style, r=random):
    b = p["bars"]; s = r.randint(0, max(0, len(b) - BARS)); seq = encode(style, p["mode"], b[s:s + BARS])
    return seq + [PAD] * (T - len(seq))
def make_eval(val, pd_val):
    er = random.Random(5)
    def fixed(items, styl, n):
        rows = [window(it, styl(it), er) for it in (items * (n // max(1, len(items)) + 1))[:n]]
        x = torch.tensor(rows); return x[:, :-1], x[:, 1:]
    EV = {"ace": fixed(val, lambda a: a["style"], 512), "pd": fixed(pd_val, lambda a: "folk", 256)}
    for st in ("jazzhop", "steady", "flow"):
        if any(a["style"] == st for a in val): EV[st] = fixed([a for a in val if a["style"] == st], lambda a: a["style"], 192)
    return EV
class Batcher:
    def __init__(s, tr, pd_tr, melw=.5, noise=0., frac=.75): s.tr, s.pd, s.melw, s.noise, s.frac = tr, pd_tr, melw, noise, frac
    def __call__(s, B):
        rows, fl = [], []
        for _ in range(B):
            if random.random() < s.frac: a = random.choice(s.tr); rows.append(window(a, a["style"])); fl.append(True)
            else: rows.append(window(random.choice(s.pd), "folk")); fl.append(False)
        x = torch.tensor(rows); xi = x[:, :-1].clone(); y = x[:, 1:]
        w = torch.ones(y.shape); af = torch.tensor(fl)[:, None]
        w[((y >= M0) & (y < V0)) & af] = s.melw
        if s.noise > 0:
            r = torch.rand(xi.shape) < s.noise
            ch = r & (xi >= CH0) & (xi < M0); me = r & (xi >= M0) & (xi < V0)
            xi[ch] = torch.randint(CH0, M0, (int(ch.sum()),)); xi[me] = torch.randint(M0, V0, (int(me.sum()),))
        return xi, y, w
def ce(lg, y, w):
    w = w * (y != PAD); l = F.cross_entropy(lg.reshape(-1, V).float(), y.reshape(-1), reduction="none") * w.reshape(-1); return l.sum() / w.sum()
@torch.no_grad()
def evaluate(m, EV, dev="cuda"):
    was = m.training; m.eval(); r = {}
    for k, (x, y) in EV.items():
        tot = 0.; n = 0
        for i in range(0, len(x), 128):
            xb, yb = x[i:i + 128].to(dev), y[i:i + 128].to(dev); lg = m(xb).float(); msk = (yb != PAD)
            l = F.cross_entropy(lg.reshape(-1, V), yb.reshape(-1), reduction="none").reshape(yb.shape)
            tot += (l * msk).sum().item(); n += msk.sum().item()
            if k == "ace":
                for nm, mm in (("ace_chord", (yb >= CH0) & (yb < M0)), ("ace_mel", (yb >= M0) & (yb < V0))):
                    r.setdefault(nm + "_s", 0.); r.setdefault(nm + "_n", 0)
                    r[nm + "_s"] += (l * mm).sum().item(); r[nm + "_n"] += mm.sum().item()
        r[k] = tot / n
    for nm in ("ace_chord", "ace_mel"): r[nm] = r.pop(nm + "_s") / max(1, r.pop(nm + "_n"))
    m.train(was); return r
# ---------------- sampling with slot masks (BAR / chord / melody positions)
def slot_mask(dev):
    mk = torch.full((10, V), float("-inf"), device=dev)
    mk[0, BAR] = 0; mk[1, CH0:M0] = 0; mk[2:, M0:V0] = 0; return mk
def _kv_step(m, K, Vv, tok, p):
    D, H = m.D, m.H; hd = D // H; B = tok.shape[0]
    x = (m.tok(tok) + m.pos.weight[p] + m.slot.weight[max(0, p - 3) % 10])[:, None]
    for li, b in enumerate(m.blocks):
        q, k, v = b.qkv(b.ln1(x)).split(D, 2)
        q, k, v = [z.view(B, 1, H, hd).transpose(1, 2) for z in (q, k, v)]
        K[li, :, :, p] = k[:, :, 0]; Vv[li, :, :, p] = v[:, :, 0]
        y = F.scaled_dot_product_attention(q, K[li, :, :, :p + 1], Vv[li, :, :, :p + 1])
        x = x + b.proj(y.transpose(1, 2).reshape(B, 1, D))
        x = x + b.out(F.gelu(b.fc(b.ln2(x)), approximate="tanh"))
    return m.head(m.lnf(x))[:, 0].float()
@torch.no_grad()
def sample(m, styles, modes, temp=1.0, dev="cuda", bs=256):
    """KV-cached slot-masked sampling. m may be a GPT or an Ens (probabilities averaged per step)."""
    ms = list(m.ms) if isinstance(m, Ens) else [m]; was = [x.training for x in ms]
    for x in ms: x.eval()
    mk = slot_mask(dev); out = []
    for i in range(0, len(styles), bs):
        st, mo = styles[i:i + bs], modes[i:i + bs]; B = len(st)
        seq = torch.tensor([[BOS, STYLE_TOK[a], MAJ if b == "major" else MIN] for a, b in zip(st, mo)], device=dev)
        caches = [(torch.zeros(x.L, B, x.H, T, x.D // x.H, device=dev, dtype=torch.bfloat16),) * 1 + (torch.zeros(x.L, B, x.H, T, x.D // x.H, device=dev, dtype=torch.bfloat16),) for x in ms]
        toks = [seq[:, j] for j in range(3)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for p in range(T):
                lps = [F.log_softmax(_kv_step(x, c[0], c[1], toks[p], p), -1) for x, c in zip(ms, caches)]
                if p + 1 >= T: break
                if p + 1 < 3: continue
                lp = torch.logsumexp(torch.stack(lps), 0) - math.log(len(lps))
                toks.append(torch.multinomial(F.softmax(lp / temp + mk[(p + 1 - 3) % 10], -1), 1)[:, 0])
        out.append(torch.stack(toks, 1).cpu())
    for x, w in zip(ms, was): x.train(w)
    return torch.cat(out)
def seq_bars(seq):
    s = list(map(int, seq))[3:]; bars = []
    for i in range(0, len(s) - 9, 10):
        if s[i] != BAR: break
        c = s[i + 1] - CH0; mel = [(-1 if t == M0 else -2 if t == M0 + 1 else LO + t - M0 - 2) for t in s[i + 2:i + 10]]
        bars.append((c // NQ, c % NQ, mel))
    return bars
def audit(bar_lists):
    st = dict(strong=0, strong_ct=0, strong_ct9=0, notes=0, rests=0, steps=0, leaps=0, ints=0, intsum=0, mel_rep4=0, mel_rep_any=0, ch_rep4=0, nb4=0, nbars=0, chords_distinct=0, songs=0, range_sum=0)
    for bars in bar_lists:
        if len(bars) < 8: continue
        st["songs"] += 1; prev = None; pitches = []
        st["chords_distinct"] += len({(r, q) for r, q, _ in bars})
        for i, (r, q, mel) in enumerate(bars):
            st["nbars"] += 1; tones = {(r + k) % 12 for k in QIV[q]}
            for k, p in enumerate(mel):
                st["steps"] += 1
                if p == -1: st["rests"] += 1
                if p >= 0:
                    st["notes"] += 1; pitches.append(p)
                    if prev is not None: d = abs(p - prev); st["ints"] += 1; st["intsum"] += d; st["leaps"] += d > 7
                    prev = p
                    if k % 2 == 0:
                        st["strong"] += 1; st["strong_ct"] += p % 12 in tones; st["strong_ct9"] += (p % 12 in tones) or (p - r) % 12 == 2
            if i >= 4:
                st["nb4"] += 1; st["mel_rep4"] += mel == bars[i - 4][2]; st["ch_rep4"] += (r, q) == bars[i - 4][:2]
            if i >= 1: st["mel_rep_any"] += any(mel == bars[j][2] for j in range(i))
        if pitches: st["range_sum"] += max(pitches) - min(pitches)
    S = max(1, st["songs"])
    return dict(songs=st["songs"], strong_beat_chord_tone=st["strong_ct"] / max(1, st["strong"]), strong_beat_chord_tone_or_9th=st["strong_ct9"] / max(1, st["strong"]),
                notes_per_bar=st["notes"] / max(1, st["nbars"]), rest_share=st["rests"] / max(1, st["steps"]), mean_interval=st["intsum"] / max(1, st["ints"]),
                leap_gt5th=st["leaps"] / max(1, st["ints"]), melody_repeat_4bars_ago=st["mel_rep4"] / max(1, st["nb4"]),
                melody_repeat_any_earlier=st["mel_rep_any"] / max(1, st["nbars"] - S), chord_repeat_4bars_ago=st["ch_rep4"] / max(1, st["nb4"]),
                distinct_chords_per_16=st["chords_distinct"] / S, pitch_range=st["range_sum"] / S)
def real_bars(items, n=200, seed=3):
    r = random.Random(seed); out = []
    for it in (items * (n // max(1, len(items)) + 1))[:n]:
        b = it["bars"]; s = r.randint(0, max(0, len(b) - BARS)); out.append([(x[0], x[1], x[2]) for x in b[s:s + BARS]])
    return out
# ---------------- fake quant (symmetric, per group of `group` input elements, STE)
class FQ(nn.Module):
    def __init__(s, bits, group): super().__init__(); s.bits, s.group = bits, group
    def forward(s, w):
        shp = w.shape; g = s.group if s.group and shp[-1] % s.group == 0 else shp[-1]
        wg = w.reshape(-1, g); qmax = 2 ** (s.bits - 1) - 1
        sc = wg.abs().amax(1, keepdim=True).clamp(min=1e-8) / qmax
        q = (wg / sc).round().clamp(-qmax - 1, qmax) * sc
        return w + (q.reshape(shp) - w).detach()
def qnames(m): return [n for n, p in m.named_parameters() if p.dim() == 2]
def add_fq(m, bits, group):
    for n in qnames(m):
        mod = m.get_submodule(n.rsplit(".", 1)[0]); parametrize.register_parametrization(mod, "weight", FQ(bits, group))
def plain_state(m):   # weights as seen by the forward pass (quantized if FQ active), with original names
    sd = {}
    for n, p in m.named_parameters():
        if ".parametrizations." in n: continue
        sd[n] = p.detach()
    for n in qnames_param(m): sd[n] = m.get_submodule(n.rsplit(".", 1)[0]).weight.detach()
    return {k: v.float().cpu() for k, v in sd.items()}
def qnames_param(m):
    out = []
    for n, mod in m.named_modules():
        if parametrize.is_parametrized(mod, "weight"): out.append(n + ".weight")
    return out
def raw_state(m):   # original (unquantized) parameters with plain names
    sd = {}
    for n, p in m.named_parameters():
        n2 = n.replace(".parametrizations.weight.original", ".weight"); sd[n2] = p.detach().float().cpu()
    return sd
def export_quant(sd, bits, group, path_bin, path_json, meta_extra):
    """Binary layout (little endian), tensors in sd order:
       f16 tensor: raw float16, row-major.
       int8 tensor: int8 row-major values q in [-128,127]; weight = q * scale[g], one fp16 scale per `group` consecutive input elements (per row if group = in-dim).
       int4 tensor: two values per byte, low nibble = even index; stored as q+8 (0..15), q in [-8,7]; same scale rule.
       Every tensor starts at a 4-byte aligned offset; scales are a separate fp16 array at scale_off."""
    buf = bytearray(); idx = {}; qmax = 2 ** (bits - 1) - 1
    def align():
        while len(buf) % 4: buf.append(0)
    for k, w in sd.items():
        a = w.numpy().astype(np.float32)
        if a.ndim == 2 and bits in (4, 8):
            g = group if group and a.shape[1] % group == 0 else a.shape[1]
            wg = a.reshape(-1, g); sc = np.maximum(np.abs(wg).max(1), 1e-8) / qmax; sc16 = sc.astype(np.float16)
            q = np.clip(np.round(wg / sc16.astype(np.float32)[:, None]), -qmax - 1, qmax).astype(np.int8).ravel()
            align(); off = len(buf)
            if bits == 8: buf += q.tobytes()
            else:
                u = (q + 8).astype(np.uint8)
                if len(u) % 2: u = np.append(u, 8)
                buf += (u[0::2] | (u[1::2] << 4)).astype(np.uint8).tobytes()
            align(); soff = len(buf); buf += sc16.tobytes()
            idx[k] = dict(shape=list(a.shape), dtype=f"int{bits}", off=off, group=g, scale_off=soff, nscale=len(sc16))
        else:
            align(); off = len(buf); buf += a.astype(np.float16).tobytes(); idx[k] = dict(shape=list(a.shape), dtype="f16", off=off)
    open(path_bin, "wb").write(bytes(buf))
    json.dump(dict(format=f"tinylofi-q{bits}", tensors=idx, **meta_extra), open(path_json, "w"))
    return len(buf)
def dequant_file(path_bin, path_json):
    meta = json.load(open(path_json)); b = open(path_bin, "rb").read(); out = {}
    for k, t in meta["tensors"].items():
        n = int(np.prod(t["shape"]))
        if t["dtype"] == "f16": out[k] = np.frombuffer(b, np.float16, n, t["off"]).astype(np.float32).reshape(t["shape"]); continue
        sc = np.frombuffer(b, np.float16, t["nscale"], t["scale_off"]).astype(np.float32)
        if t["dtype"] == "int8": q = np.frombuffer(b, np.int8, n, t["off"]).astype(np.float32)
        else:
            u = np.frombuffer(b, np.uint8, (n + 1) // 2, t["off"]); q = np.empty(2 * len(u), np.float32); q[0::2] = (u & 15).astype(np.float32) - 8; q[1::2] = (u >> 4).astype(np.float32) - 8; q = q[:n]
        out[k] = (q.reshape(-1, t["group"]) * sc[:, None]).reshape(t["shape"])
    return out, meta

# ---------------- ensembles (teacher = probability average of several trained models)
def load_models(names, O="xfer/ckpt"):
    ms = []
    for n in names.split(","):
        j = json.load(open(f"{O}/{n}.json")); m = GPT(j["D"], j["L"], j["H"], 0.).cuda()
        if os.path.exists(f"{O}/{n}.pt"): m.load_state_dict(torch.load(f"{O}/{n}.pt"))
        else: z = np.load(f"{O}/{n}.npz"); m.load_state_dict({k: torch.tensor(z[k].astype(np.float32)) for k in z.files})
        m.eval(); ms.append(m)
    return ms[0] if len(ms) == 1 else Ens(ms).eval()
@torch.no_grad()
def sample_full(m, styles, modes, temp=1.0, dev="cuda", bs=512):
    """Full-recompute sampler for any module (used for ensembles)."""
    mk = slot_mask(dev); out = []
    for i in range(0, len(styles), bs):
        st, mo = styles[i:i + bs], modes[i:i + bs]
        x = torch.tensor([[BOS, STYLE_TOK[a], MAJ if b == "major" else MIN] for a, b in zip(st, mo)], device=dev)
        for p in range(3, T):
            with torch.autocast("cuda", dtype=torch.bfloat16): lg = m(x)[:, -1].float()
            x = torch.cat([x, torch.multinomial(F.softmax(lg / temp + mk[(p - 3) % 10], -1), 1)], 1)
        out.append(x.cpu())
    return torch.cat(out)
