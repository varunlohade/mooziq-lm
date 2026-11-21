# Shared vocab + tiny transformer. One bar = 10 tokens: BAR, CHORD, 8 eighth-note melody steps.
import numpy as np
PAD, BOS, LOFI, FOLK, MAJ, MIN, BAR = range(7)
QN = ["maj7", "m7", "7", "m7b5", "sus"]; NQ = len(QN)
CH0 = 7                                   # chord token = CH0 + root*NQ + quality
M0 = CH0 + 12 * NQ                        # melody: M0 rest, M0+1 hold, M0+2+(p-LO) pitch
LO, HI = 57, 86                           # A3..D6
V = M0 + 2 + (HI - LO + 1)
BARS = 16; T = 3 + 10 * BARS              # BOS STYLE MODE + 16 bars
NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
def ch_tok(r, q): return CH0 + r * NQ + q
def ch_dec(t): t -= CH0; return t // NQ, t % NQ
def m_tok(x):
    if x == -1: return M0
    if x == -2: return M0 + 1
    while x < LO: x += 12
    while x > HI: x -= 12
    return M0 + 2 + x - LO
def m_dec(t): return -1 if t == M0 else -2 if t == M0 + 1 else LO + t - M0 - 2
def encode(style, mode, bars):
    s = [BOS, style, MAJ if mode == "major" else MIN]
    for r, q, mel in bars: s += [BAR, ch_tok(r, q)] + [m_tok(x) for x in mel]
    return s
D, L, H = 96, 4, 4
DROP = .15
try:
    import torch, torch.nn as nn, torch.nn.functional as F
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
            x = s.tok(idx) + s.pos(ar) + s.slot((ar - 3).clamp(min=0) % 10)
            for b in s.blocks: x = b(x)
            return s.head(s.lnf(x))
except ImportError:
    pass

# ---- numpy inference with a KV cache (this is exactly what the browser port does in plain JS / Float32Array)
def ln(x, g, b): m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True); return (x - m) / np.sqrt(v + 1e-5) * g + b
def gelu(x): return .5 * x * (1 + np.tanh(.7978845608 * (x + .044715 * x ** 3)))
class NPModel:
    def __init__(s, path):
        z = np.load(path); s.w = {k: z[k].astype(np.float32) for k in z.files}
    def reset(s): s.k = [[] for _ in range(L)]; s.v = [[] for _ in range(L)]; s.n = 0
    def step(s, tok):                       # feed one token, return logits for the next
        w = s.w; p = s.n
        x = w["tok.weight"][tok] + w["pos.weight"][p] + w["slot.weight"][max(0, p - 3) % 10]
        hd = D // H
        for i in range(L):
            pre = f"blocks.{i}."
            h = ln(x, w[pre + "ln1.weight"], w[pre + "ln1.bias"])
            qkv = w[pre + "qkv.weight"] @ h + w[pre + "qkv.bias"]; q, k, v = qkv[:D], qkv[D:2 * D], qkv[2 * D:]
            s.k[i].append(k); s.v[i].append(v)
            K = np.stack(s.k[i]).reshape(-1, H, hd); Vv = np.stack(s.v[i]).reshape(-1, H, hd)
            att = np.einsum("hd,thd->ht", q.reshape(H, hd), K) / np.sqrt(hd)
            att = np.exp(att - att.max(1, keepdims=True)); att /= att.sum(1, keepdims=True)
            y = np.einsum("ht,thd->hd", att, Vv).reshape(D)
            x = x + w[pre + "proj.weight"] @ y + w[pre + "proj.bias"]
            h = ln(x, w[pre + "ln2.weight"], w[pre + "ln2.bias"])
            x = x + w[pre + "out.weight"] @ gelu(w[pre + "fc.weight"] @ h + w[pre + "fc.bias"]) + w[pre + "out.bias"]
        s.n += 1
        return w["head.weight"] @ ln(x, w["lnf.weight"], w["lnf.bias"]) + w["head.bias"]
