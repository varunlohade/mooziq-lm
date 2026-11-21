# float64 numpy reference logits for a quantized student file (dequantized exactly as FORMAT.md says). Usage: parity_ref_q.py BIN JSON OUT
import sys, json, numpy as np
bp, jp, outp = sys.argv[1:4]; meta = json.load(open(jp)); b = open(bp, "rb").read(); W = {}
for k, t in meta["tensors"].items():
    n = int(np.prod(t["shape"]))
    if t["dtype"] == "f16": W[k] = np.frombuffer(b, np.float16, n, t["off"]).astype(np.float64).reshape(t["shape"]); continue
    sc = np.frombuffer(b, np.float16, t["nscale"], t["scale_off"]).astype(np.float64)
    if t["dtype"] == "int8": q = np.frombuffer(b, np.int8, n, t["off"]).astype(np.float64)
    else:
        u = np.frombuffer(b, np.uint8, (n + 1) // 2, t["off"]); q = np.empty(2 * len(u)); q[0::2] = (u & 15) - 8.; q[1::2] = (u >> 4) - 8.; q = q[:n]
    W[k] = (q.reshape(-1, t["group"]) * sc[:, None]).reshape(t["shape"])
D, L, H = meta["D"], meta["L"], meta["H"]; hd = D // H
def ln(x, g, bb): m = x.mean(); v = ((x - m) ** 2).mean(); return (x - m) / np.sqrt(v + 1e-5) * g + bb
gelu = lambda x: .5 * x * (1 + np.tanh(.7978845608 * (x + .044715 * x ** 3)))
seq = [meta["BOS"], meta["JAZZHOP"], meta["MIN"], meta["BAR"], meta["CH0"] + 2 * meta["NQ"] + 1]
K = [[] for _ in range(L)]; Vc = [[] for _ in range(L)]
for p, tok in enumerate(seq):
    x = W["tok.weight"][tok] + W["pos.weight"][p] + W["slot.weight"][max(0, p - 3) % 10]
    for i in range(L):
        pre = f"blocks.{i}."; h = ln(x, W[pre + "ln1.weight"], W[pre + "ln1.bias"]); qkv = W[pre + "qkv.weight"] @ h + W[pre + "qkv.bias"]
        K[i].append(qkv[D:2 * D]); Vc[i].append(qkv[2 * D:]); Ks = np.stack(K[i]).reshape(-1, H, hd); Vs = np.stack(Vc[i]).reshape(-1, H, hd)
        a = np.einsum("hd,thd->ht", qkv[:D].reshape(H, hd), Ks) / np.sqrt(hd); a = np.exp(a - a.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
        x = x + W[pre + "proj.weight"] @ np.einsum("ht,thd->hd", a, Vs).reshape(D) + W[pre + "proj.bias"]
        h = ln(x, W[pre + "ln2.weight"], W[pre + "ln2.bias"]); x = x + W[pre + "out.weight"] @ gelu(W[pre + "fc.weight"] @ h + W[pre + "fc.bias"]) + W[pre + "out.bias"]
lg = W["head.weight"] @ ln(x, W["lnf.weight"], W["lnf.bias"]) + W["head.bias"]
json.dump([float(v) for v in lg], open(outp, "w")); print("ref ok", len(lg))
