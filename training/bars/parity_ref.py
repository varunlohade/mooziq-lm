import sys, json, numpy as np
d = sys.argv[1]; style = sys.argv[2] if len(sys.argv) > 2 else "JAZZHOP"
meta = json.load(open(f"{d}/model.json")); raw = np.fromfile(f"{d}/model.bin", dtype=np.float16)
W = {k: raw[t["off"]:t["off"] + int(np.prod(t["shape"]))].astype(np.float64).reshape(t["shape"]) for k, t in meta["tensors"].items()}
D, L, H = meta["D"], meta["L"], meta["H"]; hd = D // H
def ln(x, g, b): m = x.mean(); v = ((x - m) ** 2).mean(); return (x - m) / np.sqrt(v + 1e-5) * g + b
gelu = lambda x: .5 * x * (1 + np.tanh(.7978845608 * (x + .044715 * x ** 3)))
seq = [meta["BOS"], meta[style], meta["MIN"], meta["BAR"], meta["CH0"] + 2 * meta["NQ"] + 1]
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
json.dump([float(v) for v in lg], open(f"{d}/parity_ref.json", "w")); print("ref ok", len(lg))
