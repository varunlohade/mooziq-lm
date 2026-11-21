# Held-out loss + music audit for every model. Models: fp16 npz (baselines), .pt (teacher), quant files (students).
import sys, json, glob, os, torch, numpy as np
sys.path.insert(0, "code"); from lm import *
O = "xfer/ckpt"; N = int(sys.argv[1]) if len(sys.argv) > 1 else 96
tr, val, pd_tr, pd_val = load_data(); EV = make_eval(val, pd_val)
def load(spec):
    kind, name = spec.split(":")
    if kind == "ens": m = load_models(name); return m, dict(params=sum(p.numel() for p in m.parameters()))
    j = json.load(open(f"{O}/{name}.json"))
    m = GPT(j["D"], j["L"], j["H"], 0.).cuda()
    if kind == "pt": m.load_state_dict(torch.load(f"{O}/{name}.pt", weights_only=True))
    elif kind == "npz": z = np.load(f"{O}/{name}.npz"); m.load_state_dict({k: torch.tensor(z[k].astype(np.float32)) for k in z.files})
    else: dq, _ = dequant_file(f"{O}/{name}.q{j['bits']}.bin", f"{O}/{name}.q{j['bits']}.json"); m.load_state_dict({k: torch.tensor(v) for k, v in dq.items()})
    return m, j
res = {"real_heldout": {st: audit(real_bars([a for a in val if a["style"] == st], N)) for st in ("jazzhop", "steady", "flow")}}
for spec in sys.argv[2:]:
    m, j = load(spec); r = dict(params=j["params"], heldout=evaluate(m, EV)); torch.manual_seed(0)
    for st in ("jazzhop", "steady", "flow"):
        x = sample(m, [st] * N, ["major", "minor"] * (N // 2)); r[st] = audit([seq_bars(s) for s in x])
    res[spec] = r; print(spec, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["heldout"].items()}), flush=True)
json.dump(res, open(f"xfer/audit_{os.environ.get('TAG', 'x')}.json", "w"), indent=1); print("audit done")
