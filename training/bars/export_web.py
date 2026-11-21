# ckpt/NAME.npz (fp16) + ckpt/NAME.json -> web/NAME/model.bin + model.json, same layout the shipped kernel reads.
# Then a numpy KV-cache forward (the same math as engine/neural/transformer.js) checks it against a full torch-free reference.
import sys, os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); NAME = sys.argv[1]
z = np.load(f"{HERE}/ckpt/{NAME}.npz"); meta = json.load(open(f"{HERE}/ckpt/{NAME}.json")); tk = meta["tokens"]
os.makedirs(f"{HERE}/web/{NAME}", exist_ok=True); off = 0; idx = {}; buf = []
for k in z.files:
    a = z[k].astype(np.float16); idx[k] = dict(shape=list(a.shape), off=off); off += a.size; buf.append(a.ravel())
np.concatenate(buf).tofile(f"{HERE}/web/{NAME}/model.bin")
out = dict(tensors=idx, D=meta["D"], L=meta["L"], H=meta["H"], V=meta["V"], T=meta["T"], **{k: tk[k] for k in ("PAD", "BOS", "LOFI", "MAJ", "MIN", "BAR", "CH0", "M0", "NQ", "LO", "HI")},
           FOLK=tk["FOLK"], STEADY=tk["STEADY"], JAZZHOP=tk["JAZZHOP"], FLOW=tk["FLOW"])
json.dump(out, open(f"{HERE}/web/{NAME}/model.json", "w"))
print(NAME, "params", meta["params"], "model.bin bytes", os.path.getsize(f"{HERE}/web/{NAME}/model.bin"))
