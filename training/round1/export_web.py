# model.npz (fp16) -> web/model.bin (raw fp16, concatenated) + web/model.json (names, shapes, offsets, vocab constants)
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import *
z = np.load(os.path.join(HERE, "model.npz")); off = 0; idx = {}; buf = []
for k in z.files:
    a = z[k].astype(np.float16); idx[k] = dict(shape=list(a.shape), off=off); off += a.size; buf.append(a.ravel())
np.concatenate(buf).tofile(os.path.join(HERE, "web/model.bin"))
json.dump(dict(tensors=idx, D=D, L=L, H=H, V=V, T=T, PAD=PAD, BOS=BOS, LOFI=LOFI, MAJ=MAJ, MIN=MIN, BAR=BAR, CH0=CH0, M0=M0, NQ=NQ, LO=LO, HI=HI),
          open(os.path.join(HERE, "web/model.json"), "w"))
print("web/model.bin bytes", os.path.getsize(os.path.join(HERE, "web/model.bin")))
nm = NPModel(os.path.join(HERE, "model.npz")); nm.reset()
for t in (BOS, LOFI, MAJ): nm.step(t)
json.dump([float(v) for v in nm.step(BAR)], open(os.path.join(HERE, "web/parity.json"), "w"))
