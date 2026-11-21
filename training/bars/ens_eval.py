# Held-out loss of a probability-averaged ensemble of trainer.py models. Usage: ens_eval.py NAME1,NAME2,...
import sys, json, torch, torch.nn.functional as F
sys.path.insert(0, "code"); from lm import *
O = "xfer/ckpt"; names = sys.argv[1].split(",")
tr, val, pd_tr, pd_val = load_data(); EV = make_eval(val, pd_val)
te = load_models(sys.argv[1])
r = evaluate(te, EV); print(sys.argv[1], json.dumps({k: round(v, 4) for k, v in r.items()}))
