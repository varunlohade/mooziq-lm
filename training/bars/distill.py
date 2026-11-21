# Distil a teacher into a browser student: soft-target KL (temperature TAU) + hard-label CE, on real windows plus a pool of
# teacher-sampled sequences; then quantization-aware fine-tuning (int8 or int4, per-group fp16 scales) and export.
# Usage: distill.py NAME D L H BITS GROUP MAX_MINUTES TEACHER  (env ALPHA TAU LR DROP SYN_FRAC QAT_MIN)
import sys, os, json, math, time, random, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "code"); from lm import *
NAME, D, L, H, BITS, GROUP, MAXMIN, TEACH = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7]), sys.argv[8]
E = lambda k, d: float(os.environ.get(k, d))
ALPHA, TAU, LR, DROP, SYNF, QATMIN, MELW = E("ALPHA", .7), E("TAU", 2.), E("LR", 6e-4), E("DROP", .1), E("SYN_FRAC", .5), E("QAT_MIN", 8), E("MELW", .5)
torch.manual_seed(0); random.seed(0)
O = "xfer/ckpt"; log = open(f"{O}/{NAME}.log", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
tr, val, pd_tr, pd_val = load_data(); EV = make_eval(val, pd_val); bat = Batcher(tr, pd_tr, MELW, 0.)
te = load_models(TEACH); TP = sum(p.numel() for p in te.parameters())
syn = torch.load(f"syn_{os.environ.get('POOL', TEACH)}.pt")   # [N, T] teacher samples
m = GPT(D, L, H, DROP).cuda(); NP = nparams(m)
P("student", NAME, "params", NP, "D L H", D, L, H, "bits", BITS, "group", GROUP, "teacher", TEACH, TP, "syn", tuple(syn.shape), "alpha", ALPHA, "tau", TAU, "lr", LR, "drop", DROP, "syn_frac", SYNF)
def get_batch(B):
    nb = int(B * SYNF); x, y, w = bat(B - nb)
    s = syn[torch.randint(0, len(syn), (nb,))]; xs, ys = s[:, :-1], s[:, 1:]
    return torch.cat([x, xs]).cuda(), torch.cat([y, ys]).cuda(), torch.cat([w, torch.ones(ys.shape)]).cuda(), B - nb
def loss_fn(x, y, w, nreal):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): tl = te(x).float()
    with torch.autocast("cuda", dtype=torch.bfloat16): sl = m(x).float()
    msk = (y != PAD).float()
    kl = (F.kl_div(F.log_softmax(sl / TAU, -1), F.log_softmax(tl / TAU, -1), log_target=True, reduction="none").sum(-1) * msk).sum() / msk.sum() * TAU ** 2
    hard = ce(sl[:nreal], y[:nreal], w[:nreal])
    return ALPHA * kl + (1 - ALPHA) * hard, kl, hard
def run(phase, minutes, lr0, patience, save):
    dec = [p for n, p in m.named_parameters() if p.dim() == 2]; nodec = [p for n, p in m.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=dec, weight_decay=.05), dict(params=nodec, weight_decay=0)], lr=lr0, betas=(.9, .98))
    t0, step, best, bad, hist = time.time(), 0, 9e9, 0, []
    while time.time() - t0 < minutes * 60:
        lr = lr0 * min(1, (step + 1) / 200) * max(.1, .5 * (1 + math.cos(math.pi * min(1, step / 20000))))
        for g in opt.param_groups: g["lr"] = lr
        loss, kl, hard = loss_fn(*get_batch(128))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1); opt.step(); step += 1
        if step % 200 == 0:
            r = evaluate(m, EV); r.update(phase=phase, step=step, kl=round(kl.item(), 4), hard=round(hard.item(), 4), secs=round(time.time() - t0)); hist.append(r)
            P(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
            crit = r["ace"] + .25 * r["pd"]
            if crit < best - 1e-3: best = crit; bad = 0; save()
            else:
                bad += 1
                if bad >= patience: P("early stop", phase); break
    return hist
hist_fp = run("fp", MAXMIN, LR, 12, lambda: torch.save(m.state_dict(), f"{O}/{NAME}.fp.pt"))
m.load_state_dict(torch.load(f"{O}/{NAME}.fp.pt")); best_fp = evaluate(m, EV)
np.savez(f"{O}/{NAME}.fp16.npz", **{k: v.cpu().numpy().astype(np.float16) for k, v in m.state_dict().items()})
if BITS == 16:
    json.dump(dict(name=NAME, params=NP, D=D, L=L, H=H, V=V, T=T, bits=16, teacher=TEACH, alpha=ALPHA, tau=TAU, best=best_fp, best_fp=best_fp, hist_fp=hist_fp, tokens=TOKENS), open(f"{O}/{NAME}.json", "w"), indent=1)
    np.savez(f"{O}/{NAME}.npz", **{k: v.cpu().numpy().astype(np.float16) for k, v in m.state_dict().items()})
    P("done", NAME, "fp16", json.dumps(best_fp)); sys.exit(0)
add_fq(m, BITS, GROUP); ptq = evaluate(m, EV); P("PTQ (no QAT)", json.dumps(ptq))
hist_q = run("qat", QATMIN, LR / 4, 6, lambda: torch.save(raw_state(m), f"{O}/{NAME}.qat_raw.pt"))
# reload best QAT raw weights into a plain model, export quantized, verify by dequantizing the file
plain = GPT(D, L, H, 0.).cuda(); plain.load_state_dict(torch.load(f"{O}/{NAME}.qat_raw.pt"))
sd = {k: v.float().cpu() for k, v in plain.state_dict().items()}
meta = dict(D=D, L=L, H=H, V=V, T=T, **TOKENS, params=NP, bits=BITS, group=GROUP, teacher=TEACH)
nbytes = export_quant(sd, BITS, GROUP, f"{O}/{NAME}.q{BITS}.bin", f"{O}/{NAME}.q{BITS}.json", meta)
dq, _ = dequant_file(f"{O}/{NAME}.q{BITS}.bin", f"{O}/{NAME}.q{BITS}.json")
chk = GPT(D, L, H, 0.).cuda(); chk.load_state_dict({k: torch.tensor(v) for k, v in dq.items()}); best_q = evaluate(chk, EV)
json.dump(dict(name=NAME, params=NP, D=D, L=L, H=H, V=V, T=T, bits=BITS, group=GROUP, bytes=nbytes, teacher=TEACH, alpha=ALPHA, tau=TAU, best_fp=best_fp, ptq=ptq,
               best_quant_file=best_q, hist_fp=hist_fp, hist_qat=hist_q, tokens=TOKENS), open(f"{O}/{NAME}.json", "w"), indent=1)
P("done", NAME, "bytes", nbytes, "fp", json.dumps(best_fp), "quant(file)", json.dumps(best_q))
