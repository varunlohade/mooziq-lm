"""Fine-tune the handoff's planner and composer on a repaired corpus (local CPU).
Arguments: repaired corpus, original songmodels.py directory, input weights directory, output directory.
"""
import argparse
import copy
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from repair_labels import repair


def tune(name, songs, source, destination, trainer):
    random.seed(17)
    torch.manual_seed(17)
    meta = json.loads((source / f"{name}.json").read_text())
    model = trainer.GPT(*(meta[k] for k in ("V", "T", "SLOT", "D", "L", "H")), drop=.1)
    data = np.fromfile(source / f"{name}.bin", dtype=np.float16)
    state = {}
    for key, tensor in meta["tensors"].items():
        start, size = tensor["off"], int(np.prod(tensor["shape"]))
        state[key] = torch.from_numpy(data[start:start+size].reshape(tensor["shape"]).astype(np.float32))
    model.load_state_dict(state)
    train, valid = trainer.split(songs)
    planner = meta["kind"] == "planner"

    def rows(items):
        sequences = []
        for song in items:
            if planner:
                seq = trainer.plan_encode(song)
                sequences.append(seq + [0] * (meta["T"] - len(seq)))
            else:
                bars = trainer.song_bars(song)
                prefix = [meta["BOS"], meta["STY0"] + meta["STYLES"].index(song["style"]),
                          meta["MAJ"] if song["mode"] == "major" else meta["MIN"]]
                for start in range(0, max(1, len(bars)-trainer.SBARS+1), trainer.SBARS//2):
                    seq = prefix + [t for bar in bars[start:start+trainer.SBARS] for t in bar]
                    sequences.append(seq + [0] * (meta["T"] - len(seq)))
        return torch.tensor(sequences)

    def loss(batch):
        target = batch[:, 1:]
        weights = (target != 0).float() if planner else trainer.song_weights(target)
        logits = model(batch[:, :-1]).transpose(1, 2)
        return (F.cross_entropy(logits, target, reduction="none") * weights).sum(), weights.sum()

    train, valid = rows(train), rows(valid)

    @torch.no_grad()
    def evaluate():
        model.eval()
        totals = [loss(batch) for batch in valid.split(16)]
        return sum(v.item() for v, _ in totals) / sum(n.item() for _, n in totals)

    baseline = best = evaluate()
    winner = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=.0004 if planner else .00012)
    for epoch in range(6):
        model.train()
        for batch in torch.randperm(len(train))[:800].split(16):
            value, count = loss(train[batch])
            opt.zero_grad()
            (value/count).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
        score = evaluate()
        print(name, epoch+1, 'validation', round(score, 5), 'baseline', round(baseline, 5), flush=True)
        if score < best:
            best, winner = score, copy.deepcopy(model.state_dict())
    if best >= baseline:
        raise RuntimeError(f"{name} did not improve")
    model.load_state_dict(winner)
    del meta["tensors"]
    trainer.export(model, meta, str(destination / name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("corpus", "trainer_dir", "weights", "output"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("songmodels", args.trainer_dir / "songmodels.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    torch.set_num_threads(4)
    songs = [song for song in json.loads(args.corpus.read_text()) if song.get("sections")]
    for song in songs:
        repair(song)
    args.output.mkdir(exist_ok=True, parents=True)
    for name in ("planner", "songlm_s"):
        tune(name, songs, args.weights, args.output, trainer)
