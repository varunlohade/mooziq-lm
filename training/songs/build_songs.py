# merge sym/*.json -> xfer/songs/songs_dataset.json + stats (pulled to the Mac in batches)
import json, glob, os, collections, time
ok, err = [], []
for f in sorted(glob.glob("sym/*.json")):
    d = json.load(open(f)); (err if "error" in d else ok).append(d)
drop = set(open("xfer/drop.txt").read().split()) if os.path.exists("xfer/drop.txt") else set()
new_only = [d for d in ok]
if os.path.exists("data/old_songs.json"): ok = [d for d in json.load(open("data/old_songs.json")) if d["name"] not in drop] + [d for d in ok if not d["name"].startswith("s")]
os.makedirs("xfer/songs", exist_ok=True); json.dump(ok, open("xfer/songs/songs_dataset.json", "w")); json.dump(new_only, open("xfer/songs/new_songs.json", "w"))
c = collections.Counter(d["style"] for d in ok); b = collections.Counter(); s = collections.Counter(); g = collections.Counter(d["grid"] for d in ok)
for d in ok: b[d["style"]] += len(d["bars"]); s[d["style"]] += len(d["sections"])
st = dict(songs=dict(c), bars=dict(b), sections=dict(s), grid=dict(g), errors=len(err), err_examples=[e["error"][:120] for e in err[:3]],
          max_vocals_ratio=max([d["vocals_rms"] / (d["other_rms"] + 1e-9) for d in ok] or [0]), t=time.strftime("%H:%M:%S"))
json.dump(st, open("xfer/songs/stats.json", "w")); print(json.dumps(st))
