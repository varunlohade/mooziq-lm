# merge out/bars/*.json -> data/ace2_bars.json (+ old round-1 lofi clips as steady) and a pull bundle in xfer/data/
import json, glob, os, collections, tarfile, time
items, errs = [], 0
for f in sorted(glob.glob("out/bars/*.json")):
    d = json.load(open(f))
    if "error" in d or not d.get("bars"): errs += 1; continue
    items.append(d)
old = json.load(open("data/old_ace_bars.json")) if os.path.exists("data/old_ace_bars.json") else []
for o in old: o["style"] = "steady"; o["name"] = "r1_" + o.get("name", "?")
os.makedirs("data", exist_ok=True); os.makedirs("xfer/data", exist_ok=True)
json.dump(items + old, open("data/ace2_bars.json", "w"))
c = collections.Counter(); b = collections.Counter(); g = collections.Counter()
for d in items: c[d["style"]] += 1; b[d["style"]] += len(d["bars"]); g[d["style"] + ":" + d.get("grid", "?")] += 1
stat = dict(clips=dict(c), bars=dict(b), grid=dict(g), errors=errs, old_r1_clips=len(old), old_r1_bars=sum(len(o["bars"]) for o in old), t=time.strftime("%H:%M:%S"))
json.dump(stat, open("xfer/data/stats.json", "w"))
json.dump(items, open("xfer/data/ace2_bars.json", "w"))
print(json.dumps(stat))
