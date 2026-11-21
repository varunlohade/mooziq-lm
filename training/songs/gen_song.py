# ACE-Step 1.5 (MIT) full instrumental songs, 150-240 s. Keeps every LM "thinking" output we can find.
# Usage: gen_song.py WORKER NWORKERS DEADLINE_EPOCH   (env B=batch size, NB=number of batches)
import os, sys, time, json, shutil, random
ROOT = "ACE-Step-1.5"; sys.path.insert(0, ROOT); os.chdir(ROOT)
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
W, NW, DEAD = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
B = int(os.environ.get("B", "4")); NB = int(os.environ.get("NB", "75"))
DST = os.environ.get("DST", "songs"); TMP = f"tmp{W}"; os.makedirs(DST, exist_ok=True); os.makedirs(TMP, exist_ok=True)
SONG = "a complete instrumental song with clear sections: an intro, a main groove, a fuller hook section, a stripped-down breakdown, a build back up, a final hook and an outro; instruments enter and leave between sections; no vocals, "
JAZZ = [
 "mellow Japanese jazz hip hop, swung boom bap drums, Rhodes chords with major seventh and minor ninth voicings, ii-V-I movement, alto saxophone melody, upright bass, vinyl crackle",
 "off-grid swung jazz hop, lazy swung drums, dusty sampled piano chords, flute melody, round bass, vinyl noise",
 "late night jazz hop, warm electric piano, muted trumpet phrases, walking bass, swung hi hats, crisp snare, record crackle",
 "jazzy boom bap instrumental, vibraphone melody over Rhodes chords, tenor saxophone answers, deep bass, dusty swung drums",
]
STEADY = [
 "calm lofi hip hop, soft Rhodes seventh chords, simple piano melody, warm sub bass, gentle dusty drums, vinyl crackle",
 "mellow lofi beat, felt piano chords, gentle guitar melody, soft kick and rim, swung hats, tape hiss",
 "chillhop, clean jazz guitar chords, electric piano lead, muted bass, relaxed steady drums, rain ambience",
]
FLOW = [
 "ambient piano music, slow sparse piano melody over a warm drone, soft pads, sections without percussion and a gentle soft pulse in the middle, calm",
 "ambient lofi, slow Rhodes chords and simple melody over a warm analog drone, soft bass enters later, airy, meditative",
 "drumless ambient study music, delicate piano melody, sustained strings, warm low bass notes, slow and spacious",
]
MAJ = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
acc = {"jazzhop": 0, "steady": 0, "flow": 0}; share = {"jazzhop": .55, "steady": .25, "flow": .20}; sty = []
for b in range(NB):
    s = max(share, key=lambda k: share[k] * (b + 1) - acc[k]); acc[s] += 1; sty.append(s)
rng = random.Random(77); jobs = []
for b, s in enumerate(sty):
    if s == "jazzhop": bpm = rng.choice([80, 82, 84, 86, 88, 90, 92]); cap = JAZZ[b % len(JAZZ)]; minor = rng.random() < .6
    elif s == "steady": bpm = rng.choice([70, 72, 74, 76]); cap = STEADY[b % len(STEADY)]; minor = rng.random() < .45
    else: bpm = rng.choice([60, 62, 64, 66]); cap = FLOW[b % len(FLOW)]; minor = rng.random() < .4
    dur = rng.choice([150, 165, 180, 195, 210, 225, 240])
    jobs.append(dict(b=b, style=s, bpm=bpm, key=MAJ[rng.randrange(12)] + (" minor" if minor else " major"), duration=dur, cap=SONG + cap, seeds=[50000 + b * 10 + k for k in range(B)]))
jobs = [j for j in jobs if j["b"] == int(os.environ["ONLY"])] if os.environ.get("ONLY") else jobs[W::NW]
def jsonable(o, depth=0):
    if depth > 4: return repr(o)[:500]
    if isinstance(o, (str, int, float, bool)) or o is None: return o if not isinstance(o, str) else o[:20000]
    if isinstance(o, dict): return {str(k): jsonable(v, depth + 1) for k, v in o.items() if not hasattr(v, "shape") or getattr(v, "ndim", 2) == 0}
    if isinstance(o, (list, tuple)): return [jsonable(v, depth + 1) for v in o[:64]]
    if hasattr(o, "__dict__"): return jsonable(vars(o), depth + 1)
    return repr(o)[:500]
dit = AceStepHandler(); print(dit.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="cuda", offload_to_cpu=False), flush=True)
llm = LLMHandler(); print(llm.initialize(checkpoint_dir=os.path.join(ROOT, "checkpoints"), lm_model_path="acestep-5Hz-lm-1.7B", backend="vllm", device="cuda", offload_to_cpu=False, dtype=None), flush=True)
for j in jobs:
    if time.time() > DEAD: print("deadline", flush=True); break
    if os.path.exists(f"{DST}/s{j['b']:03d}_0.json"): continue
    p = GenerationParams(task_type="text2music", thinking=True, caption=j["cap"], lyrics="[Instrumental]", instrumental=True, bpm=j["bpm"], keyscale=j["key"],
                         timesignature="4", vocal_language="en", duration=j["duration"], inference_steps=8, guidance_scale=1.0, seed=j["seeds"][0])
    t0 = time.time()
    try: r = generate_music(dit, llm, params=p, config=GenerationConfig(batch_size=B, allow_lm_batch=True, audio_format="wav", use_random_seed=False, seeds=j["seeds"]), save_dir=TMP)
    except Exception as e: print("ERR", j["b"], repr(e)[:300], flush=True); continue
    dt = time.time() - t0; n = 0
    lm = {k: jsonable(v) for k, v in vars(r).items() if k != "audios"} if hasattr(r, "__dict__") else {}
    if r.success:
        for k, a in enumerate(r.audios or []):
            if not a.get("path"): continue
            name = f"s{j['b']:03d}_{k}"; shutil.move(a["path"], f"{DST}/{name}.wav.part"); os.rename(f"{DST}/{name}.wav.part", f"{DST}/{name}.wav")
            json.dump(dict(j, seed=j["seeds"][k], name=name, batch_seconds=round(dt, 1), ace_audio_meta=jsonable({kk: vv for kk, vv in a.items() if kk != "path"}), ace_result=lm),
                      open(f"{DST}/{name}.json", "w"))
            n += 1
    print(f"batch {j['b']} {j['style']} {j['duration']}s {j['key']} {j['bpm']} ok={r.success} n={n} {dt:.1f}s {dt / max(1, n):.1f}s/song", flush=True)
print("worker done", flush=True)
