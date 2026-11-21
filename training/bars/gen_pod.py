# ACE-Step 1.5 (MIT) teacher clips on CUDA. Usage: gen_pod.py WORKER NWORKERS DEADLINE_EPOCH
import os, sys, time, json, shutil, random
ROOT = "ACE-Step-1.5"; sys.path.insert(0, ROOT); os.chdir(ROOT)
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
W, NW, DEAD = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
DST = "clips"; TMP = f"tmp{W}"; os.makedirs(DST, exist_ok=True); os.makedirs(TMP, exist_ok=True)
F = "instrumental, no vocals, "
JAZZ = [
 "mellow Japanese jazz hip hop, swung boom bap drums, jazzy Rhodes electric piano chords, major seventh and minor ninth chords, ii-V-I progression, soulful alto saxophone phrases, warm upright bass, vinyl crackle",
 "off-grid swung jazz hop beat, lazy off-grid swung drums, dusty sampled jazz piano chords, minor ninth chords, short flute melody phrases, round bass, vinyl noise",
 "mellow lofi jazz hip hop, mellow piano chords with ii-V-I movement, smooth tenor saxophone melody, crisp boom bap kick and snare, swung hi hats, deep bass, record crackle",
 "jazzy hip hop instrumental, Rhodes comping maj7 and m9 chords, breathy flute lead melody, swung boom bap groove, walking bass, warm tape saturation, vinyl",
 "chill jazz rap beat without vocals, sampled jazz quartet feel, piano voicings with ninths, muted trumpet phrases, head-nodding swung boom bap drums, upright bass, dusty vinyl",
]
STEADY = [
 "calm lofi hip hop beat, soft Rhodes seventh chords, simple sparse piano melody, warm sub bass, gentle dusty drums, vinyl crackle, study music",
 "mellow lofi beat, felt piano chords, lush major seventh chords, gentle melody, soft kick and rim, swung hats, tape hiss, relaxing",
 "chillhop study beat, clean jazz guitar chords, slow simple electric piano melody, muted bass, relaxed steady drums, rain ambience",
 "lofi hip hop, warm Rhodes looping a four bar progression, soft lead melody, deep bass, lazy brushed drums, calm and even",
]
FLOW = [
 "ambient piano, no drums, no percussion, slow sparse felt piano melody over a warm sustained synth drone, gentle chords, calm and spacious, soft reverb",
 "ambient lofi without drums, slow soft piano phrases over a warm analog pad drone, seventh chords, tape warmth, peaceful, no beat",
 "beatless ambient music, slow Rhodes chords and a simple piano melody over a low warm drone, airy, meditative, no drums",
 "drumless ambient study music, delicate piano melody, long sustained string pad drone, warm and slow, no percussion",
]
MAJ = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
B = int(os.environ.get("B", "4"))             # clips per batch (same prompt, B seeds)
NB0 = int(os.environ.get("NB0", "0")); NB = int(os.environ.get("NB", "100"))   # batch id range [NB0, NB)
# style per batch: interleaved so every prefix keeps ~55/25/20
sty = []
acc = {"jazzhop": 0, "steady": 0, "flow": 0}; share = {"jazzhop": .55, "steady": .25, "flow": .20}
for b in range(NB):
    s = max(share, key=lambda k: share[k] * (b + 1) - acc[k]); acc[s] += 1; sty.append(s)
rng = random.Random(7 + NB0)
JAZZ += [
 "late night jazz hop, warm Rhodes ninth chords, ii-V-I turnarounds, mellow trumpet melody, swung boom bap drums with ghost notes, upright bass, crackling vinyl",
 "jazz-inspired instrumental hip hop, piano and flute melody, minor seventh chord progression, steady swung boom bap beat, warm bass, soft vinyl texture",
 "jazzy boom bap instrumental, vibraphone melody over Rhodes maj7 chords, dusty swung drums, round bass, lo-fi tape warmth",
]
STEADY += [
 "soft lofi beat for studying, gentle piano melody over warm seventh chords, simple steady drums, mellow bass, light vinyl hiss",
 "dreamy lofi hip hop, electric piano chords, soft guitar melody, steady relaxed beat, warm bass, cozy and calm",
]
FLOW += [
 "slow ambient piano, no drums, warm cello drone underneath, simple melody, sustained chords, peaceful and slow",
 "ambient electric piano, no percussion, slow melody over a soft warm drone and pad, gentle seventh chords, calm",
]
jobs = []
for b, s in enumerate(sty):
    if b < NB0: continue
    if s == "jazzhop": bpm = rng.choice([80, 82, 84, 85, 86, 88, 90, 92]); cap = JAZZ[b % len(JAZZ)]; minor = rng.random() < .6
    elif s == "steady": bpm = rng.choice([70, 72, 74, 76]); cap = STEADY[b % len(STEADY)]; minor = rng.random() < .45
    else: bpm = rng.choice([60, 62, 64, 66]); cap = FLOW[b % len(FLOW)]; minor = rng.random() < .4
    key = MAJ[rng.randrange(12)] + (" minor" if minor else " major")
    jobs.append(dict(b=b, style=s, bpm=bpm, key=key, cap=F + cap, seeds=[10000 + b * 10 + k for k in range(B)]))
jobs = jobs[W::NW]
dit = AceStepHandler(); print(dit.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="cuda", offload_to_cpu=False), flush=True)
llm = LLMHandler(); print(llm.initialize(checkpoint_dir=os.path.join(ROOT, "checkpoints"), lm_model_path="acestep-5Hz-lm-1.7B", backend=os.environ.get("LMB", "vllm"), device="cuda", offload_to_cpu=False, dtype=None), flush=True)
for j in jobs:
    if time.time() > DEAD: print("deadline", flush=True); break
    if os.path.exists(f"{DST}/c{j['b']:03d}_0.json"): continue
    p = GenerationParams(task_type="text2music", thinking=True, caption=j["cap"], lyrics="[Instrumental]", instrumental=True, bpm=j["bpm"], keyscale=j["key"],
                         timesignature="4", vocal_language="en", duration=90, inference_steps=8, guidance_scale=1.0, seed=j["seeds"][0]) \
        if "instrumental" in GenerationParams.__dataclass_fields__ else \
        GenerationParams(task_type="text2music", thinking=True, caption=j["cap"], lyrics="[Instrumental]", bpm=j["bpm"], keyscale=j["key"],
                         timesignature="4", vocal_language="en", duration=90, inference_steps=8, guidance_scale=1.0, seed=j["seeds"][0])
    t0 = time.time()
    try:
        r = generate_music(dit, llm, params=p, config=GenerationConfig(batch_size=B, allow_lm_batch=True, audio_format="wav", use_random_seed=False, seeds=j["seeds"]), save_dir=TMP)
    except Exception as e:
        print("ERR", j["b"], repr(e)[:300], flush=True); continue
    dt = time.time() - t0; n = 0
    if r.success:
        for k, a in enumerate(r.audios or []):
            if not a.get("path"): continue
            name = f"c{j['b']:03d}_{k}"; shutil.move(a["path"], f"{DST}/{name}.wav.part"); os.rename(f"{DST}/{name}.wav.part", f"{DST}/{name}.wav")
            json.dump(dict(j, seed=j["seeds"][k], name=name, batch_seconds=round(dt, 1)), open(f"{DST}/{name}.json", "w")); n += 1
    print(f"batch {j['b']} {j['style']} {j['key']} {j['bpm']} ok={r.success} n={n} {dt:.1f}s {dt/max(1,n):.1f}s/clip", ("" if r.success else str(r.error)[:300]), flush=True)
print("worker done", flush=True)
