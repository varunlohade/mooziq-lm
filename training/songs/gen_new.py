# ACE-Step 1.5 (MIT) full instrumental songs, 150-240 s. Keeps every LM "thinking" output we can find.
# Usage: gen_song.py WORKER NWORKERS DEADLINE_EPOCH   (env B=batch size, NB=number of batches)
import os, sys, time, json, shutil, random
ROOT = "ACE-Step-1.5"; sys.path.insert(0, ROOT); os.chdir(ROOT)
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
W, NW, DEAD = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
B = int(os.environ.get("B", "4")); NB = int(os.environ.get("NB", "75"))
DST = "songs"; TMP = f"tmp{W}"; os.makedirs(DST, exist_ok=True); os.makedirs(TMP, exist_ok=True)
SONG = "calm gentle background music for 24/7 listening, a complete instrumental song with clear sections: an intro, a main groove, a fuller hook section, a stripped-down breakdown, a build back up, a final hook and an outro; instruments enter and leave between sections; no vocals, "
CAPS = {
 "rainy_cafe": ["calm background music, soft jazz piano trio, brushed drums, upright bass, gentle rain and quiet cafe ambience feel, warm and unhurried", "rainy day cafe jazz, mellow piano chords and melody, soft brushes on snare, walking upright bass, rain on the window feel, cozy"],
 "sleep": ["deep ambient sleep music, no drums, no percussion, slow warm pads, very sparse soft piano notes, long sustained tones, soothing", "slow drifting ambient for sleep, warm analog pads, gentle low drone, rare soft bell notes, no beat, very calm"],
 "bossa": ["bossa nova lofi, nylon string guitar comping, light brushes and shaker, soft bass, gentle Rhodes melody, relaxed and warm", "calm bossa nova, nylon guitar chords, soft flute melody, brushed drums and shaker, round bass, sunny and gentle"],
 "study_piano": ["solo calm piano, minimal and gentle, soft felt piano, simple melody over warm chords, no drums, peaceful study music", "minimal solo piano for studying, slow soft melody, sparse chords, room reverb, no other instruments, calm"],
 "chillhop": ["bouncy warm chillhop beat, round bass, swung drums, Rhodes chords, playful soft guitar melody, warm and relaxed", "warm lofi hip hop with a bouncy groove, round deep bass, crisp gentle drums, electric piano and vibraphone melody, cozy"],
 "nature": ["soft keys and warm pads, no drums, gentle nature feel like birdsong and a quiet forest, slow simple melody, calm and airy", "peaceful ambient with soft piano and pads, no percussion, flowing stream and forest feel, slow and soothing"],
}
BPM = {"rainy_cafe": [70, 72, 75, 78, 80], "sleep": [50, 52, 55, 58, 60], "bossa": [80, 82, 85, 88, 90], "study_piano": [60, 64, 66, 68, 72], "chillhop": [85, 88, 90, 92, 95], "nature": [55, 60, 64, 66, 70]}
MAJ = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
NS = list(CAPS); rng = random.Random(99); jobs = []
NB0 = int(os.environ.get("NB0", "0"))
for b in range(NB0, NB0 + NB):
    st = NS[b % len(NS)]; minor = rng.random() < .5
    jobs.append(dict(b=b, style=st, bpm=rng.choice(BPM[st]), key=MAJ[rng.randrange(12)] + (" minor" if minor else " major"), duration=rng.choice([150, 165, 180]),
                     cap=SONG + CAPS[st][(b // len(NS)) % 2], seeds=[60000 + b * 10 + k for k in range(B)]))
jobs = jobs[W::NW]
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
    if os.path.exists(f"{DST}/n{j['b']:03d}_0.json"): continue
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
            name = f"n{j['b']:03d}_{k}"; shutil.move(a["path"], f"{DST}/{name}.wav.part"); os.rename(f"{DST}/{name}.wav.part", f"{DST}/{name}.wav")
            json.dump(dict(j, seed=j["seeds"][k], name=name, batch_seconds=round(dt, 1), ace_audio_meta=jsonable({kk: vv for kk, vv in a.items() if kk != "path"}), ace_result=lm),
                      open(f"{DST}/{name}.json", "w"))
            n += 1
    print(f"batch {j['b']} {j['style']} {j['duration']}s {j['key']} {j['bpm']} ok={r.success} n={n} {dt:.1f}s {dt / max(1, n):.1f}s/song", flush=True)
print("worker done", flush=True)
