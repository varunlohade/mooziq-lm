# Teacher data: ask ACE-Step 1.5 (MIT, open model) for calm focus lo-fi in KNOWN keys and tempos.
# Known key + bpm lets us transcribe onto a beat grid and transpose to C without guessing.
import os, sys, time, json, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); DST = os.path.join(HERE, "data", "ace"); os.makedirs(DST, exist_ok=True)
ROOT = os.path.expanduser("~/ACE-Step-1.5"); sys.path.insert(0, ROOT); os.chdir(ROOT)
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
part = int(sys.argv[1])
F = "lyrics-free background music for deep work and studying, calm lofi hip hop, "
STY = [
  "soft Rhodes electric piano playing jazzy seventh and ninth chords, simple sparse piano melody, warm sub bass, dusty soft boom bap drums, vinyl crackle, instrumental",
  "mellow felt piano chords, lush major seventh chords, gentle melody, round bass, brushed snare, swung hats, tape hiss, instrumental",
  "clean jazz guitar chords, minor ninth chords, slow simple melody on electric piano, upright bass, soft kick and rim, instrumental",
  "warm Rhodes chords looping a four bar progression, soft lead melody, deep bass, lazy swung drums, rain ambience, instrumental",
  "jazzy lofi beat, smooth electric piano comping minor seventh chords, simple vibraphone melody, mellow bass, soft dusty drums, instrumental",
  "chillhop, warm piano with gentle seventh chord progression, soft flute-like melody, muted bass, relaxed drums with rimshot, vinyl noise, instrumental",
]
KEYS = ["C major", "F major", "D minor", "A minor", "G major", "E minor", "Eb major", "C minor", "Bb major"]
jobs = []
N0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
for i in range(N0, N0 + 18):
    jobs.append(dict(i=i, bpm=[70, 74, 78, 72, 68, 76][i % 6], key=KEYS[i % len(KEYS)], cap=F + STY[i % len(STY)]))
jobs = jobs[part::2]
dit = AceStepHandler(); dit.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="auto", offload_to_cpu=False)
llm = LLMHandler(); llm.initialize(checkpoint_dir=os.path.join(ROOT, "checkpoints"), lm_model_path="acestep-5Hz-lm-0.6B", backend="mlx", device="auto", offload_to_cpu=False, dtype=None)
for j in jobs:
    p = GenerationParams(task_type="text2music", thinking=True, caption=j["cap"], lyrics="[Instrumental]", bpm=j["bpm"], keyscale=j["key"],
                         timesignature="4", vocal_language="en", duration=90, inference_steps=8, guidance_scale=1.0, seed=1000 + j["i"])
    t0 = time.time(); r = generate_music(dit, llm, params=p, config=GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=False, seeds=[1000 + j["i"]]), save_dir=os.path.join(HERE, "tmp"))
    if r.success and r.audios:
        name = f"ace{j['i']:02d}"; shutil.copy(r.audios[0]["path"], os.path.join(DST, name + ".wav"))
        json.dump(j, open(os.path.join(DST, name + ".json"), "w"))
    print(j["i"], r.success, round(time.time() - t0, 1), "s", flush=True)
