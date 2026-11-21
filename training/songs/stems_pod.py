# Demucs (MIT, htdemucs) stem separation on GPU. Watches songs, writes stems/<name>/{drums,bass,other,vocals}.wav (22.05 kHz mono)
import os, sys, glob, time, json, numpy as np, soundfile as sf, torch, torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
m = get_model("htdemucs").cuda().eval(); SR = m.samplerate
os.makedirs("stems", exist_ok=True)
while True:
    todo = [f for f in sorted(glob.glob("songs/*.wav")) if os.path.exists(f[:-4] + ".json") and not os.path.exists(f"stems/{os.path.basename(f)[:-4]}/done")]
    if not todo:
        if os.path.exists("GEN_DONE"): break
        time.sleep(5); continue
    for f in todo:
        name = os.path.basename(f)[:-4]; t0 = time.time(); od = f"stems/{name}"; os.makedirs(od, exist_ok=True)
        x, sr = sf.read(f, dtype="float32", always_2d=True); x = torch.tensor(x.T)
        if x.shape[0] == 1: x = x.repeat(2, 1)
        if sr != SR: x = torchaudio.functional.resample(x, sr, SR)
        ref = x.mean(0); mu, sd = ref.mean(), ref.std() + 1e-8
        with torch.no_grad(): s = apply_model(m, ((x - mu) / sd)[None].cuda(), split=True, overlap=.25, progress=False)[0] * sd + mu
        s = torchaudio.functional.resample(s.mean(1).cpu(), SR, 22050).numpy()
        rms = {}
        for i, nm in enumerate(m.sources):
            sf.write(f"{od}/{nm}.wav", s[i], 22050, subtype="PCM_16"); rms[nm] = float(np.sqrt((s[i] ** 2).mean()))
        json.dump(dict(rms=rms, secs=round(time.time() - t0, 1)), open(f"{od}/done", "w"))
        print(name, json.dumps(rms), round(time.time() - t0, 1), "s", flush=True)
print("stems done", flush=True)
