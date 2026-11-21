# Is the heavy vocal-stem energy in s024_1 singing or an instrument? Regenerated with the same seed (gen_song ONLY=24),
# separated with Demucs, then Whisper (MIT) transcribes the vocal stem. Real lyrics -> drop the song from the data.
import os, time, json, numpy as np, soundfile as sf, torch, torchaudio
while not os.path.exists("vcheck/s024_1.json"): time.sleep(5)
from demucs.pretrained import get_model
from demucs.apply import apply_model
torch.set_num_threads(4); m = get_model("htdemucs").cuda().eval()
x, sr = sf.read("vcheck/s024_1.wav", dtype="float32", always_2d=True); x = torch.tensor(x.T)
x = torchaudio.functional.resample(x, sr, m.samplerate); ref = x.mean(0); mu, sd = ref.mean(), ref.std() + 1e-8
with torch.no_grad(): s = apply_model(m, ((x - mu) / sd)[None].cuda(), split=True, overlap=.25, progress=False, device="cuda")[0] * sd + mu
st = {n: torchaudio.functional.resample(s[i].mean(0).cpu(), m.samplerate, 16000).numpy() for i, n in enumerate(m.sources)}
rms = {k: float(np.sqrt((v ** 2).mean())) for k, v in st.items()}
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0, chunk_length_s=30)
out = {}
for k in ("vocals", "other"):
    r = asr({"raw": st[k], "sampling_rate": 16000}, return_timestamps=True, generate_kwargs={"language": "en", "task": "transcribe"})
    out[k] = r["text"][:1500]
words = [w for w in out["vocals"].split() if w.isalpha()]
uniq = len(set(w.lower() for w in words))
verdict = "vocals" if len(words) >= 30 and uniq >= 15 else "instrument"
res = dict(song="s024_1", regenerated_ratio_vocals_over_other=rms["vocals"] / (rms["other"] + 1e-9), original_ratio=0.985, rms=rms, whisper_vocals=out["vocals"], whisper_other=out["other"],
           n_words=len(words), n_unique=uniq, rule="vocals if Whisper finds >= 30 words with >= 15 distinct", verdict=verdict)
json.dump(res, open("xfer/vcheck.json", "w"), indent=1)
if verdict == "vocals": open("xfer/drop.txt", "w").write("s024_1\n")
print(json.dumps({k: v for k, v in res.items() if k not in ("whisper_vocals", "whisper_other")}))
