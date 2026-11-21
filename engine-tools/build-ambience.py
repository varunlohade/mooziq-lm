# Builds seamless ambience loops for public/samples/ambience/ from openly licensed field
# recordings (see CREDITS.md): a stretch of each file, with its tail crossfaded into its
# head so it loops without a seam, level-matched, then AAC-encoded with ffmpeg.
# Usage: uv run --with numpy python -I engine-tools/build-ambience.py <source dir> <out dir>
import sys, os, subprocess, numpy as np
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
SR = 44100
# name: (file, start seconds, loop seconds, channels)
LOOPS = {
    "rain": ("Rain_against_the_window.ogg", 10, 40, 1),
    "cafe": ("Restaurant_ambience.ogg", 8, 40, 2),
    "forest": ("Waidachswald_Oberschefflenz_Abend_20250616_2005.ogg", 30, 45, 2),
    "water": ("363120_fractalstudios_water_trickle.wav", 2, 28, 2),
}
def load(path, start, seconds, ch):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(seconds), "-i", path, "-ac", str(ch), "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, ch).copy()
for name, (file, start, seconds, ch) in LOOPS.items():
    fade = 2.0
    x = load(os.path.join(src, file), start, seconds + fade, ch)
    n, f = int(seconds * SR), int(fade * SR)
    loop, tail = x[:n].copy(), x[n:n + f]
    ramp = np.linspace(0, 1, f)[:, None]
    loop[:f] = loop[:f] * np.sin(ramp * np.pi / 2) + tail * np.cos(ramp * np.pi / 2)
    # Gentle high-pass (remove rumble) and level to -30 dBFS RMS; the engine sets the final level.
    loop -= loop.mean(0)
    rms = np.sqrt(np.mean(loop ** 2))
    loop *= 10 ** (-30 / 20) / max(rms, 1e-9)
    wav = os.path.join(out, name + ".f32")
    loop.astype(np.float32).tofile(wav)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", str(ch), "-i", wav, "-af", "highpass=f=40",
                    "-c:a", "aac", "-b:a", "64k" if ch == 1 else "80k", os.path.join(out, name + ".m4a")], check=True)
    os.remove(wav)
    print(name, seconds, "s", ch, "ch", os.path.getsize(os.path.join(out, name + ".m4a")), "bytes")
