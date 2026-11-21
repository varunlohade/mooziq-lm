import json, glob, os, subprocess, collections
# 3 MP3 examples per style for the owner (48 kbps, full length). Skips styles that already have 3 in the Mac-side index (passed in data/audio_have.json).
O = 'xfer/audio'; os.makedirs(O, exist_ok=True)
have = collections.Counter(json.load(open('data/audio_have.json'))) if os.path.exists('data/audio_have.json') else collections.Counter()
idx = json.load(open(f'{O}/index.json')) if os.path.exists(f'{O}/index.json') else []
for e in idx: have[e['style']] += 1
used = {e['song'] for e in idx}; by = collections.defaultdict(list)
for f in sorted(glob.glob('sym/*.json')):
    d = json.load(open(f))
    if 'error' not in d: by[d['style']].append(d)
for st, L in by.items():
    L = sorted(L, key=lambda d: (-len(d['sections']), d['name'])); seen_b = set()
    for d in L:
        if have[st] >= 3: break
        b = d['name'].split('_')[0]
        if d['name'] in used or b in seen_b: continue
        seen_b.add(b); n = have[st] + 1; out = f'{O}/{st}-{n}.mp3'
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', f"songs/{d['name']}.wav", '-b:a', '48k', out], check=True)
        m = json.load(open(f"songs/{d['name']}.json"))
        idx.append(dict(file=os.path.basename(out), style=st, song=d['name'], prompt=m['cap'], bpm=m['bpm'], key=m['key'], duration_s=m['duration'], seed=m['seed'], detected_tempo=d['tempo'],
                        sections=[dict(type=s['type'], start_bar=s['start'], bars=s['len'], energy=s['energy'], active=''.join(c for c, x in zip('DBKL', s['active']) if x)) for s in d['sections']], bytes=os.path.getsize(out)))
        have[st] += 1; used.add(d['name'])
json.dump(idx, open(f'{O}/index.json', 'w'), indent=1); print(dict(have), round(sum(e['bytes'] for e in idx) / 1e6, 1), 'MB new')
