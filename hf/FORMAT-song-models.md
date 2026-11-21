# Full-song models: planner + composer (9 styles)

Each model has two files: `models/NAME.bin` (raw fp16, row-major) and `models/NAME.json`. The JSON holds tensor offsets plus every vocab constant listed below.
- The network is the shipped one, with one difference: the slot embedding repeats every `SLOT` tokens, not every 10.
- The kernel change is one line: `slot = max(0, p - 3) % meta.SLOT`. See `transformer_slot.js`.
- `songs/parity_song.mjs` checks a model against the numpy reference (max logit gap about 5e-7).

**Styles** (`STYLES`, token = `STY0 + index`):
0 jazzhop, 1 steady, 2 flow, 3 rainy_cafe, 4 sleep, 5 bossa, 6 study_piano, 7 chillhop, 8 nature.

**Tracks** (`TRACKS`): drums, bass, keys, lead. In an active set, bit 0 is drums, bit 1 bass, bit 2 keys and bit 3 lead.

## Planner (`planner`, SLOT = 5, V = 50, T = 124)
A song is `[BOS, STY0 + style, MAJ|MIN]`, then up to 24 sections, then `EOS`. Each section is 5 tokens:
1. `TYPE0 + t`, where t indexes intro, groove, build, hook, break, outro. `EOS` may appear at this slot.
2. `LEN0 + l`, where l indexes LENS = [4, 8, 12, 16, 24] bars.
3. `EN0 + e`, where e is the energy bucket 0–3 within the song.
4. `ACT0 + a`, where a is the 4-bit active-track set (0–15).
5. `TR0 + r`, where r indexes start, up, down, same, drop. "Drop" means drums return.

Token ids: PAD 0, BOS 1, EOS 2, STY0 3, MAJ 12, MIN 13, TYPE0 14, LEN0 20, EN0 25, ACT0 29, TR0 45.
To sample, mask each slot to its own token range, as `plan_mask` in `training/songs/songmodels.py` does.

## Composer (`songlm_s` is the promoted one; SLOT = 20, V = 176, T = 243, 12-bar context)
The prefix is `[BOS, STY0 + style, MAJ|MIN]`. Each bar is 20 tokens:
1. `BAR`, fed by the engine.
2. `SEC0 + type*4 + energy`, fed from the plan.
3. `ACT0 + set`, fed from the plan.
4. Chord: `CH0 + root*5 + quality`, with qualities maj7, m7, 7, m7b5, sus. Sampled.
5. 8 lead tokens: `M0` rest, `M0+1` hold, `M0+2+(p-57)` for p in 57..86. Sampled.
6. 8 bass tokens: `B0` rest, `B0+1` hold, `B0+2+(p-28)` for p in 28..55. Sampled.

Token ids: PAD 0, BOS 1, STY0 2, MAJ 11, MIN 12, BAR 13, SEC0 14, ACT0 38, CH0 54, M0 114, B0 146.
Pitches are relative to C major / A minor; the engine transposes them.
