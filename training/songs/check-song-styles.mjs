// Exercise the full arrangement, including transitions, motif development and the ending.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {STYLES} from '../public/engine/styles/index.js';
import {PRESETS} from '../public/engine/plan.js';
import {learnedPlan} from '../public/engine/learned.js';
import {createArrangedSong} from '../public/engine/arranger.js';
import {learnedModels} from './learned-models.mjs';

const learned = learnedModels();
const root = new URL('../public/model/', import.meta.url);
const meta = JSON.parse(fs.readFileSync(new URL('model.json', root)));
const bytes = Uint8Array.from(fs.readFileSync(new URL('model.bin', root))).buffer;
const drums = new Set(['kick', 'hat', 'rim', 'brush', 'snare', 'shaker']);
for (const [style, engine] of Object.entries(STYLES)) {
  for (let seed = 0; learned && seed < 100; seed++) {
    const {bpm, mode} = engine.songPlan(seed);
    const plan = learnedPlan(learned.planner, style, seed, {bpm, mode});
    const seconds = plan.totalBars * 240 / bpm;
    assert(seconds >= 180 && seconds <= 270, JSON.stringify({style, seed, seconds}));
  }
  for (const seed of [1, 12, 123]) {
    const song = createArrangedSong(engine, meta, bytes, seed, style, undefined, undefined, learned);
    for (let i = 0; i < song.totalBars; i++) {
      const bar = song.nextBar();
      for (const event of bar) {
        if (!PRESETS[style].beat) assert(!drums.has(event.type), style + ' ' + event.type);
        if (PRESETS[style].solo) assert(['keys', 'felt', 'arp', 'swell'].includes(event.type) && event.inst === 'piano', JSON.stringify(event));
      }
    }
  }
  console.log(style, 'passed');
}
