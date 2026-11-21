// Steady seed 2801 used to flatten a returning three-note dip into one pitch.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createArrangedSong} from '../site/public/engine/arranger.js';
import {STYLES} from '../site/public/engine/styles/index.js';
import {learnedModels} from './learned-models.mjs';

const meta = JSON.parse(fs.readFileSync(new URL('../site/public/model/model.json', import.meta.url)));
const bin = fs.readFileSync(new URL('../site/public/model/model.bin', import.meta.url));
const bytes = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
const song = createArrangedSong(STYLES.steady, meta, bytes, 2801, 'steady', undefined, undefined, learnedModels());
const bars = Array.from({length: song.totalBars}, () => song.nextBar());
const lead = start => bars.slice(start, start + 2).flat().filter(n => n.type === 'felt' && !n.harmony);
const contour = notes => notes.slice(1).map((n, i) => Math.sign(n.pitch - notes[i].pitch));
const theme = lead(song.development.motif.bar);
const refrains = song.development.occurrences.filter(o => o.kind === 'refrain');
assert.deepEqual(contour(theme), [-1, 1]);
assert.ok(refrains.length >= 2);
for (const refrain of refrains) {
  assert.deepEqual(contour(lead(refrain.bar)), contour(theme), 'Lost the melodic dip at bar ' + refrain.bar);
}
console.log('The returning Steady theme keeps its dip over the new chords.');
