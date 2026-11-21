// Check an isolated piano pluck and the bass pitch that used to wrap down at MIDI 46.
import assert from 'node:assert/strict';
import {refineRegister} from '../site/public/engine/register.js';
import {addSample, createBank, createRenderer, SR} from '../site/public/engine/render.js';

function pitches(notes, beats, type = 'keys', pattern = 'composer') {
  const bar = notes.map((pitch, i) => ({pitch, beat: beats[i], type}));
  refineRegister([bar], {sections: [{start: 0, bars: 1, tracks: {bass: {pattern}}}]});
  return bar.map(n => n.pitch);
}

assert.deepEqual(pitches([58, 41, 51], [0, .75, 2]), [58, 53, 51]);
assert.deepEqual(pitches([58, 41, 51], [0, .02, .04]), [58, 41, 51]);
assert.deepEqual(pitches([48, 36, 48], [0, 1, 2], 'bass', 'octave'), [48, 36, 48]);

const hz = note => 440 * 2 ** ((note - 69) / 12);
const bank = createBank();
const left = Float32Array.from({length: 2 * SR}, (_, i) => .1 * Math.sin(2 * Math.PI * hz(39) * i / SR));
addSample(bank, {kind: 'bass', name: '39', left});
for (const pitch of [45, 46]) {
  const bar = [{type: 'bass', pitch, beat: 0, duration: 2, gain: .3, pan: 0}];
  bar.info = {arranged: true};
  const render = createRenderer({bpm: 80, nextBar: () => bar}, 123, bank, 'steady');
  const samples = render().left;
  let crossings = 0;
  for (let i = SR / 4; i < SR * .75; i++) {
    if (samples[i-1] <= 0 && samples[i] > 0) crossings++;
  }
  assert.ok(Math.abs(crossings * 2 - hz(pitch)) < 3, 'Rendered bass must match its written pitch');
}
console.log('Isolated pluck, chord voicing, octave pattern and rendered bass pitches passed.');
