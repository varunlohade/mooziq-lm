// Render the real worklet in 128-frame blocks. A constant signal makes both
// the last outgoing sample and the first incoming sample unambiguous.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source = fs.readFileSync(new URL('../site/public/engine/worklet.js', import.meta.url), 'utf8');
const sr = 44100, block = 128;
for (const change of ['scheduled', 'manual', 'replace-during-gap', 'replace-during-fade']) {
  let Mixer;
  const context = vm.createContext({
    sampleRate: sr, currentTime: 0, currentFrame: 0,
    AudioWorkletProcessor: class { port = {postMessage() {}}; },
    registerProcessor(name, ctor) { Mixer = ctor; }
  });
  vm.runInContext(source, context);
  const mixer = new Mixer();
  function queue(id) {
    const pcm = new Float32Array(8 * sr).fill(id === 1 ? .25 : -.25);
    mixer.handle({t:'chunk', deck:id, left:pcm, right:pcm, bpm:90});
  }
  queue(1);
  let lastPositive = 0, firstNegative = 0, replaced = false;
  for (let frame = 0; frame < 6 * sr; frame += block) {
    context.currentFrame = frame;
    context.currentTime = frame / sr;
    if (frame === 2 * block) {
      queue(2);
      mixer.handle({t:'fade', from:1, to:2, at:change === 'scheduled' ? 1895 : -1, frames:12007});
    }
    const outgoing = mixer.decks.get(1);
    if (!replaced && ((change === 'replace-during-gap' && !outgoing) ||
        (change === 'replace-during-fade' && outgoing?.fadeOut))) {
      queue(3);
      mixer.handle({t:'fade', to:3, at:-1, frames:12007});
      replaced = true;
    }
    const left = new Float32Array(block), right = new Float32Array(block);
    mixer.process([], [[left, right]]);
    assert.deepEqual(left, right);
    for (let i = 0; i < block; i++) {
      if (left[i] > 0) lastPositive = frame+i;
      if (!firstNegative && left[i] < 0) firstNegative = frame+i;
    }
    if (firstNegative) break;
  }
  const silence = firstNegative-lastPositive-1;
  assert(silence >= sr && silence <= sr+2, change+' silent frames: '+silence);
  assert.equal(mixer.underrun, 0);
  assert.equal(replaced, change.startsWith('replace-'));
  if (replaced) assert(!mixer.decks.has(2));
  console.log(change, silence+' silent frames');
}
