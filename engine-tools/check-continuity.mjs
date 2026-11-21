// Exercise the production mixer across automatic and beat-aligned manual handoffs.
// Constant PCM distinguishes the inter-song pause from musical rests.
import assert from 'node:assert/strict';
globalThis.sampleRate = 1024;
globalThis.currentTime = 0;
globalThis.currentFrame = 0;
globalThis.AudioWorkletProcessor = class { port = {postMessage() {}}; };
let Mixer;
globalThis.registerProcessor = (_, processor) => { Mixer = processor; };
await import('../site/public/engine/worklet.js');
const mixer = new Mixer(), pcm = new Float32Array(6 * sampleRate).fill(.1);
const block = new Float32Array(32), right = new Float32Array(32);
function enqueue(id) {
  mixer.handle({t:'chunk', deck:id, left:pcm, right:pcm, bpm:79});
}
let silentFrames = 0;
function advance() {
  mixer.process([],[[block, right]]);
  for (const value of block) {
    assert.ok(value >= 0 && value <= .10001, 'overlapping decks');
    if (value === 0) silentFrames++;
  }
  globalThis.currentFrame += block.length;
  globalThis.currentTime += block.length / sampleRate;
}
enqueue(1);
while (!mixer.decks.get(1).pos || mixer.decks.get(1).fadeIn) advance();
for (let next=2; next<=1001; next++) {
  const old = mixer.decks.get(next-1);
  silentFrames = 0;
  enqueue(next);
  const at = next % 2 ? old.pos + 123 : -1;
  mixer.handle({t:'fade', from:old.id, to:next, at, frames:256});
  mixer.handle({t:'end',deck:old.id});
  assert.ok(old.sched.at >= old.pos);
  if (at < 0) assert.ok(Math.abs(old.sched.at - Math.round(old.sched.at / old.beatFrames) * old.beatFrames) <= .5);
  while (!mixer.decks.get(next).pos || mixer.decks.get(next).fadeIn) advance();
  assert.equal(silentFrames, sampleRate + 2);
  assert.ok(!mixer.decks.has(old.id));
  assert.equal(mixer.decks.size,1);
}
assert.equal(mixer.underrun,0);
console.log('1,000 mixer handoffs with one second between songs, zero underruns and retired outgoing decks.');
