// Fetches and decodes the recorded samples in two tiers, so the first song can start
// before everything has arrived.
const RATE = 44100;
function trim(channel) {
  let peak = 0;
  for (let i = 0; i < channel.length; i++) peak = Math.max(peak, Math.abs(channel[i]));
  let start = 0;
  while (start < channel.length && Math.abs(channel[start]) < peak * .01) start++;
  return Math.max(0, start - 32);
}
async function decode(url, kind, name) {
  const bytes = await (await fetch(url)).arrayBuffer();
  const buffer = await new OfflineAudioContext(2, 1, RATE).decodeAudioData(bytes);
  const left = buffer.getChannelData(0), right = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : left;
  // Ambience loops play from their first sample so the loop point stays seamless.
  const start = kind === 'ambience' ? 0 : trim(left);
  return {kind, name, left: left.slice(start), right: right === left ? null : right.slice(start)};
}
const loadedAmbience = new Set();
// Ambience loops for a style, fetched once each the first time a style needs them.
const AMBIENCE = ['rain', 'cafe', 'forest', 'water'];
export function loadAmbience(base, kinds, onSample) {
  return Promise.all(kinds.filter(k => AMBIENCE.includes(k) && !loadedAmbience.has(k)).map(k => { loadedAmbience.add(k); return decode(`${base}samples/ambience/${k}.m4a`, 'ambience', k).then(onSample); }));
}
// Styles without drums start on piano alone, so they need only the piano first.
const QUIET = new Set(['flow', 'sleep', 'studypiano', 'nature']);
export async function loadSamples(base, style, onSample, onReady, later = Promise.resolve()) {
  const index = await (await fetch(base + 'samples/index.json')).json();
  const tier = (list, kind) => list.map(name => decode(`${base}samples/${kind}/${name}.m4a`, kind, name).then(onSample));
  // Tier 1 is what the first song needs: soft piano, plus bass and drums for beat styles.
  const band = QUIET.has(style) ? [] : [...tier(index.bass, 'bass'), ...tier(index.drums, 'drums')];
  await Promise.all([...tier(index.piano.filter(n => n.endsWith('L')), 'piano'), ...band]);
  onReady();
  // The rest waits until the first song is playing, so it does not slow the start.
  await later;
  await Promise.all([
    ...(QUIET.has(style) ? [...tier(index.bass, 'bass'), ...tier(index.drums, 'drums')] : []),
    ...tier(index.piano.filter(n => n.endsWith('H')), 'piano'),
    // Lead instruments that songs bring in later (the plan falls back to piano until they arrive).
    ...tier(index.vibes || [], 'vibes'), ...tier(index.guitar || [], 'guitar')
  ]);
}
