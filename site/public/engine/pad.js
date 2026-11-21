// A soft root-and-fifth pad under each bar (or each two-bar Flow chord), so the low mids
// never fall silent. It sounds only inside its own span and waits for any lead note from the previous bar
// that would sit a semitone away, so it adds no minor-2nd or minor-9th clash.
const QIV = [[0, 4, 7, 11], [0, 3, 7, 10], [0, 4, 7, 10], [0, 3, 6, 10], [0, 5, 7, 10]];
export const PAD_RELEASE = .35;
const mod = (n, m) => (n % m + m) % m;
export function padEvents(info, shift, bpm, previous = [], span = 4) {
  const root = 46 + mod(info.root + shift - 46, 12), pitches = [root, root + QIV[info.quality][2]];
  const beatsPerSec = bpm / 60;
  let start = 0;
  for (const e of previous) {
    if (e.type !== 'felt') continue;
    const end = e.beat + e.duration + (e.releaseSec ?? .32) * beatsPerSec - 4;
    if (end > start && pitches.some(p => Math.abs(p - e.pitch) % 12 === 1)) start = end + .05;
  }
  const duration = span - start - PAD_RELEASE * beatsPerSec - .02;
  if (duration < 1) return [];
  return pitches.map((pitch, i) => ({type: 'pad', beat: start, pitch, duration, gain: i ? .040 : .052,
    pan: i ? .22 : -.22, seed: (pitch * 2654435761 + info.root * 97 + info.quality) >>> 0, releaseSec: PAD_RELEASE}));
}
