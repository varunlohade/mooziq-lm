// An isolated octave error keeps the right pitch class, so a chord check misses
// it. Compare a pluck with the notes on either side in the same musical line.
import weights from './register-weights.js';

const threshold = Math.log(12);

export function refineRegister(bars, plan) {
  for (const section of plan.sections) {
    const lines = {lead: [], keys: [], bass: []};
    for (let i = section.start; i < section.start + section.bars; i++) {
      for (const note of bars[i]) {
        const part = note.type === 'felt' && !note.harmony ? 'lead' : note.type;
        if (part === 'bass' && section.tracks.bass?.pattern === 'octave') continue;
        if (Object.hasOwn(lines, part)) lines[part].push({note, at: i * 4 + note.beat});
      }
    }
    for (const [part, line] of Object.entries(lines)) {
      line.sort((a, b) => a.at - b.at);
      // Decide against the original phrase so one correction cannot trigger another.
      const changes = [];
      for (let i = 1; i + 1 < line.length; i++) {
        const left = line[i-1], middle = line[i], right = line[i+1];
        const before = middle.at - left.at, after = right.at - middle.at;
        // A block chord or a new phrase is not a three-note melodic gesture.
        if (Math.min(before, after) < .2 || Math.max(before, after) > 4) continue;
        const depth = Math.min(left.note.pitch, right.note.pitch) - middle.note.pitch;
        if (depth <= 6) continue;
        const scores = part === 'bass' ? weights.bass : weights.lead;
        if (scores[Math.min(depth, scores.length - 1)] >= threshold) changes.push(middle.note);
      }
      for (const note of changes) note.pitch += 12;
    }
  }
}
