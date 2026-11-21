// LEARNED=1 runs a tool on the learned song path (engine/learned.js): this loads the planner
// and song composer models from public/model/learned/, or returns undefined.
import fs from 'node:fs';
export function learnedModels() {
  if (!process.env.LEARNED) return undefined;
  const part = name => {
    const meta = JSON.parse(fs.readFileSync(new URL(`../public/model/learned/${name}.json`, import.meta.url)));
    const b = fs.readFileSync(new URL(`../public/model/learned/${name}.bin`, import.meta.url));
    return {meta, bytes: b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)};
  };
  return {planner: part('planner'), composer: part('songlm_s')};
}
