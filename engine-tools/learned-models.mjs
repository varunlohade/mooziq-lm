// Match the production engine. Set LEARNED=0 to test the preset fallback.
import fs from 'node:fs';
export function learnedModels() {
  if (process.env.LEARNED === '0') return undefined;
  const part = name => {
    const meta = JSON.parse(fs.readFileSync(new URL(`../site/public/model/learned/${name}.json`, import.meta.url)));
    const b = fs.readFileSync(new URL(`../site/public/model/learned/${name}.bin`, import.meta.url));
    return {meta, bytes: b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)};
  };
  return {planner: part('planner'), composer: part('songlm_s')};
}
