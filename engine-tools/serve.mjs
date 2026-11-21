// Serve the player on loopback. Offline renderers may also supply exact file routes.
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {pipeline} from 'node:stream/promises';

const root = await fs.realpath(fileURLToPath(new URL('../site/public/', import.meta.url)));
const config = JSON.parse(await fs.readFile(new URL('../site/vercel.json', import.meta.url)));
const types = {'.html':'text/html; charset=utf-8', '.js':'application/javascript', '.css':'text/css',
  '.json':'application/json', '.png':'image/png', '.m4a':'audio/mp4'};
const rules = config.headers.map(rule => ({...rule, pattern:new RegExp('^' + rule.source + '$')}));

export async function serve(port = 0, mounts = new Map()) {
  const server = http.createServer(async (req, res) => {
    if (!['127.0.0.1', 'localhost'].some(host => req.headers.host === host + ':' + server.address().port)) {
      res.writeHead(403).end(); return;
    }
    if (!['GET','HEAD'].includes(req.method)) { res.writeHead(405,{Allow:'GET, HEAD'}).end(); return; }
    let route;
    try { route = decodeURIComponent(new URL(req.url, 'http://localhost').pathname); }
    catch { res.writeHead(400).end(); return; }
    if (route.includes('\0')) { res.writeHead(400).end(); return; }
    for (const rule of rules) {
      if (rule.pattern.test(route)) for (const {key,value} of rule.headers) res.setHeader(key,value);
    }
    let handle;
    try {
      let file = mounts.get(route);
      if (!file) {
        file = await fs.realpath(path.resolve(root, '.' + route));
        if (file === root || (await fs.stat(file)).isDirectory()) file = await fs.realpath(path.join(file,'index.html'));
        if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
      }
      handle = await fs.open(file, 'r');
      const stat = await handle.stat();
      if (!stat.isFile()) { res.writeHead(404).end(); return; }
      res.writeHead(200, {'Content-Type':types[path.extname(file)] || 'application/octet-stream','Content-Length':stat.size});
      if (req.method === 'HEAD') res.end();
      else await pipeline(handle.createReadStream(), res);
    } catch {
      if (!res.headersSent && !res.destroyed) res.writeHead(404).end();
      else if (!res.destroyed) res.destroy();
    } finally { await handle?.close().catch(() => {}); }
  });
  return new Promise((resolve,reject) => {
    server.once('error', reject);
    server.listen(port,'127.0.0.1',() => resolve(server));
  });
}
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const server = await serve(Number(process.argv[2] || 8080));
  console.log('http://127.0.0.1:' + server.address().port + '/');
}
