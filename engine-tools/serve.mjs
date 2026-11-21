// Local static server for public/ with the exact headers from vercel.json.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
const root = new URL('../site/public/', import.meta.url).pathname;
const config = JSON.parse(fs.readFileSync(new URL('../site/vercel.json', import.meta.url)));
const types = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json', '.bin': 'application/octet-stream', '.png': 'image/png'};
export function serve(port = 0) {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://x');
    // Vercel serves its analytics script on the deployed site; locally it is an empty stand-in.
    if (url.pathname === '/_vercel/insights/script.js') {
      for (const {key, value} of config.headers[0].headers) res.setHeader(key, value);
      res.writeHead(200, {'Content-Type': 'text/javascript'}).end('');
      return;
    }
    let file = path.join(root, decodeURIComponent(url.pathname));
    if (!file.startsWith(root)) { res.writeHead(403).end(); return; }
    if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, 'index.html');
    if (!fs.existsSync(file)) { res.writeHead(404).end('not found'); return; }
    for (const rule of config.headers) {
      const re = new RegExp('^' + rule.source.replace('(.*)', '(.*)') + '$');
      if (re.test(url.pathname)) for (const {key, value} of rule.headers) res.setHeader(key, value);
    }
    res.setHeader('Content-Type', types[path.extname(file)] || 'application/octet-stream');
    fs.createReadStream(file).pipe(res);
  });
  return new Promise(resolve => server.listen(port, '127.0.0.1', () => resolve(server)));
}
// Run directly to serve the site: node engine-tools/serve.mjs [port=8080]
if (import.meta.url === `file://${process.argv[1]}`) {
  const server = await serve(Number(process.argv[2] || 8080));
  console.log('http://127.0.0.1:' + server.address().port + '/');
}
