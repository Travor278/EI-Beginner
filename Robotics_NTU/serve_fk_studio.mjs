import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const port = Number(process.env.PORT || process.argv[2] || 8080);

const mime = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

const server = http.createServer(async (req, res) => {
  try {
    const rawPath = decodeURIComponent((req.url || '/').split('?')[0]);
    const relativePath = rawPath === '/' ? '/fk_studio.html' : rawPath;
    const normalized = path.normalize(relativePath).replace(/^(\.\.[/\\])+/, '');
    const filePath = path.join(__dirname, normalized);

    const body = await fs.readFile(filePath);
    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, { 'Content-Type': mime[ext] || 'application/octet-stream' });
    res.end(body);
  } catch {
    res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Not found');
  }
});

server.listen(port, '127.0.0.1', () => {
  console.log(`FK Studio server running: http://127.0.0.1:${port}/fk_studio.html`);
  console.log(`RRR entry: http://127.0.0.1:${port}/rrr_fk_viewer.html`);
  console.log(`Robot entry: http://127.0.0.1:${port}/robot_fk_viewer.html`);
});
