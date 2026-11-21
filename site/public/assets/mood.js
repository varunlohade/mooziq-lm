'use strict';
// Day/night switch with the tile-flip ripple from varunlohade.com. Adapted from the main
// site's inline script: lofi uses data-mood (day/night), an external stylesheet, and canvases.
(() => {
  const toggle = document.querySelector('[data-theme-toggle]');
  if (!toggle) return;
  const root = document.documentElement;
  const repaint = () => { if (typeof paintRoll === 'function') paintRoll(); };
  const apply = (mood) => {
    params.delete('mood');
    displayMood(mood);
    try { localStorage.setItem(MOOD_KEY, mood); } catch {}
    repaint();
  };
  // Tile flip: the page splits into tiny tiles that lift off the screen, drift,
  // tilt and flip over to the new mood, in a wobbly ripple from the button.
  const SPEED = 1.1, FLIP = 300;
  const css = () => [...document.styleSheets].flatMap((sheet) => [...sheet.cssRules].map((rule) => rule.cssText))
    .join('\n').replace(/:root/g, 'html') + '\n*{transition:none!important;animation:none!important}';
  // A foreignObject snapshot cannot copy canvas pixels, so each canvas goes in as an image
  // of its current frame at its on-screen size. Only the roll depends on the mood.
  const encode = (c, dpr) => {
    const out = document.createElement('canvas');
    out.width = Math.max(1, Math.round(c.offsetWidth * dpr)); out.height = Math.max(1, Math.round(c.offsetHeight * dpr));
    out.getContext('2d').drawImage(c, 0, 0, out.width, out.height);
    return out.toDataURL();
  };
  const canvasImages = (moods, dpr) => {
    const live = root.dataset.mood, canvases = [...document.querySelectorAll('canvas:not(.px-fx)')];
    const fixed = canvases.map((c) => c.id === 'roll' ? null : encode(c, dpr));
    const images = {};
    for (const mood of moods) {
      if (mood !== root.dataset.mood) { root.dataset.mood = mood; repaint(); }
      images[mood] = canvases.map((c, i) => fixed[i] ?? encode(c, dpr));
    }
    if (root.dataset.mood !== live) { root.dataset.mood = live; repaint(); }
    return images;
  };
  // Draw the page, in the given mood, as it looks in the viewport right now. The copy lives in
  // an inert document, and canvas sizes go into the snapshot's own stylesheet, not style attributes.
  const snapshot = async (mood, pixels, W, H, dpr) => {
    const doc = document.implementation.createHTMLDocument('');
    const html = doc.importNode(root, true);
    html.dataset.mood = mood;
    html.querySelectorAll('script, .px-fx').forEach((n) => n.remove());
    html.querySelectorAll('[style]').forEach((n) => n.removeAttribute('style'));
    const live = document.querySelectorAll('canvas:not(.px-fx)'), copies = html.querySelectorAll('canvas:not(.px-fx)');
    let rules = '';
    copies.forEach((copy, i) => {
      const img = doc.createElement('img');
      for (const {name, value} of copy.attributes) if (name !== 'width' && name !== 'height' && name !== 'style') img.setAttribute(name, value);
      img.setAttribute('src', pixels[i]);
      img.setAttribute('data-snap', i);
      rules += `[data-snap="${i}"]{display:block;width:${live[i].offsetWidth}px;height:${live[i].offsetHeight}px;transform:${live[i].style.transform || 'none'}}\n`;
      copy.replaceWith(img);
    });
    const inputs = document.querySelectorAll('input'), inputCopies = html.querySelectorAll('input');
    inputCopies.forEach((input, i) => input.setAttribute('value', inputs[i].value));
    const style = doc.createElement('style');
    style.textContent = css() + '\n' + rules;
    html.querySelector('head').replaceChildren(style);
    const xml = new XMLSerializer().serializeToString(html);
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}">` +
      `<foreignObject x="0" y="${-scrollY}" width="${W}" height="${root.scrollHeight}">${xml}</foreignObject></svg>`;
    const img = new Image();
    img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
    await img.decode();
    const c = document.createElement('canvas');
    c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
    c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    return c;
  };
  const tileFlip = async (x, y, next, swap, stats) => {
    const W = root.clientWidth, H = innerHeight;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const now = root.dataset.mood, pixels = canvasImages([now, next], dpr);
    const [before, after] = await Promise.all([snapshot(now, pixels[now], W, H, dpr), snapshot(next, pixels[next], W, H, dpr)]);
    const T = W < 600 ? 10 : 14, P = T * dpr;
    const cols = Math.ceil(W / T), rows = Math.ceil(H / T), n = cols * rows;
    const delay = new Float32Array(n), tilt = new Float32Array(n), ux = new Float32Array(n), uy = new Float32Array(n);
    const axisY = new Uint8Array(n), dir = new Int8Array(n);
    let last = 0;
    const ph1 = Math.random() * 6.3, ph2 = Math.random() * 6.3;
    for (let j = 0, k = 0; j < rows; j++) for (let i = 0; i < cols; i++, k++) {
      const dx = (i + .5) * T - x, dy = (j + .5) * T - y;
      const d = Math.hypot(dx, dy) || 1, a = Math.atan2(dy, dx);
      const wobble = (22 * Math.sin(3 * a + ph1) + 10 * Math.sin(7 * a + ph2)) * Math.min(1, d / 200);
      delay[k] = Math.max(0, (d + wobble) / SPEED + Math.random() * 25);
      if (delay[k] > last) last = delay[k];
      tilt[k] = (Math.random() - .5) * .9;
      ux[k] = dx / d; uy[k] = dy / d;
      axisY[k] = Math.abs(dx) > Math.abs(dy) ? 1 : 0;
      dir[k] = axisY[k] ? (Math.sign(dx) || 1) : (Math.sign(dy) || 1);
    }
    const fx = document.createElement('canvas');
    fx.className = 'px-fx'; fx.setAttribute('aria-hidden', 'true');
    fx.width = Math.round(W * dpr); fx.height = Math.round(H * dpr);
    const g = fx.getContext('2d');
    const ease = (p) => p < .5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
    const moving = [];
    const draw = (t) => {
      g.setTransform(1, 0, 0, 1, 0, 0);
      g.drawImage(after, 0, 0);
      moving.length = 0;
      for (let j = 0, k = 0; j < rows; j++) for (let i = 0; i < cols; i++, k++) {
        const p = (t - delay[k]) / FLIP;
        if (p <= 0) g.drawImage(before, i * P, j * P, P, P, i * P, j * P, P, P);
        else if (p < 1) moving.push(k, p);
      }
      // Lifted tiles: perspective scale, drift, tilt and a flip around one axis.
      for (let m = 0; m < moving.length; m += 2) {
        const k = moving[m], p = moving[m + 1], i = k % cols, j = (k - i) / cols;
        const z = Math.pow(Math.sin(Math.PI * p), .7);
        const s = 1 / (1 - .42 * z);
        const flip = Math.PI * ease(p);
        const c = Math.cos(flip), face = Math.abs(c);
        const cx = (i + .5) * T, cy = (j + .5) * T;
        const px = cx + (cx - x) * (s - 1) * .12 + ux[k] * T * .5 * z;
        const py = cy + (cy - y) * (s - 1) * .12 + uy[k] * T * .5 * z - T * .4 * z;
        // soft shadow on the page below
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.globalAlpha = .22 * z;
        g.fillStyle = '#000';
        g.fillRect(cx - T / 2 + T * .5 * z, cy - T / 2 + T * .7 * z, T, T);
        const rot = tilt[k] * z, cs = Math.cos(rot), sn = Math.sin(rot);
        const sx = s * (axisY[k] ? Math.max(face, .02) : 1), sy = s * (axisY[k] ? 1 : Math.max(face, .02));
        g.setTransform(dpr * cs * sx, dpr * sn * sx, -dpr * sn * sy, dpr * cs * sy, dpr * px, dpr * py);
        g.globalAlpha = 1;
        g.drawImage(c > 0 ? before : after, i * P, j * P, P, P, -T / 2, -T / 2, T, T);
        g.globalAlpha = .3 * (1 - face) + .08 * z;
        g.fillRect(-T / 2, -T / 2, T, T);
      }
      g.globalAlpha = 1;
    };
    draw(0);
    document.body.append(fx);
    stats.shown = performance.now();
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    swap();
    const total = last + FLIP;
    await new Promise((done) => {
      const t0 = performance.now();
      const frame = (now) => { const t = now - t0; draw(t); stats.frames++; t < total ? requestAnimationFrame(frame) : done(); };
      requestAnimationFrame(frame);
    });
    stats.end = performance.now();
    fx.remove();
  };
  displayMood(root.dataset.mood);
  toggle.removeAttribute('hidden');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const flips = (window.lofiDebug || {}).flips = [];
  let busy = false;
  toggle.addEventListener('click', () => {
    if (busy) return;
    const next = root.dataset.mood === 'night' ? 'day' : 'night';
    if (reduced.matches) return apply(next);
    const r = toggle.getBoundingClientRect(), stats = {click: performance.now(), frames: 0};
    flips.push(stats);
    busy = true;
    let swapped = false;
    const swap = () => { swapped = true; apply(next); };
    tileFlip(r.left + r.width / 2, r.top + r.height / 2, next, swap, stats)
      .catch(() => { document.querySelector('.px-fx')?.remove(); if (!swapped) swap(); })
      .finally(() => { busy = false; });
  });
})();
