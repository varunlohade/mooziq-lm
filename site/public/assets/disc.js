// The compact disc, in three stacked layers sized to the screen (on-screen size times the
// pixel ratio, so it stays crisp on retina):
//   #disc          the physical disc and its print. Rendered once per size, then only rotated.
//   #disc-rainbow  diffraction from the pits, fixed in screen space (it depends on light and eye).
//   #disc-glint    the white specular reflection, also fixed in screen space.
// The two light layers blend with `screen` and turn only when the light direction moves.
// All geometry is in a 1000-unit space; the disc radius is about 497 units.
const cache = new Map();
let discLabel = ['', '', ''];
// Radii as a fraction of the disc radius, from a real 120 mm CD.
const HOLE = .125, FROST_IN = .16, STACK = .238, CLEAR_OUT = .27, MIRROR_OUT = .37, RIM = .985;
function discSize() {
  const canvas = document.querySelector('#disc');
  const css = canvas.getBoundingClientRect().width || 560;
  // Round up to 100 px steps so small layout changes do not force a repaint.
  return Math.min(1600, Math.max(400, Math.ceil(css * (devicePixelRatio || 1) / 100) * 100));
}
function seeded(seed) {
  return () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
}
function basePressing(n) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = n;
  const ctx = canvas.getContext('2d'), c = n / 2, radius = c - 2 * n / 1000, unit = 1000 / n;
  const pixels = ctx.createImageData(n, n), data = pixels.data, grain = seeded(271);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const dx = x + .5 - c, dy = y + .5 - c, r = Math.hypot(dx, dy), u = r / radius, r1 = r * unit;
    if (u > 1 || u < HOLE) continue;
    const i = (y * n + x) * 4, g = (grain() - .5) * 3;
    // Only concentric detail here: anything angular would visibly turn with the disc.
    let v, alpha = 255, tint = 3;
    if (u >= MIRROR_OUT) {
      // Data area: neutral silver, fine micro-rings, a slight falloff to the rim, faint track boundaries.
      v = 172 + 1.4 * Math.sin(r1 * 9.1) + .9 * Math.sin(r1 * 3.3) + .7 * Math.sin(r1 * 23.7) - 14 * (u - MIRROR_OUT) + g;
      for (const band of [.52, .71, .9]) v -= 7 * Math.exp(-((((u - band) * radius) / 1.6) ** 2));
      if (u > RIM) { v = 236; alpha = 70 + 150 * Math.max(0, (u - .996) / .004); }
    } else if (u >= CLEAR_OUT) {
      v = 188 + .6 * Math.sin(r1 * 2.1) + g * .4;
      if (u > MIRROR_OUT - .006) v -= 18 * (1 - (MIRROR_OUT - u) / .006);
    } else if (u >= STACK + .012) {
      v = 228; alpha = 60; tint = 5;
    } else if (u >= STACK) {
      // The raised stacking ring: a bright edge then a soft shadow.
      const t = (u - STACK) / .012;
      v = 250 - 70 * t; alpha = 150 - 70 * t; tint = 5;
    } else if (u >= FROST_IN) {
      v = 222 + g; alpha = 205; tint = 6;
    } else {
      v = 230; alpha = 55 + 80 * Math.exp(-((u - HOLE) * radius) / 2); tint = 5;
    }
    data[i] = v; data[i + 1] = v + 1; data[i + 2] = v + tint;
    data[i + 3] = alpha * Math.min(1, radius - r, (r - radius * HOLE) / 1.2);
  }
  ctx.putImageData(pixels, 0, 0);
  ctx.save(); ctx.translate(c, c); ctx.scale(n / 1000, n / 1000);
  // A few dust specks and hairline scratches, so the turning disc reads as turning.
  const dust = seeded(9001);
  for (let k = 0; k < 9; k++) {
    const a = dust() * Math.PI * 2, rr = 200 + dust() * 280, s = .6 + dust() * 1.4;
    ctx.fillStyle = dust() < .5 ? 'rgba(255,255,255,.75)' : 'rgba(60,60,64,.45)';
    ctx.beginPath(); ctx.arc(Math.cos(a) * rr, Math.sin(a) * rr, s, 0, Math.PI * 2); ctx.fill();
  }
  ctx.lineWidth = .6;
  for (let k = 0; k < 3; k++) {
    const a = dust() * Math.PI * 2, rr = 240 + dust() * 200, len = .08 + dust() * .12;
    ctx.strokeStyle = 'rgba(255,255,255,.35)';
    ctx.beginPath(); ctx.arc(0, 0, rr, a, a + len); ctx.stroke();
  }
  // Print on the disc.
  ctx.fillStyle = ctx.strokeStyle = '#1c3489'; ctx.textAlign = 'center';
  function curved(str, rad, center, spacing, size) {
    ctx.font = '600 ' + size + 'px Arial';
    [...str].forEach((ch, i) => { ctx.save(); ctx.rotate(center + (i - (str.length - 1) / 2) * spacing / rad); ctx.fillText(ch, 0, -rad); ctx.restore(); });
  }
  curved('UNLIMITED LOFI MUSIC    /    GENERATIVE STEREO RECORDINGS', 448, 0, 10, 15);
  ctx.font = 'italic 700 57px Arial'; ctx.fillText('unlimited', 0, -243);
  ctx.font = '600 15px Arial'; ctx.fillText('S O U N D   S Y S T E M', 0, -212);
  for (let row = 0; row < 19; row++) {
    ctx.beginPath();
    for (let k = 0; k <= 220; k++) {
      const t = k / 220, xp = (t - .5) * 535, envelope = Math.pow(Math.sin(Math.PI * t), 1.1);
      const yp = 244 + (row - 9) * 4.1 + envelope * (26 * Math.sin(t * 10 + row * .15) + 25 * Math.cos(t * 5 - row * .12));
      if (k === 0) ctx.moveTo(xp, yp); else ctx.lineTo(xp, yp);
    }
    ctx.lineWidth = 1.8; ctx.stroke();
  }
  ctx.font = 'bold 17px Arial'; ctx.fillText('DIGITAL AUDIO', 0, 346);
  ctx.font = '13px monospace'; ctx.fillText('ULR  /  STEREO', 0, 369);
  ctx.textAlign = 'left'; ctx.font = 'bold 16px Arial'; ctx.fillText('ULR—CD', -396, 6);
  ctx.textAlign = 'right'; ctx.fillText('STEREO', 393, 6);
  ctx.restore();
  return canvas;
}
// Visible spectrum, softened towards white so the colours stay calm.
function spectrum(nm) {
  let r = 0, g = 0, b = 0;
  if (nm < 440) { r = (440 - nm) / 60; b = 1; }
  else if (nm < 490) { g = (nm - 440) / 50; b = 1; }
  else if (nm < 510) { g = 1; b = (510 - nm) / 20; }
  else if (nm < 580) { r = (nm - 510) / 70; g = 1; }
  else if (nm < 645) { r = 1; g = (645 - nm) / 65; }
  else r = 1;
  const edge = nm < 420 ? .3 + .7 * (nm - 380) / 40 : nm > 680 ? .3 + .7 * (720 - nm) / 40 : 1;
  return [r, g, b].map(v => (.72 * v + .28) * edge);
}
// Light arrives along angle 0 of these layers; CSS rotates them to the light direction.
function lightLayers(n) {
  const make = () => { const c = document.createElement('canvas'); c.width = c.height = n; return c; };
  const rainbow = make(), glint = make(), c = n / 2, radius = c - 2 * n / 1000;
  const rp = rainbow.getContext('2d').createImageData(n, n), gp = glint.getContext('2d').createImageData(n, n);
  const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
  // Two opposite first-order wedges and a faint second-order pair beside them.
  const wedges = [[1.75, .36, 1], [1.75 + Math.PI, .36, 1], [1.75 + .78, .22, .28], [1.75 + Math.PI + .78, .22, .28]];
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const dx = x + .5 - c, dy = y + .5 - c, r = Math.hypot(dx, dy), u = r / radius;
    if (u < CLEAR_OUT || u > RIM) continue;
    const a = Math.atan2(dy, dx), i = (y * n + x) * 4;
    const edge = Math.min(1, (r - radius * CLEAR_OUT) / 1.2, (radius * RIM - r) / 1.2);
    // Diffraction lives on the pits, brightest at mid radius, fading to hub and rim.
    if (u >= MIRROR_OUT) {
      const envelope = Math.sin(Math.PI * (u - MIRROR_OUT) / (RIM - MIRROR_OUT)) ** 1.3;
      let red = 0, green = 0, blue = 0;
      for (const [center, width, gain] of wedges) {
        const d = wrap(a - center), w = Math.exp(-2 * (d / width) ** 2) * gain * envelope;
        if (w < .002) continue;
        // Wavelength sweeps across the wedge and shifts a little with radius.
        const nm = Math.max(385, Math.min(715, 550 + 330 * d / width * .5 + 90 * (u - .68)));
        const [sr, sg, sb] = spectrum(nm);
        red += sr * w; green += sg * w; blue += sb * w;
      }
      rp.data[i] = 200 * red; rp.data[i + 1] = 200 * green; rp.data[i + 2] = 200 * blue; rp.data[i + 3] = 255 * edge;
    }
    // One soft white reflection towards the light, strongest on the mirror band and mid disc.
    const d = wrap(a), lobe = Math.exp(-2 * (d / .3) ** 2), broad = .5 * Math.exp(-2 * (d / 1.2) ** 2);
    const radial = u < MIRROR_OUT ? 1 : .55 + .45 * Math.exp(-(((u - .6) / .22) ** 2));
    const v = Math.min(255, 235 * (lobe + broad) * radial);
    gp.data[i] = gp.data[i + 1] = gp.data[i + 2] = v; gp.data[i + 3] = 255 * edge;
  }
  rainbow.getContext('2d').putImageData(rp, 0, 0);
  glint.getContext('2d').putImageData(gp, 0, 0);
  return {rainbow, glint};
}
function layersFor(n) {
  if (!cache.has(n)) cache.set(n, {base: basePressing(n), ...lightLayers(n)});
  return cache.get(n);
}
function paintDisc(tune, key, bpm) {
  discLabel = [tune, key, bpm];
  const canvas = document.querySelector('#disc'), n = discSize(), layers = layersFor(n);
  for (const [id, source] of [['#disc-rainbow', layers.rainbow], ['#disc-glint', layers.glint]]) {
    const light = document.querySelector(id);
    if (light.width !== n) { light.width = light.height = n; light.getContext('2d').drawImage(source, 0, 0); }
  }
  if (canvas.width !== n) canvas.width = canvas.height = n;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, n, n);
  ctx.drawImage(layers.base, 0, 0);
  // The live edition is printed on the frosted clamping ring, on opposite arcs.
  ctx.save();
  ctx.translate(n / 2, n / 2); ctx.scale(n / 1000, n / 1000);
  ctx.fillStyle = '#23324d';
  ctx.font = 'italic 24px Georgia, serif';
  ctx.textAlign = 'center';
  function printArc(label, angle, radius) {
    const letters = Array.from(label);
    const widths = letters.map(letter => ctx.measureText(letter).width + 1);
    let cursor = angle - widths.reduce((sum, width) => sum + width, 0) / (2 * radius);
    letters.forEach((letter, i) => {
      ctx.save(); ctx.rotate(cursor + widths[i] / (2 * radius)); ctx.fillText(letter, 0, -radius); ctx.restore();
      cursor += widths[i] / radius;
    });
  }
  if (key) {
    printArc('Tune ' + tune, -Math.PI / 2, 86);
    printArc(key + ' / ' + bpm + ' BPM', Math.PI / 2, 86);
  }
  ctx.restore();
  canvas.setAttribute('aria-label', 'Silver CD' + (key ? ', tune ' + tune + ', ' + key + ', ' + bpm + ' BPM' : ''));
}
// Repaint at the new resolution when the disc's on-screen size or the pixel ratio changes.
let resizeTimer;
addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (discSize() !== document.querySelector('#disc').width) paintDisc(...discLabel); }, 200);
});

// Light direction: from the upper left, easing towards the pointer on desktop or following
// the phone's tilt where the browser allows it without a prompt. Reduced motion keeps it still.
(() => {
  const BASE = -128, SWING = 20, layers = ['#disc-rainbow', '#disc-glint'].map(id => document.querySelector(id));
  let angle = 0, target = 0, frame = 0;
  const apply = () => layers.forEach(layer => { layer.style.transform = 'rotate(' + (BASE + angle) + 'deg)'; });
  const step = () => {
    angle += (target - angle) * .06;
    apply();
    frame = Math.abs(target - angle) > .02 ? requestAnimationFrame(step) : 0;
  };
  const aim = deg => { target = Math.max(-SWING, Math.min(SWING, deg)); if (!frame) frame = requestAnimationFrame(step); };
  apply();
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const disc = document.querySelector('#disc');
  addEventListener('pointermove', event => {
    if (event.pointerType !== 'mouse') return;
    const box = disc.getBoundingClientRect();
    const toward = Math.atan2(event.clientY - box.top - box.height / 2, event.clientX - box.left - box.width / 2) * 180 / Math.PI;
    aim(Math.atan2(Math.sin((toward - BASE) * Math.PI / 180), Math.cos((toward - BASE) * Math.PI / 180)) * 180 / Math.PI * .25);
  });
  if ('DeviceOrientationEvent' in window && typeof DeviceOrientationEvent.requestPermission !== 'function') {
    addEventListener('deviceorientation', event => { if (event.gamma !== null) aim(event.gamma * .5); });
  }
})();
