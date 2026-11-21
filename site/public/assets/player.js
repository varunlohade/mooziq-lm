'use strict';
const $ = selector => document.querySelector(selector);
const disc = $('#disc'), play = $('#play'), notes = [];
const motion = matchMedia('(prefers-reduced-motion: reduce)');
const query = new URLSearchParams(location.search);
const SR = 44100, STYLE_KEY = 'lofi-style';
const STYLE_NAMES = {steady: 'Steady', flow: 'Flow', sleep: 'Sleep', studypiano: 'Study Piano', nature: 'Nature', rainycafe: 'Rainy Café', chillhop: 'Chillhop', jazzhop: 'jazz(experimental)'};
const STYLES = Object.keys(STYLE_NAMES);
const SECTION_NAMES = {intro: 'intro', A: 'theme', A2: 'theme', B: 'second theme', B2: 'second theme', break: 'break', bridge: 'bridge', outro: 'outro'};
// Wander was replaced by Jazz Hop; old links and saved choices map across.
const styleName = value => value === 'wander' ? 'jazzhop' : value;
const shared = /^[0-9a-f]{1,8}$/i.test(query.get('seed') || '') ? parseInt(query.get('seed'), 16) >>> 0 : null;
const sessionSeed = shared ?? crypto.getRandomValues(new Uint32Array(1))[0];
const seedText = sessionSeed.toString(16).padStart(8, '0').toUpperCase();
function storedStyle() { try { return styleName(localStorage.getItem(STYLE_KEY)); } catch { return null; } }
const askedStyle = styleName(query.get('style'));
let style = STYLES.includes(askedStyle) ? askedStyle : STYLES.includes(storedStyle()) ? storedStyle() : STYLES[0];
let playing = false, tune = 0, bpm = 0, key = '', position = 0, frame = 0, speed = 0, lastFrame = 0;
let ctx, master, node, report = null, shown = 0, cursor = null;
const subtitle = size => `A never-ending radio, written live by a ${size} model on your device.`;
const tunes = new Map(), written = new Map();
let playedSection = '', melody = '', playedBar = 0, writingBar = 0, previousToken = '', currentToken = '', nextToken = '';
const trackNumber = () => String(tune).padStart(3, '0');
// Vercel Web Analytics: queue custom events until its script loads. Only the style is sent.
const track = (name, data) => window.va?.('event', {name, data});
const debug = window.lofiDebug = {sessionSeed, shownLog: [], fades: [], barMs: [], report: null, skips: [],
  get ctx() { return ctx; }, get master() { return master; }};

function updatePanel(force = false) {
  if (playing && motion.matches && !force) return;
  paintRoll();
  const bars = melody.match(/[^|]+(?:\||$)/g) || [];
  $('#code').replaceChildren(...bars.slice(-2).map(bar => {
    const span = document.createElement('span');
    span.textContent = bar;
    return span;
  }));
  const cells = $('#ticker').children;
  cells[0].textContent = previousToken;
  cells[1].textContent = currentToken;
  cells[2].textContent = nextToken;
  $('#progress').textContent = 'Playing bar ' + playedBar + (playedSection ? ' of the ' + playedSection : '') + ', writing bar ' + writingBar + '.';
  $('#bar').textContent = 'Bar ' + String(writingBar).padStart(3, '0');
}
function paintRoll() {
  const c=$('#roll'), ctx=c.getContext('2d'), style=getComputedStyle(document.documentElement);
  const w=c.width,h=c.height;
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle=style.getPropertyValue('--line').trim();ctx.lineWidth=1;
  for(let i=0;i<=16;i++){
    const x=Math.min(w-1,i*w/16)+.5;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();
  }
  for(let i=0;i<=8;i++){
    const y=Math.min(h-1,i*h/8)+.5;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();
  }
  const recent=notes.slice(-16);
  recent.forEach((note,i)=>{
    ctx.fillStyle=style.getPropertyValue(i===recent.length-1?'--ink':'--fg').trim();
    ctx.fillRect(i*w/16+1,(7-note)*h/8+4,w/16-4,8);
  });
}
const barText = bar => bar.chord + ' ' + bar.tokens.join(' ') + '|';
const hooks = {
  onTune(data) {
    tune = data.tune;
    bpm = data.bpm;
    key = data.key;
    $('#tune').textContent = trackNumber();
    $('#key').textContent = key;
    $('#bpm').textContent = bpm;
    notes.length = 0;
    melody = '';
    playedBar = writingBar = 0;
    previousToken = currentToken = nextToken = '';
    cursor = null;
    for (const bar of written.get(shown) || []) hooks.onBar(bar, false);
    paintDisc(trackNumber(), key, bpm);
    updateSession();
    updatePanel(true);
  },
  onNote(degree) {
    notes.push(Math.max(0, Math.min(7, degree)));
    if (notes.length > 16) notes.shift();
  },
  onBar(bar, paint = true) {
    writingBar = bar.bar;
    melody += barText(bar);
    const bars = melody.split('|');
    if (bars.length > 4) melody = bars.slice(-4).join('|');
    $('#roll').setAttribute('aria-label', 'Piano roll of the current melody, writing ' + bar.chord);
    if (paint) updatePanel();
  },
  onPlayback(bar, slot) {
    const bars = written.get(shown) || [], i = bars.indexOf(bar);
    const flat = [...(bars[i - 1]?.tokens || []), ...bar.tokens, ...(bars[i + 1]?.tokens || [])];
    const at = (bars[i - 1] ? 8 : 0) + slot;
    playedBar = bar.bar;
    playedSection = SECTION_NAMES[bar.section] || '';
    previousToken = flat[at - 1] || '';
    currentToken = flat[at];
    nextToken = flat[at + 1] || '';
    for (const note of bar.notes) if (note.slot === slot) hooks.onNote(note.degree);
    if (!motion.matches && slot % 2 === 0) $('#roll').animate([{opacity:.76},{opacity:1}],{duration:200});
    updatePanel();
  }
};

// The engine: a Worker writes and renders bars, an AudioWorklet mixes them.
const worker = new Worker('engine/worker.js', {type: 'module'});
worker.onerror = event => console.error('Engine failed', event.message);
worker.onmessage = ({data}) => {
  if (data.t === 'change-ready') {
    changeBusy = false;
    if (pendingChange) requestChange(pendingChange);
    return;
  }
  if (data.t === 'writing') {
    modelState.writing = data.section < data.of ? [data.section, data.of] : null;
    showModelStatus();
    return;
  }
  if (data.t === 'tune') {
    tunes.set(data.deck, data.info);
    written.set(data.deck, []);
    if (!shown || data.now || (data.manual && !playing)) show(data.deck);
  } else if (data.t === 'bar') {
    firstBar();
    const bars = written.get(data.deck);
    if (!bars) return;
    bars.push(data);
    if (bars.length > 12) bars.shift();
    debug.barMs.push(data.ms);
    if (debug.barMs.length > 600) debug.barMs.splice(0, 300);
    if (data.deck === shown) hooks.onBar(data);
  } else if (data.t === 'dropped') {
    tunes.delete(data.deck);
    written.delete(data.deck);
  }
};
// The page fetches the model weights itself, so the download starts before the worker's
// own imports have loaded.
// The model (483K parameters, 0.97 MB) downloads as the page opens and is kept in Cache
// Storage, so a repeat visit starts at once. Music starts once it is ready and the first
// song is written.
// Rename the cache whenever the shipped model changes; older copies are deleted.
const MODEL = {bin: 'model/model.bin', json: 'model/model.json'}, MODEL_CACHE = 'lofi-model-song-v2';
caches?.keys?.().then(names => names.filter(n => n.startsWith('lofi-model') && n !== MODEL_CACHE).forEach(n => caches.delete(n))).catch(() => {});
const modelState = debug.model = {phase: 'loading', received: 0, total: 0, cached: false, writing: null};
async function cachedFetch(url, onProgress) {
  let cache = null;
  try { cache = await caches.open(MODEL_CACHE); const hit = await cache.match(url); if (hit) { modelState.cached = true; return hit.arrayBuffer(); } } catch {}
  const response = await fetch(url);
  if (!response.ok) throw new Error(url + ' ' + response.status);
  const total = +response.headers.get('content-length') || 0, reader = response.body.getReader(), chunks = [];
  let received = 0;
  for (;;) {
    const {done, value} = await reader.read();
    if (done) break;
    chunks.push(value); received += value.length;
    onProgress?.(received, total);
  }
  const blob = new Blob(chunks);
  cache?.put(url, new Response(blob, {headers: {'Content-Type': 'application/octet-stream'}})).catch(() => {});
  return blob.arrayBuffer();
}
const mb = n => (n / 1e6).toFixed(1);
function showModelStatus() {
  const line = $('#model-status'), waiting = playing && !debug.firstAudio;
  if (modelState.phase === 'loading' && waiting) line.textContent = `Downloading the model… ${mb(modelState.received)} / ${mb(modelState.total || 967306)} MB`;
  else if (modelState.writing && waiting) line.textContent = `Writing the song… section ${modelState.writing[0]} of ${modelState.writing[1]}`;
  else line.textContent = '';
  line.hidden = !line.textContent;
}
(async () => {
  try {
    const part = async (json, bin, progress) => ({
      meta: JSON.parse(new TextDecoder().decode(await cachedFetch(json))),
      bytes: await cachedFetch(bin, progress)
    });
    const progress = (received, total) => {
      Object.assign(modelState, {received, total});
      showModelStatus();
    };
    const loadLearned = async () => {
      const [planner, composer] = await Promise.all([
        part('model/learned/planner.json', 'model/learned/planner.bin'),
        part('model/learned/songlm_s.json', 'model/learned/songlm_s.bin?v=balanced1', progress)
      ]);
      return {planner, composer};
    };
    const learned = await loadLearned().catch(error => {
      console.warn('Using the preset engine', error);
      return null;
    });
    const composer = learned?.composer ?? await part(MODEL.json, MODEL.bin, progress);
    modelState.phase = 'ready'; modelState.readyAt = performance.now();
    const buffers = [composer.bytes];
    if (learned) buffers.push(learned.planner.bytes);
    worker.postMessage({t: 'init', seed: sessionSeed, style, ...composer, learned}, buffers);
    showModelStatus();
  } catch (error) { console.error('Model failed to load', error); }
})();
let firstBar;
const firstBarWritten = new Promise(resolve => { firstBar = resolve; });
// Recorded instruments: decode on this thread (the worker has no audio decoder), then hand them over.
import(new URL('engine/loader.js', location.href).href).then(({loadSamples}) => loadSamples(new URL('.', location.href).href, style,
  sample => worker.postMessage({t: 'sample', sample}, [sample.left.buffer, ...(sample.right ? [sample.right.buffer] : [])]),
  () => { debug.samplesReady = performance.now(); worker.postMessage({t: 'bank-ready'}); },
  firstBarWritten
)).catch(error => console.error('Samples failed to load', error));
// Ambience recordings (rain, café, forest, water) load the first time a style uses them.
function loadAmbience(kind) {
  Promise.all([import(new URL('engine/loader.js', location.href).href), import(new URL('engine/plan.js', location.href).href)])
    .then(([{loadAmbience}, {PRESETS}]) => loadAmbience(new URL('.', location.href).href, PRESETS[kind]?.textures || [],
      sample => worker.postMessage({t: 'sample', sample}, [sample.left.buffer, ...(sample.right ? [sample.right.buffer] : [])])))
    .catch(error => console.error('Ambience failed to load', error));
}
loadAmbience(style);
function show(id) {
  if (id === shown || !tunes.has(id)) return;
  shown = id;
  debug.shownLog.push({deck: id, style: tunes.get(id).style, at: performance.now()});
  if (debug.shownLog.length > 200) debug.shownLog.splice(0, 100);
  debug.engine = tunes.get(id).engine;
  $('#subtitle-text').textContent = debug.engine === 'learned'
    ? subtitle('507K-parameter')
    : subtitle('483K-parameter');
  hooks.onTune(tunes.get(id));
}
function onMixer({data}) {
  if (data.t === 'pos') {
    report = debug.report = {...data, at: ctx.currentTime};
    rememberPlayback(data);
    if (!debug.firstAudio && data.decks.some(d => d[1] > 0)) {
      debug.firstAudio = performance.now();
    }
    let lead = null;
    for (const d of data.decks) if (d[4] === 'play' && (!lead || d[2] > lead[2])) lead = d;
    if (lead && lead[0] > shown) show(lead[0]);
  } else if (data.t === 'fade') {
    debug.fades.push({to: data.to, at: performance.now()});
    if (debug.fades.length > 200) debug.fades.splice(0, 100);
  }
}
// Follow the sounding bar and eighth of the shown song.
function follow() {
  if (!report || !ctx) return;
  const deck = report.decks.find(d => d[0] === shown), bars = written.get(shown);
  if (!deck || !bars) return;
  const elapsed = ctx.state === 'running' ? (ctx.currentTime - report.time) * SR : 0;
  const pos = deck[1] + Math.max(0, Math.min(deck[3], elapsed));
  const bar = bars.find(b => pos >= b.start && pos < b.start + b.frames);
  if (!bar) return;
  const slot = Math.min(7, Math.floor((pos - bar.start) / bar.frames * 8));
  if (cursor && cursor.bar === bar && cursor.slot === slot) return;
  cursor = {bar, slot};
  hooks.onPlayback(bar, slot);
}
setInterval(follow, 40);

function updateSession() {
  if (!('mediaSession' in navigator) || !tune) return;
  navigator.mediaSession.metadata = new MediaMetadata({
    title: 'Tune ' + trackNumber() + ', ' + key,
    artist: 'Unlimited lofi music',
    album: STYLE_NAMES[tunes.get(shown)?.style || style] + ' style, ' + bpm + ' BPM',
    artwork: [{src: new URL('assets/cover.png', location.href).href, sizes: '256x256', type: 'image/png'}]
  });
  navigator.mediaSession.playbackState = playing ? 'playing' : 'paused';
}
function volume() { return Number($('#volume').value) / 100; }
async function start() {
  if (playing) return;
  playing = true;
  track('play', {style});
  updateTransport();
  if (!ctx) {
    ctx = new AudioContext({sampleRate: SR, latencyHint: 'playback'});
    master = ctx.createGain();
    master.gain.value = 0;
    master.connect(ctx.destination);
    ctx.onstatechange = () => { if (playing && ctx.state !== 'running' && !document.hidden) ctx.resume().catch(() => {}); };
    await ctx.audioWorklet.addModule('engine/worklet.js');
    node = new AudioWorkletNode(ctx, 'lofi-mixer', {numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [2]});
    node.port.onmessage = onMixer;
    const channel = new MessageChannel();
    node.port.postMessage({t: 'port', port: channel.port2}, [channel.port2]);
    worker.postMessage({t: 'port', port: channel.port1}, [channel.port1]);
    node.connect(master);
  }
  await ctx.resume().catch(() => {});
  if (playing) {
    const gain = master.gain.value;
    master.gain.cancelScheduledValues(ctx.currentTime);
    master.gain.setValueAtTime(gain, ctx.currentTime);
    master.gain.linearRampToValueAtTime(volume(), ctx.currentTime + .6);
  }
}
function pause() {
  if (!playing) return;
  playing = false;
  updateTransport();
  if (!ctx) return;
  master.gain.cancelScheduledValues(ctx.currentTime);
  master.gain.setTargetAtTime(0, ctx.currentTime, .025);
  setTimeout(() => { if (!playing) ctx.suspend().catch(() => {}); }, 160);
}
// The worker keeps the current deck audible while preparing the requested tune.
let changeBusy = false, pendingChange = null;
function requestChange(message) {
  if (changeBusy) {
    pendingChange = message;
    return;
  }
  changeBusy = true;
  pendingChange = null;
  worker.postMessage({...message, from: playing ? shown : undefined});
}
function skip() {
  requestChange({t: 'skip', style});
  debug.skips.push(performance.now());
  if (debug.skips.length > 200) debug.skips.shift();
  track('new_tune', {style});
}
function updateTransport() {
  showModelStatus();
  play.querySelector('span').textContent = playing ? 'Pause' : 'Play';
  $('#play-icon').setAttribute('d', playing ? 'M5 3h5v18H5zM14 3h5v18h-5z' : 'M6 3 21 12 6 21Z');
  if ('mediaSession' in navigator) navigator.mediaSession.playbackState = playing ? 'playing' : 'paused';
  updateSpin();
}
if ('mediaSession' in navigator) {
  for (const [action, handler] of [['play', start], ['pause', pause], ['nexttrack', skip]]) {
    try { navigator.mediaSession.setActionHandler(action, handler); } catch {}
  }
}
// 24/7: if the audio context stops (sleep and wake, a Bluetooth or device change, an
// interruption), try to resume it while the listener still wants music.
setInterval(() => { if (playing && ctx && ctx.state !== 'running' && ctx.state !== 'closed') ctx.resume().catch(() => {}); }, 5000);
navigator.mediaDevices?.addEventListener?.('devicechange', () => { if (playing && ctx) ctx.resume().catch(() => {}); });
document.addEventListener('visibilitychange', () => {
  if (!document.hidden && playing && ctx && ctx.state !== 'running') ctx.resume().catch(() => {});
});

function spin(now){
  const dt=Math.min((now-lastFrame)/1000,.08);lastFrame=now;
  speed+=((playing?32:0)-speed)*(1-Math.exp(-dt/(playing?.8:1)));
  position=(position+speed*dt)%360;disc.style.transform='rotate('+position+'deg)';
  frame=(!motion.matches&&(playing||speed>.12))?requestAnimationFrame(spin):0;
}
function updateSpin(){
  if(motion.matches){cancelAnimationFrame(frame);frame=0;speed=0;return}
  if(!frame){lastFrame=performance.now();frame=requestAnimationFrame(spin)}
}
function shareUrl() {
  const url = new URL('?seed=' + seedText + '&style=' + style, location.href);
  if (query.get('engine') === 'learned') url.searchParams.set('engine', 'learned');
  return url.href;
}
function selectStyle(kind, user) {
  if (user) requestChange({t: 'style', style: kind});
  if (user) track('style_change', {from: style, to: kind});
  style = kind;
  try { localStorage.setItem(STYLE_KEY, kind); } catch {}
  const row = $('#styles');
  document.querySelectorAll('[data-style]').forEach(b => {
    const on = b.dataset.style === kind;
    b.setAttribute('aria-pressed', String(on));
    // On phones the row scrolls sideways: centre the chosen style (the page itself never moves).
    if (on && row.scrollWidth > row.clientWidth) row.scrollTo({left: b.offsetLeft - row.offsetLeft - (row.clientWidth - b.offsetWidth) / 2, behavior: user ? 'smooth' : 'auto'});
  });
  $('#share').href = shareUrl();
  if (user) loadAmbience(kind);
}
// The style row: arrow keys move between styles, Home and End jump; Enter or Space picks.
$('#styles').addEventListener('keydown', event => {
  const buttons = [...document.querySelectorAll('[data-style]')], i = buttons.indexOf(document.activeElement);
  if (i < 0) return;
  const to = {ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: buttons.length - 1}[event.key];
  if (to === undefined) return;
  event.preventDefault();
  buttons[(to + buttons.length) % buttons.length].focus();
});
document.querySelectorAll('[data-style]').forEach(b=>b.onclick=()=>{ if (b.dataset.style !== style) selectStyle(b.dataset.style, true); });
play.onclick = () => playing ? pause() : start();
$('#skip').onclick = skip;
$('#volume').oninput = () => { if (master && playing) master.gain.setTargetAtTime(volume(), ctx.currentTime, .03); };
$('#seed').textContent = seedText;
$('#share').onclick = async event => {
  event.preventDefault();
  const url = shareUrl(), label = $('#share');
  try {
    if (navigator.share) await navigator.share({title: document.title, url});
    else { await navigator.clipboard.writeText(url); label.textContent = 'Link copied'; setTimeout(() => { label.textContent = 'Share replay'; }, 1800); }
  } catch {}
};
// Credits: a small panel that opens from the footer link and closes on Esc or an outside click.
const credits = $('#credits'), creditsToggle = $('#credits-toggle');
function showCredits(open) {
  credits.hidden = !open;
  creditsToggle.setAttribute('aria-expanded', String(open));
  if (open) credits.focus();
}
creditsToggle.onclick = event => { event.stopPropagation(); showCredits(credits.hidden); };
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !credits.hidden) { showCredits(false); creditsToggle.focus(); }
});
document.addEventListener('click', event => { if (!credits.hidden && !credits.contains(event.target)) showCredits(false); });
motion.addEventListener('change',updateSpin);
paintDisc('', '', '');
selectStyle(style, false);
updatePanel(true);
systemTheme.addEventListener('change', () => {
  setMood();
  updatePanel(true);
});
