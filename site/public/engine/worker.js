// Song scheduler. Runs the transformer and the band off the main thread and streams
// one bar of stereo PCM at a time to the mixer worklet, which it talks to directly.
import {STYLES} from './styles/index.js';
import {createRenderer, createBank, addSample} from './render.js';
import {createArrangedSong} from './arranger.js';
import {random} from './neural/random.js';

// Manual changes fade the new song in quickly; the mixer adds the silent gap before it.
const SR = 44100, AHEAD = 13 * SR, SKIP_FADE = Math.round(.15 * SR);
const engines = STYLES;
const bank = createBank();
let bankReady = false;
const NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];
const QUALITY = ['maj7', 'm7', '7', 'm7b5', '7sus4'];
const SCALE = [0, 2, 4, 5, 7, 9, 11];
const mod = (n, m) => (n % m + m) % m;
const noteName = p => NAMES[mod(p, 12)] + (Math.floor(p / 12) - 1);

let meta, bytes, learned = null, allLearned = false, station, style, mixer, outbox = [], tuneIndex = 0, spare, busy = false, slowdown = 0;
// Test hook: ?slowdown=ms adds that much work per bar, to stand in for a slow device.
const stall = ms => { const end = performance.now() + ms; while (performance.now() < end); };
const decks = new Map();

// Tune n of a station always gets the same seed, so ?seed= replays the same run of songs.
function tuneSeed(n, attempt) {
  const r = random(station ^ Math.imul(n, 0x9e3779b9) ^ Math.imul(attempt, 0x85ebca6b));
  r();
  return 1 + Math.floor(r() * 0x7ffffffe);
}
function keyName({shift, mode}) {
  return mode === 'major' ? NAMES[mod(shift, 12)] + ' major' : NAMES[mod(shift + 9, 12)] + ' minor';
}
let previous = null;
function makeTune(kind) {
  const n = tuneIndex + 1, engine = engines[kind];
  let seed, plan;
  // A new song always moves to a new key and a new tempo.
  for (let attempt = 0; attempt < 64; attempt++) {
    seed = tuneSeed(n, attempt);
    plan = engine.songPlan(seed);
    if (!previous || (plan.bpm !== previous.bpm && keyName(plan) !== previous.key)) break;
  }
  const r = random(seed ^ 0x51ed27);
  // Each song follows its own plan (sections, tracks, transitions); the next song starts
  // as this one's outro ends.
  const models = allLearned || kind === 'jazzhop' ? learned : null;
  const song = createArrangedSong(engine, meta, bytes, seed, kind, undefined, (section, of) => postMessage({t: 'writing', tune: n, section, of}), models);
  const fade = 6 + 2 * r(), duration = Math.max(30, song.totalBars * 4 * 60 / song.bpm - fade);
  return {
    n, seed, style: kind, song, render: createRenderer(song, seed, bank, kind),
    info: {tune: n, seed, style: kind, engine: models ? 'learned' : 'preset', key: keyName(plan), bpm: plan.bpm, shift: plan.shift, mode: plan.mode,
      duration, fade, sections: song.plan.sections.map(s => [s.type, s.bars])}
  };
}
function takeTune(kind) {
  const tune = spare && spare.style === kind && spare.n === tuneIndex + 1 ? spare : makeTune(kind);
  spare = null;
  tuneIndex = tune.n;
  previous = tune.info;
  return tune;
}

function send(message, transfer = []) {
  if (mixer) mixer.postMessage(message, transfer);
  else outbox.push([message, transfer]);
}
function degreeOf(pitch, shift, mode) {
  const tonic = mod(shift + (mode === 'major' ? 0 : 9), 12);
  let base = 60 - mod(60 - tonic, 12), steps = 0;
  for (let p = base; p < pitch; p++) if (SCALE.includes(mod(p - shift, 12))) steps++;
  while (steps > 7) steps -= 7;
  return Math.max(0, steps);
}

function openDeck(kind, reason) {
  const tune = takeTune(kind);
  const deck = {id: tune.n, ...tune, rendered: 0, read: 0, bars: 0, fadeAt: 0, stopAt: Infinity, ended: false};
  decks.set(deck.id, deck);
  postMessage({t: 'tune', deck: deck.id, now: reason !== 'auto', info: deck.info});
  return deck;
}
function renderBar(deck) {
  const t0 = performance.now();
  const {left, right, events} = deck.render();
  if (slowdown) stall(slowdown);
  const ms = performance.now() - t0;
  const {info: bar} = events, {shift, mode} = deck.info;
  // The melody tokens come from whichever composer wrote the song.
  const m = deck.info.engine === 'learned' ? learned.composer.meta : meta;
  const tokens = bar.tokens.map(t => t === m.M0 ? '·' : t === m.M0 + 1 ? '~' : noteName(m.LO + t - m.M0 - 2 + shift));
  const notes = events.filter(e => e.type === 'felt').map(e => ({
    slot: e.slot, pitch: e.pitch, degree: degreeOf(e.pitch, shift, mode)
  }));
  const start = deck.rendered, frames = left.length;
  deck.rendered += frames;
  deck.bars++;
  send({t: 'chunk', deck: deck.id, left, right}, [left.buffer, right.buffer]);
  postMessage({t: 'bar', deck: deck.id, bar: deck.bars, start, frames, ms,
    chord: (bar.chords || [[bar.root, bar.quality]]).map(([r, q]) => NAMES[mod(r + shift, 12)] + QUALITY[q]).join(' '), tokens, notes,
    section: bar.sectionType ?? null});
  if (deck.rendered >= deck.stopAt && !deck.ended) {
    deck.ended = true;
    send({t: 'end', deck: deck.id});
  }
}
function scheduleAutoChange(deck) {
  if (deck.fadeAt || deck.rendered < deck.info.duration * SR) return;
  const next = openDeck(style, 'auto');
  const frames = Math.round(deck.info.fade * SR);
  deck.fadeAt = deck.rendered;
  deck.stopAt = deck.fadeAt + frames;
  renderBar(next);
  send({t: 'fade', from: deck.id, to: next.id, at: deck.fadeAt, frames});
}
// Fill the deck that is closest to running dry, one bar per task so messages keep flowing.
function pump() {
  if (busy || !meta || !bankReady) return;
  let target = null, least = Infinity;
  for (const deck of decks.values()) {
    if (deck.ended) continue;
    const ahead = deck.rendered - deck.read;
    if (ahead < AHEAD && ahead < least) { least = ahead; target = deck; }
  }
  if (!target) {
    if (!spare) {
      busy = true;
      setTimeout(() => { busy = false; if (!spare) spare = makeTune(style); pump(); }, 0);
    }
    return;
  }
  busy = true;
  setTimeout(() => {
    busy = false;
    if (decks.has(target.id) && !target.ended) {
      if (!target.fadeAt) scheduleAutoChange(target);
      renderBar(target);
    }
    pump();
  }, 0);
}
function changeNow(kind) {
  style = kind;
  if (!mixer) {
    // Nothing has played yet: replace the preloaded song outright.
    outbox = [];
    decks.clear();
    const deck = openDeck(kind, 'skip');
    renderBar(deck);
    pump();
    return;
  }
  // Everything queued already covers the fade-out. Songs that were only waiting
  // for a scheduled crossfade are discarded by the mixer.
  for (const other of decks.values()) if (!other.ended) { other.ended = true; send({t: 'end', deck: other.id}); }
  const deck = openDeck(kind, 'skip');
  // Start with about 5 s of the new song queued (the 1 s gap covers the time this takes);
  // the pump fills the rest of the queue right after.
  while (deck.rendered < 5 * SR) renderBar(deck);
  send({t: 'fade', to: deck.id, at: -1, frames: SKIP_FADE});
  pump();
}

onmessage = async ({data}) => {
  if (data.t === 'init') {
    station = data.seed >>> 0;
    style = data.style;
    slowdown = data.slowdown ?? slowdown;
    ({meta, bytes} = data);
    learned = data.learned || null;
    allLearned = data.allLearned;
    if (bankReady) openDeck(style, 'first');
    pump();
  } else if (data.t === 'sample') {
    addSample(bank, data.sample);
  } else if (data.t === 'bank-ready') {
    bankReady = true;
    if (meta && !decks.size) openDeck(style, 'first');
    pump();
  } else if (data.t === 'port') {
    mixer = data.port;
    mixer.onmessage = ({data: report}) => {
      if (report.t === 'pos') {
        for (const [id, read] of report.decks) { const deck = decks.get(id); if (deck) deck.read = read; }
        pump();
      } else if (report.t === 'dropped') {
        decks.delete(report.deck);
        postMessage({t: 'dropped', deck: report.deck});
        pump();
      }
    };
    for (const [message, transfer] of outbox) mixer.postMessage(message, transfer);
    outbox = [];
  } else if (data.t === 'skip') {
    if (meta && bankReady && decks.size) changeNow(style);
  } else if (data.t === 'style') {
    if (meta && bankReady && decks.size && data.style !== style) changeNow(data.style);
    else style = data.style;
  }
};
