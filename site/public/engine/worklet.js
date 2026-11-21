// Two-deck mixer on the audio thread. Each deck is one song: a queue of PCM bars from the
// worker. Each change fades out, leaves a second of silence, then fades in.
const HALF_PI = Math.PI / 2, REPORT_BLOCKS = 16;
// Final ceiling: transparent below -4 dBFS, then a soft knee that never passes -3.3 dBFS,
// tools/render.mjs applies the same curve.
const KNEE = 10 ** (-4 / 20), TOP = 10 ** (-3.3 / 20);
const ceiling = x => { const a = Math.abs(x); return a <= KNEE ? x : Math.sign(x) * (KNEE + (TOP - KNEE) * Math.tanh((a - KNEE) / (TOP - KNEE))); };

class Mixer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.decks = new Map();
    this.blocks = 0;
    this.nextStartFrame = 0;
    this.underrun = 0;
    this.minAhead = Infinity;
    this.steadyMinAhead = Infinity;
    this.port.onmessage = ({data}) => {
      if (data.t !== 'port') return;
      this.link = data.port;
      this.link.onmessage = ({data: message}) => this.handle(message);
    };
  }
  deck(id) {
    let deck = this.decks.get(id);
    if (!deck) {
      const first = this.decks.size === 0;
      // A song that starts with nothing playing waits until a few seconds are queued, so a
      // slow first bar cannot leave it short.
      deck = {id, chunks: [], off: 0, pos: 0, queued: 0, gain: 0, state: 'wait', autostart: first,
        ended: false, fadeIn: null, fadeOut: null, sched: null};
      this.decks.set(id, deck);
    }
    return deck;
  }
  handle(message) {
    if (message.t === 'chunk') {
      const deck = this.deck(message.deck);
      deck.beatFrames = 60 * sampleRate / message.bpm;
      deck.chunks.push(message);
      deck.queued += message.left.length;
    } else if (message.t === 'end') {
      const deck = this.decks.get(message.deck);
      if (deck) deck.ended = true;
    } else if (message.t === 'fade') {
      if (message.at >= 0) {
        const from = this.decks.get(message.from);
        if (from) {
          this.deck(message.to).autostart = false;
          from.sched = {to: message.to, at: message.at, frames: message.frames};
        }
        else this.drop(this.decks.get(message.to));
        return;
      }
      const next = this.deck(message.to);
      next.autostart = false;
      let lead;
      for (const deck of [...this.decks.values()]) {
        if (deck === next) continue;
        deck.sched = null;
        if (deck.state === 'wait') this.drop(deck);
        else if (!lead || deck.gain > lead.gain) lead = deck;
      }
      // Start the outgoing fade on the beat if enough samples are queued.
      // Take over immediately if the station is just starting or running dry.
      if (lead?.fadeOut) {
        this.queueEntrance(next, message.frames);
        return;
      }
      if (lead) {
        const delay = Math.ceil(lead.pos / lead.beatFrames) * lead.beatFrames - lead.pos;
        const frames = Math.min(message.frames, Math.max(1, lead.queued));
        const at = lead.pos + (lead.queued >= delay + frames ? Math.round(delay) : 0);
        lead.sched = {to: next.id, at, frames};
        lead.fadeOut = null;
        for (const deck of this.decks.values()) {
          if (deck !== lead && deck.state === 'play' && !deck.fadeOut) this.fadeOut(deck, frames);
        }
        if (lead.queued) return;
        lead.sched = null;
        this.drop(lead);
      }
      this.queueEntrance(next, message.frames);
    }
  }
  queueEntrance(deck, frames) {
    deck.autostart = false;
    deck.entrance = {at: this.nextStartFrame, frames: Math.min(frames, Math.round(2.5 * sampleRate))};
  }
  fadeIn(deck, frames, offset = 0) {
    deck.startOffset = offset;
    deck.state = 'play';
    deck.fadeIn = {n: 0, len: frames};
    deck.gain = 0;
    this.port.postMessage({t: 'fade', to: deck.id, time: currentTime});
  }
  fadeOut(deck, frames) {
    deck.fadeIn = null;
    deck.fadeOut = {n: 0, len: frames, from: deck.gain};
  }
  drop(deck) {
    if (!deck || !this.decks.delete(deck.id)) return;
    this.link?.postMessage({t: 'dropped', deck: deck.id});
    this.port.postMessage({t: 'dropped', deck: deck.id});
  }
  process(inputs, outputs) {
    const left = outputs[0][0], right = outputs[0][1] || outputs[0][0], n = left.length;
    left.fill(0);
    if (right !== left) right.fill(0);
    let dry = 0;
    for (const deck of this.decks.values())
      if (deck.autostart && deck.state === 'wait' && (deck.queued >= 5 * sampleRate || deck.ended)) { deck.autostart = false; this.fadeIn(deck, Math.round(2.5 * sampleRate)); }
    for (const deck of [...this.decks.values()]) {
      if (deck.entrance && deck.queued && deck.entrance.at < currentFrame + n) {
        this.fadeIn(deck, deck.entrance.frames, Math.max(0, deck.entrance.at - currentFrame));
        deck.entrance = null;
      }
      if (deck.state !== 'play') continue;
      if (!deck.ended) {
        this.minAhead = Math.min(this.minAhead, deck.queued);
        if (deck.pos > 15 * sampleRate && !deck.fadeIn) this.steadyMinAhead = Math.min(this.steadyMinAhead, deck.queued);
      }
      const startOffset = deck.startOffset;
      deck.startOffset = 0;
      for (let i = startOffset; i < n; i++) {
        if (deck.sched && deck.pos >= deck.sched.at) {
          const {to, frames} = deck.sched;
          deck.sched = null;
          const next = this.decks.get(to);
          if (next) {
            this.fadeOut(deck, frames);
            this.nextStartFrame = currentFrame + i + frames + Math.round(sampleRate);
            this.queueEntrance(next, frames);
          }
        }
        let gain = deck.gain;
        if (deck.fadeIn) {
          gain = Math.sin(HALF_PI * deck.fadeIn.n / deck.fadeIn.len);
          if (++deck.fadeIn.n >= deck.fadeIn.len) { deck.fadeIn = null; gain = 1; }
        } else if (deck.fadeOut) {
          gain = deck.fadeOut.from * Math.cos(HALF_PI * deck.fadeOut.n / deck.fadeOut.len);
          if (++deck.fadeOut.n >= deck.fadeOut.len) { this.drop(deck); break; }
        }
        deck.gain = gain;
        const chunk = deck.chunks[0];
        if (!chunk) {
          if (deck.ended) { this.drop(deck); break; }
          if (gain > 0) dry = Math.max(dry, n - i);
          break;
        }
        left[i] += chunk.left[deck.off] * gain;
        right[i] += chunk.right[deck.off] * gain;
        deck.pos++;
        deck.queued--;
        if (++deck.off >= chunk.left.length) { deck.chunks.shift(); deck.off = 0; }
      }
    }
    for (let i = 0; i < n; i++) { left[i] = ceiling(left[i]); if (right !== left) right[i] = ceiling(right[i]); }
    this.underrun += dry;
    if (++this.blocks % REPORT_BLOCKS === 0) {
      const decks = [...this.decks.values()].map(d => [d.id, d.pos, d.gain, d.queued, d.state]);
      this.link?.postMessage({t: 'pos', decks: decks.map(d => [d[0], d[1]])});
      this.port.postMessage({t: 'pos', time: currentTime, decks, underrun: this.underrun,
        minAhead: this.minAhead / sampleRate, steadyMinAhead: this.steadyMinAhead / sampleRate});
    }
    return true;
  }
}
registerProcessor('lofi-mixer', Mixer);
