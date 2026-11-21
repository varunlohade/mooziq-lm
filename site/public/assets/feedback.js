// Keep an eight-second history of audible deck positions, including both sides of a crossfade.
// Audio-context time stops during a pause. Render-ahead and the selected UI deck can be ahead of the listener.
const heard = [], feedbackOpen = $('#feedback-open'), feedbackDialog = $('#feedback-dialog');
const feedbackForm = $('#feedback-form');
let pendingFeedback, previousReportTime;
function rememberPlayback(data) {
  const dt = previousReportTime == null ? 0 : Math.min(.1, data.time - previousReportTime);
  previousReportTime = data.time;
  while (heard.length && heard[0].time < data.time - 8) heard.shift();
  for (const [id, pos, gain, , state] of data.decks) {
    const info = tunes.get(id);
    if (state !== 'play' || gain <= .001 || !info || !pos || !dt || !volume()) continue;
    const toMs = Math.round(pos * 1000 / SR);
    const fromMs = Math.max(0, toMs - Math.round(dt * 1000));
    if (fromMs >= toMs) continue;
    const bar = written.get(id)?.find(b => pos > b.start && pos <= b.start + b.frames);
    heard.push({time:data.time, id, info, fromMs, toMs, bar});
  }
  feedbackOpen.disabled = !heard.length;
}

function inspectPassage(bars, fromMs, toMs) {
  const observations = [];
  const lines = new Map();
  let previousSection;
  for (const bar of bars) {
    if (!bar) continue;
    const at = Math.round(bar.start * 1000 / SR);
    if (at < toMs) {
      observations.push({kind:'bar', at:Math.max(at,fromMs), detail:bar.section+' '+bar.chord});
      if (previousSection && previousSection !== bar.section) observations.push({kind:'transition',at:Math.max(at,fromMs),detail:previousSection+' → '+bar.section});
      previousSection = bar.section;
    }
    for (const [part, time, pitch] of bar.attacks || []) {
      if (time < fromMs || time > toMs) continue;
      if (!lines.has(part)) lines.set(part, []);
      lines.get(part).push([time,pitch]);
    }
  }
  for (const [part, line] of lines) {
    line.sort((a,b) => a[0]-b[0]);
    if (part === 'felt') {
      for (let i=2;i<line.length;i++) {
        if (line[i][1] === line[i-1][1] && line[i][1] === line[i-2][1])
          observations.push({kind:'repeated lead',at:line[i][0],detail:'Three attacks on MIDI '+line[i][1]});
      }
    }
    for (let i=1;i+1<line.length;i++) {
      const [at,pitch] = line[i], prev = line[i-1], next = line[i+1];
      const depth = Math.min(prev[1],next[1])-pitch;
      if (depth >= 7 && at-prev[0] >= 120 && next[0]-at >= 120)
        observations.push({kind:'low dip', at, detail:part+' '+prev[1]+' → '+pitch+' → '+next[1]});
    }
  }
  return observations.slice(0,16);
}

feedbackOpen.onclick = () => {
  if (!heard.length) return;
  const groups = new Map();
  for (const tick of heard) {
    let group = groups.get(tick.id);
    if (!group) {
      const {style,seed,engine,tune} = tick.info;
      group = {style,seed,engine,tune,fromMs:tick.fromMs,toMs:tick.toMs,bars:new Set()};
      groups.set(tick.id, group);
    }
    group.toMs = tick.toMs;
    group.bars.add(tick.bar);
  }
  const passages = Array.from(groups.values(), ({bars,...p}) => {
    p.fromMs = Math.max(p.fromMs,p.toMs-8000);
    return {...p,observations:inspectPassage(bars,p.fromMs,p.toMs)};
  });
  pendingFeedback = {id:crypto.randomUUID(), passages};
  const seconds = ms => (ms/1000).toFixed(1)+'s';
  $('#feedback-passage').textContent = passages.map(p => STYLE_NAMES[p.style]+' · Tune '+p.tune+' · '+seconds(p.fromMs)+'–'+seconds(p.toMs)).join(' / ');
  feedbackForm.reset();
  $('#feedback-status').textContent = '';
  $('#feedback-sent').textContent = '';
  feedbackDialog.showModal();
};
$('#feedback-close').onclick = () => feedbackDialog.close();
feedbackForm.onsubmit = event => {
  event.preventDefault();
  if (!pendingFeedback) return;
  const report = {...pendingFeedback, reason:feedbackForm.elements.reason.value, comment:feedbackForm.elements.comment.value};
  const url = URL.createObjectURL(new Blob([JSON.stringify(report)], {type:'application/json'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'mooziq-feedback.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  feedbackDialog.close();
  pendingFeedback = null;
  $('#feedback-sent').textContent = 'Feedback exported.';
};
