// DooDL front end: upload, outline, track, check, fix.
import * as api from './api.js';
import { Stage } from './stage.js';

const $ = (id) => document.getElementById(id);
const els = Object.fromEntries(
  [
    'workbench', 'stage', 'dropzone', 'limits', 'choose', 'sample', 'file', 'upload-note', 'viewport', 'frame', 'layers', 'ink',
    'result-video', 'stage-progress', 'stage-progress-text', 'stage-progress-fill', 'hint', 'timeline', 'strip',
    'marks', 'scrub', 'frame-label', 'objects', 'objects-empty', 'undo', 'track', 'progress', 'progress-text',
    'progress-fill', 'cancel', 'result', 'result-summary', 'fix', 'downloads', 'new-video',
    'toast', 'toast-text', 'toast-close',
  ].map((id) => [id, $(id)]),
);

const DOWNLOAD_LABELS = {
  'overlay.mp4': 'Tracked video (MP4)',
  'cutout.mp4': 'Objects on black (MP4)',
  'masks.zip': 'Mask images (ZIP)',
  'annotations.json': 'Annotations (JSON)',
};

const state = {
  health: null,
  session: null,
  frame: 0,
  tool: 'outline',
  selected: null,
  previews: new Map(), // frame -> Map(objectId -> base64 PNG)
  marks: new Map(), // "objectId:frame" -> [{x, y, label}]
  phase: 'empty', // empty | edit | tracking | result
  busy: false,
  jobId: null,
  poll: null,
};

const stage = new Stage(
  { stage: els.stage, viewport: els.viewport, frame: els.frame, layers: els.layers, ink: els.ink, video: els['result-video'] },
  {
    getTool: () => (state.phase === 'edit' && !state.busy ? state.tool : null),
    onOutline: (points) => commitOutline(points),
    onPoint: (point, label) => commitPoint(point, label),
    onTooSmall: () => toast('That outline is too small. Draw it all the way around the object.'),
  },
);

// ------------------------------------------------------------------ helpers

let toastTimer = 0;
function toast(message) {
  els['toast-text'].textContent = message;
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (els.toast.hidden = true), 9000);
}
els['toast-close'].addEventListener('click', () => (els.toast.hidden = true));

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const dataUrl = (b64) => `data:image/png;base64,${b64}`;
const sessionPath = (suffix = '') => `/api/sessions/${state.session.id}${suffix}`;
const selectedObject = () => state.session?.objects.find((o) => o.id === state.selected) ?? null;
const marksKey = (objectId, frame) => `${objectId}:${frame}`;

async function guarded(work, { onError } = {}) {
  if (state.busy) return;
  state.busy = true;
  render();
  try {
    await work();
  } catch (error) {
    if (onError) onError(error);
    toast(error instanceof api.ApiError ? error.message : 'Something went wrong. Try again.');
    if (!(error instanceof api.ApiError)) console.error(error);
  } finally {
    state.busy = false;
    render();
  }
}

// ------------------------------------------------------------------- render

function hintText() {
  const s = state.session;
  if (!s) return '';
  const notices = [];
  if (s.trimmed && state.phase === 'edit' && !s.result) {
    notices.push(`Only the first ${s.duration_s} s were loaded (the limit is ${state.health.limits.max_duration_s} s).`);
  }
  if (state.phase === 'tracking') return 'Tracking. You can leave this tab open.';
  if (state.phase === 'result') return 'Play the result, then use Fix a frame if a mask drifted.';
  if (s.result?.stale) return 'Objects changed since the last run. Press Track again to update the video.';
  const obj = selectedObject();
  if (s.result && state.tool === 'outline') {
    return 'Select an object, then use Add point or Remove point to correct this frame. Press Track again when done.';
  }
  if (state.tool === 'add' && obj) notices.push(`Click the part of ${obj.name} the mask missed.`);
  else if (state.tool === 'remove' && obj) notices.push(`Click the area that should not be part of ${obj.name}.`);
  else if (!s.objects.length) notices.push('Scrub to a frame where the object is clear, then draw a loose outline around it.');
  else notices.push('Outline another object, or press Track.');
  return notices.join(' ');
}

function renderObjects() {
  const s = state.session;
  const list = els.objects;
  list.replaceChildren();
  els['objects-empty'].hidden = Boolean(s && s.objects.length);
  for (const obj of s?.objects ?? []) {
    const item = document.createElement('li');
    const select = document.createElement('button');
    select.type = 'button';
    select.className = 'obj';
    select.setAttribute('aria-pressed', String(obj.id === state.selected));
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = obj.color;
    const label = document.createElement('span');
    const frames = obj.prompt_frames;
    label.innerHTML = '<span></span><small></small>';
    label.firstChild.textContent = obj.name;
    label.lastChild.textContent = `Outlined on ${frames.length === 1 ? `frame ${frames[0] + 1}` : plural(frames.length, 'frame')}`;
    select.append(swatch, label);
    select.addEventListener('click', () => {
      state.selected = obj.id;
      render();
      drawMarks();
    });
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'icon';
    remove.textContent = 'Remove';
    remove.setAttribute('aria-label', `Remove ${obj.name}`);
    remove.addEventListener('click', () => removeObject(obj.id));
    item.append(select, remove);
    list.append(item);
  }
}

function renderTimeline() {
  const s = state.session;
  if (!s) return;
  els.scrub.max = String(s.frame_count - 1);
  els.scrub.value = String(state.frame);
  els['frame-label'].textContent =
    `Frame ${state.frame + 1} of ${s.frame_count}, ${(state.frame / s.fps).toFixed(1)} s`;
  els.marks.replaceChildren();
  for (const obj of s.objects) {
    for (const f of obj.prompt_frames) {
      const mark = document.createElement('span');
      mark.className = 'mark';
      mark.style.left = `${(f / Math.max(s.frame_count - 1, 1)) * 100}%`;
      mark.style.background = obj.color;
      els.marks.append(mark);
    }
  }
}

function render() {
  const s = state.session;
  const phase = state.phase;
  els.stage.dataset.phase = phase;
  els.workbench.dataset.phase = phase;
  els.viewport.hidden = !s || phase === 'empty' || phase === 'result';
  els['result-video'].hidden = phase !== 'result';
  els.timeline.hidden = !s || phase === 'empty' || phase === 'result';
  els['new-video'].hidden = !s;
  els['stage-progress'].hidden = phase !== 'tracking';

  const editing = phase === 'edit';
  const hasSelected = Boolean(selectedObject());
  for (const radio of document.querySelectorAll('input[name="tool"]')) {
    radio.disabled = !editing || (radio.value !== 'outline' && !hasSelected);
    radio.checked = radio.value === state.tool;
  }
  const marksHere = hasSelected ? state.marks.get(marksKey(state.selected, state.frame))?.length ?? 0 : 0;
  els.undo.disabled = !editing || state.busy || marksHere === 0;

  const count = s?.objects.length ?? 0;
  const noun = count ? plural(count, 'object') : 'objects';
  els.track.textContent = s?.result ? `Track ${noun} again` : `Track ${noun}`;
  els.track.disabled = !editing || state.busy || count === 0;
  els.track.hidden = phase === 'tracking' || phase === 'result';

  els.progress.hidden = phase !== 'tracking';
  els.result.hidden = phase !== 'result';
  els.hint.textContent = hintText();
  els['choose'].disabled = els['sample'].disabled = state.busy;
  renderObjects();
  if (s) renderTimeline();
}

function renderLayers(animateObject = null) {
  const s = state.session;
  if (!s) return;
  const entries = [];
  const here = state.previews.get(state.frame);
  if (here && here.size) {
    for (const [objectId, b64] of here) entries.push({ src: dataUrl(b64), snap: objectId === animateObject });
  } else if (s.result) {
    entries.push({ src: api.url(sessionPath(`/overlay/${state.frame}.png`), { v: s.result.version }) });
  }
  stage.setLayers(entries);
}

function drawMarks() {
  if (!state.session || state.selected === null) return stage.clearInk();
  stage.drawMarks(state.marks.get(marksKey(state.selected, state.frame)) ?? []);
}

// ------------------------------------------------------------------- frames

let pendingFrame = null;
function setFrame(index) {
  const s = state.session;
  if (!s) return;
  const next = Math.min(Math.max(0, Math.round(index)), s.frame_count - 1);
  if (pendingFrame !== null) {
    pendingFrame = next;
    return;
  }
  pendingFrame = next;
  requestAnimationFrame(() => {
    state.frame = pendingFrame;
    pendingFrame = null;
    els.frame.src = api.url(sessionPath(`/frames/${state.frame}`));
    renderLayers();
    drawMarks();
    render();
  });
}
els.scrub.addEventListener('input', () => setFrame(Number(els.scrub.value)));
document.addEventListener('keydown', (event) => {
  if (state.phase !== 'edit' || event.target === els.scrub || event.target.closest('input, textarea')) return;
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    const step = (event.shiftKey ? 10 : 1) * (event.key === 'ArrowLeft' ? -1 : 1);
    setFrame(state.frame + step);
    event.preventDefault();
  }
});

function buildStrip() {
  const s = state.session;
  els.strip.replaceChildren();
  const count = Math.min(12, s.frame_count);
  for (let i = 0; i < count; i++) {
    const index = count === 1 ? 0 : Math.round((i * (s.frame_count - 1)) / (count - 1));
    const img = document.createElement('img');
    img.alt = '';
    img.loading = 'lazy';
    img.src = api.url(sessionPath(`/frames/${index}`), { w: 160 });
    els.strip.append(img);
  }
}

// ------------------------------------------------------------------ session

function loadSession(session) {
  state.session = session;
  state.previews.clear();
  state.marks.clear();
  state.selected = null;
  state.tool = 'outline';
  state.phase = session.result ? 'result' : 'edit';
  state.frame = 0;
  els['upload-note'].textContent = '';
  stage.setSource(session.width, session.height);
  buildStrip();
  setFrame(0);
  render();
}

function resetToEmpty() {
  clearInterval(state.poll);
  const old = state.session;
  if (old) api.del(`/api/sessions/${old.id}`).catch(() => {});
  Object.assign(state, { session: null, previews: new Map(), marks: new Map(), selected: null, phase: 'empty', jobId: null, tool: 'outline' });
  els['result-video'].removeAttribute('src');
  els['result-video'].load();
  els['upload-note'].textContent = '';
  stage.clearInk();
  stage.setLayers([]);
  render();
}
els['new-video'].addEventListener('click', resetToEmpty);

// ------------------------------------------------------------------- upload

function validateFile(file) {
  const limits = state.health.limits;
  const extension = `.${file.name.split('.').pop().toLowerCase()}`;
  if (!limits.extensions.includes(extension)) {
    return `That file type is not supported. Use ${limits.extensions.map((e) => e.slice(1).toUpperCase()).join(', ')}.`;
  }
  if (file.size > limits.max_upload_mb * 1024 * 1024) {
    return `That file is larger than the ${limits.max_upload_mb} MB limit.`;
  }
  return null;
}

async function handleFile(file) {
  const problem = validateFile(file);
  if (problem) return toast(problem);
  await guarded(async () => {
    els['upload-note'].textContent = 'Uploading 0%';
    const session = await api.upload(file, (fraction) => {
      els['upload-note'].textContent = fraction >= 1 ? 'Preparing frames' : `Uploading ${Math.round(fraction * 100)}%`;
    });
    loadSession(session);
  }, { onError: () => (els['upload-note'].textContent = '') });
}

els.choose.addEventListener('click', () => els.file.click());
els.file.addEventListener('change', () => {
  if (els.file.files[0]) handleFile(els.file.files[0]);
  els.file.value = '';
});
els.sample.addEventListener('click', () =>
  guarded(async () => {
    els['upload-note'].textContent = 'Preparing the sample clip';
    loadSession(await api.post('/api/samples/synthetic'));
  }, { onError: () => (els['upload-note'].textContent = '') }),
);
for (const type of ['dragenter', 'dragover']) {
  els.stage.addEventListener(type, (event) => {
    if (state.phase !== 'empty') return;
    event.preventDefault();
    els.dropzone.dataset.over = 'true';
  });
}
for (const type of ['dragleave', 'drop']) {
  els.stage.addEventListener(type, () => (els.dropzone.dataset.over = 'false'));
}
els.stage.addEventListener('drop', (event) => {
  if (state.phase !== 'empty') return;
  event.preventDefault();
  const file = event.dataTransfer?.files?.[0];
  if (file) handleFile(file);
});

// ------------------------------------------------------------------ objects

function mergeObject(object) {
  const objects = state.session.objects;
  const index = objects.findIndex((o) => o.id === object.id);
  if (index === -1) objects.push(object);
  else objects[index] = object;
  if (state.session.result) state.session.result.stale = true;
}

function rememberPreview(objectId, frame, b64) {
  if (!state.previews.has(frame)) state.previews.set(frame, new Map());
  state.previews.get(frame).set(objectId, b64);
}

function commitOutline(points) {
  guarded(
    async () => {
      const result = await api.post(sessionPath('/objects'), {
        frame_idx: state.frame,
        polygon: points.map((p) => [p.x, p.y]),
      });
      mergeObject(result.object);
      rememberPreview(result.object.id, result.frame_idx, result.png_b64);
      state.selected = result.object.id;
      renderLayers(result.object.id);
      stage.fadeDoodle();
    },
    { onError: () => stage.clearInk() },
  );
}

function commitPoint(point, label) {
  const obj = selectedObject();
  if (!obj) return;
  guarded(async () => {
    const result = await api.post(sessionPath(`/objects/${obj.id}/clicks`), {
      frame_idx: state.frame,
      points: [[point.x, point.y]],
      labels: [label],
    });
    const key = marksKey(obj.id, state.frame);
    state.marks.set(key, [...(state.marks.get(key) ?? []), { ...point, label }]);
    mergeObject(result.object);
    rememberPreview(obj.id, result.frame_idx, result.png_b64);
    renderLayers();
    drawMarks();
  });
}

els.undo.addEventListener('click', () => {
  const obj = selectedObject();
  if (!obj) return;
  guarded(async () => {
    const result = await api.post(sessionPath(`/objects/${obj.id}/clicks`), { frame_idx: state.frame, undo: true });
    const key = marksKey(obj.id, state.frame);
    state.marks.set(key, (state.marks.get(key) ?? []).slice(0, -1));
    mergeObject(result.object);
    rememberPreview(obj.id, result.frame_idx, result.png_b64);
    renderLayers();
    drawMarks();
  });
});

function removeObject(objectId) {
  guarded(async () => {
    await api.del(sessionPath(`/objects/${objectId}`));
    const s = state.session;
    s.objects = s.objects.filter((o) => o.id !== objectId);
    for (const previews of state.previews.values()) previews.delete(objectId);
    if (state.selected === objectId) state.selected = null;
    if (!s.objects.length) {
      s.result = null;
      state.tool = 'outline';
    } else if (s.result) {
      s.result.stale = true;
    }
    renderLayers();
    drawMarks();
  });
}

for (const radio of document.querySelectorAll('input[name="tool"]')) {
  radio.addEventListener('change', () => {
    state.tool = radio.value;
    render();
  });
}

// ----------------------------------------------------------------- tracking

els.track.addEventListener('click', () =>
  guarded(async () => {
    const job = await api.post(sessionPath('/track'));
    state.jobId = job.id;
    state.phase = 'tracking';
    stage.clearInk();
    updateProgress(job);
    clearInterval(state.poll);
    state.poll = setInterval(pollJob, 600);
  }),
);

els.cancel.addEventListener('click', async () => {
  if (!state.jobId) return;
  try {
    await api.post(`/api/jobs/${state.jobId}/cancel`);
  } catch (error) {
    toast(error.message);
  }
});

function updateProgress(job) {
  const percent = Math.round(job.progress * 100);
  els['progress-text'].textContent = job.message;
  els['progress-fill'].style.width = `${percent}%`;
  els.progress.querySelector('[role="progressbar"]').setAttribute('aria-valuenow', String(percent));
  els['stage-progress-text'].textContent = `${job.message}, ${percent}%`;
  els['stage-progress-fill'].style.width = `${percent}%`;
  render();
}

let pollFailures = 0;
async function pollJob() {
  try {
    const job = await api.get(`/api/jobs/${state.jobId}`);
    pollFailures = 0;
    updateProgress(job);
    if (job.status === 'done') await finishTracking();
    else if (job.status === 'failed') endTracking(job.error || 'Tracking failed.');
    else if (job.status === 'cancelled') endTracking(null, 'Tracking cancelled. Your outlines are still here.');
  } catch (error) {
    if (++pollFailures >= 4) endTracking(error.message);
  }
}

function endTracking(errorMessage, info) {
  clearInterval(state.poll);
  state.phase = 'edit';
  state.jobId = null;
  if (errorMessage) toast(errorMessage);
  else if (info) toast(info);
  render();
}

async function finishTracking() {
  clearInterval(state.poll);
  const fresh = await api.get(sessionPath());
  state.session = fresh;
  state.previews.clear();
  state.phase = 'result';
  const result = fresh.result;
  const parts = [`Tracked ${plural(result.frames, 'frame')} in ${result.elapsed_s.toFixed(1)} s (${result.fps.toFixed(1)} frames per second)`];
  if (result.peak_gpu_gb) parts.push(`peak GPU memory ${result.peak_gpu_gb} GB`);
  els['result-summary'].textContent = `${parts.join(', ')}.`;
  els.downloads.replaceChildren();
  for (const [name, path] of Object.entries(result.downloads)) {
    const item = document.createElement('li');
    const link = document.createElement('a');
    link.href = api.url(path, { save: 1 });
    link.textContent = DOWNLOAD_LABELS[name] ?? name;
    link.setAttribute('download', '');
    item.append(link);
    els.downloads.append(item);
  }
  const video = els['result-video'];
  video.src = api.url(result.downloads['overlay.mp4'], { v: result.version });
  stage.fit();
  render();
  video.play().catch(() => {});
}

els.fix.addEventListener('click', () => {
  const video = els['result-video'];
  const fps = state.session.fps;
  const frame = Math.round(video.currentTime * fps);
  video.pause();
  state.phase = 'edit';
  stage.fit();
  render();
  setFrame(frame);
});

// --------------------------------------------------------------------- boot

async function boot() {
  try {
    state.health = await api.get('/api/health');
  } catch {
    els['upload-note'].textContent = 'Could not reach the DooDL server.';
    els.choose.disabled = els.sample.disabled = true;
    return;
  }
  const h = state.health;
  const order = ['.mp4', '.mov', '.webm', '.mkv', '.m4v', '.avi'];
  const shown = [...h.limits.extensions].sort((a, b) => order.indexOf(a) - order.indexOf(b));
  els.limits.textContent =
    `${shown.map((e) => e.slice(1).toUpperCase()).join(', ')}. ` +
    `Up to ${h.limits.max_duration_s} seconds and ${h.limits.max_upload_mb} MB.`;
  if (h.auth_required && !api.token) {
    els['upload-note'].textContent = 'This server needs an access token. Open the full link you were given.';
    els.choose.disabled = els.sample.disabled = true;
  }
  render();
}

boot();
