// The drawing surface: fits the frame to the stage, captures outlines and clicks,
// and plays the "snap" where the rough doodle gives way to the model's mask.

const MARKER = '#8ee040';
const ADD_COLOR = '#8ee040';
const REMOVE_COLOR = '#ff5a4d';

function distanceToSegment(p, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lengthSq = dx * dx + dy * dy;
  const t = lengthSq === 0 ? 0 : Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / lengthSq));
  return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

/** Ramer-Douglas-Peucker, iterative so long strokes cannot overflow the call stack. */
export function simplify(points, epsilon) {
  if (points.length < 3) return points.slice();
  const keep = new Array(points.length).fill(false);
  keep[0] = keep[points.length - 1] = true;
  const stack = [[0, points.length - 1]];
  while (stack.length) {
    const [start, end] = stack.pop();
    let worst = 0;
    let index = -1;
    for (let i = start + 1; i < end; i++) {
      const d = distanceToSegment(points[i], points[start], points[end]);
      if (d > worst) {
        worst = d;
        index = i;
      }
    }
    if (worst > epsilon && index !== -1) {
      keep[index] = true;
      stack.push([start, index], [index, end]);
    }
  }
  return points.filter((_, i) => keep[i]);
}

export class Stage {
  constructor(elements, handlers) {
    this.stage = elements.stage;
    this.viewport = elements.viewport;
    this.frame = elements.frame;
    this.layers = elements.layers;
    this.ink = elements.ink;
    this.video = elements.video;
    this.ctx = this.ink.getContext('2d');
    this.handlers = handlers;
    this.size = { w: 16, h: 9 };
    this.drag = null;

    window.addEventListener('resize', () => this.fit());
    // The stage changes width when the side panel appears; refit whenever it does.
    if (typeof ResizeObserver !== 'undefined') {
      new ResizeObserver(() => this.fit()).observe(this.stage);
    }
    this.ink.addEventListener('pointerdown', (e) => this.down(e));
    this.ink.addEventListener('pointermove', (e) => this.move(e));
    this.ink.addEventListener('pointerup', (e) => this.up(e));
    this.ink.addEventListener('pointercancel', () => this.cancel());
  }

  setSource(width, height) {
    this.size = { w: width, h: height };
    this.fit();
  }

  fit() {
    const { w, h } = this.size;
    const available = this.stage.clientWidth - 24;
    if (available <= 0) return;
    const maxHeight = Math.max(260, Math.min(window.innerHeight - 320, 760));
    const scale = Math.min(available / w, maxHeight / h);
    const width = Math.max(120, Math.floor(w * scale));
    const height = Math.max(80, Math.floor(h * scale));
    this.viewport.style.width = `${width}px`;
    this.viewport.style.height = `${height}px`;
    this.video.style.maxHeight = `${maxHeight}px`;

    const ratio = window.devicePixelRatio || 1;
    this.ink.width = Math.round(width * ratio);
    this.ink.height = Math.round(height * ratio);
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.cssSize = { width, height };
  }

  // ----------------------------------------------------------------- pointer

  norm(event) {
    const rect = this.ink.getBoundingClientRect();
    return {
      x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
      y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
    };
  }

  down(event) {
    const tool = this.handlers.getTool();
    if (!tool || (event.pointerType === 'mouse' && event.button !== 0)) return;
    event.preventDefault();
    this.ink.setPointerCapture(event.pointerId);
    const start = this.norm(event);
    this.drag = { id: event.pointerId, tool, start, points: [start], travelled: 0 };
    if (tool === 'outline') this.clearInk();
  }

  move(event) {
    if (!this.drag || event.pointerId !== this.drag.id) return;
    const events = event.getCoalescedEvents ? event.getCoalescedEvents() : [event];
    for (const e of events.length ? events : [event]) {
      const p = this.norm(e);
      const last = this.drag.points[this.drag.points.length - 1];
      const step = Math.hypot(p.x - last.x, p.y - last.y);
      if (step < 0.002) continue;
      this.drag.travelled += step;
      this.drag.points.push(p);
    }
    if (this.drag.tool === 'outline') this.drawDoodle(this.drag.points, false);
  }

  up(event) {
    if (!this.drag || event.pointerId !== this.drag.id) return;
    const { tool, points, start, travelled } = this.drag;
    this.drag = null;
    if (tool === 'outline') {
      const xs = points.map((p) => p.x);
      const ys = points.map((p) => p.y);
      const area = (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
      if (points.length < 8 || area < 0.01) {
        this.clearInk();
        this.handlers.onTooSmall();
        return;
      }
      this.drawDoodle(points, true);
      let epsilon = 0.003;
      let simple = simplify(points, epsilon);
      while (simple.length > 200) simple = simplify(points, (epsilon *= 1.5));
      this.handlers.onOutline(simple);
    } else if (travelled < 0.02) {
      this.handlers.onPoint(start, tool === 'add' ? 1 : 0);
    }
  }

  cancel() {
    this.drag = null;
    this.clearInk();
  }

  // ----------------------------------------------------------------- drawing

  clearInk() {
    if (!this.cssSize) return;
    this.ctx.clearRect(0, 0, this.cssSize.width, this.cssSize.height);
  }

  px(p) {
    return [p.x * this.cssSize.width, p.y * this.cssSize.height];
  }

  drawDoodle(points, closed) {
    this.clearInk();
    const ctx = this.ctx;
    const trace = () => {
      ctx.beginPath();
      points.forEach((p, i) => (i ? ctx.lineTo(...this.px(p)) : ctx.moveTo(...this.px(p))));
      if (closed) ctx.closePath();
    };
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    trace();
    ctx.strokeStyle = 'rgba(255,255,255,0.6)';
    ctx.lineWidth = 7;
    ctx.stroke();
    trace();
    ctx.strokeStyle = MARKER;
    ctx.lineWidth = 4;
    ctx.stroke();
    if (!closed && points.length > 2) {
      ctx.setLineDash([4, 8]);
      ctx.beginPath();
      ctx.moveTo(...this.px(points[points.length - 1]));
      ctx.lineTo(...this.px(points[0]));
      ctx.strokeStyle = 'rgba(142,224,64,0.6)';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  /** Draw the add (+) and remove (-) points the user has clicked on this frame. */
  drawMarks(marks) {
    this.clearInk();
    const ctx = this.ctx;
    for (const m of marks) {
      const [x, y] = this.px(m);
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fillStyle = m.label === 1 ? ADD_COLOR : REMOVE_COLOR;
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#fff';
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x - 4, y);
      ctx.lineTo(x + 4, y);
      if (m.label === 1) {
        ctx.moveTo(x, y - 4);
        ctx.lineTo(x, y + 4);
      }
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#1b1e22';
      ctx.stroke();
    }
  }

  // ------------------------------------------------------------------ layers

  /** Replace the mask layers. Each entry is {src, snap?}. */
  setLayers(entries) {
    this.layers.replaceChildren();
    for (const { src, snap } of entries) {
      const img = document.createElement('img');
      img.alt = '';
      img.draggable = false;
      if (snap) img.className = 'snap-in';
      img.addEventListener('error', () => img.remove());
      img.src = src;
      this.layers.append(img);
    }
  }

  /** The signature moment: the mask fades in while the rough doodle fades out. */
  fadeDoodle() {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      this.clearInk();
      return;
    }
    const animation = this.ink.animate([{ opacity: 1 }, { opacity: 0 }], {
      duration: 520,
      delay: 140,
      easing: 'ease-out',
      fill: 'forwards',
    });
    animation.onfinish = () => {
      this.clearInk();
      animation.cancel();
    };
  }
}
