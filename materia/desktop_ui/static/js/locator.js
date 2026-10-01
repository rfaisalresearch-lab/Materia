/**
 * Magnification callout.
 *
 * At every scale below the wafer view the specimen is drawn small in the
 * corner with the current field of view boxed on it, and guide lines fan from
 * that box out to the frame of the view you are actually looking at. It is the
 * inset-and-callout convention used in microscopy figures, so the relationship
 * between the two scales is visible rather than implied.
 */

import { fitCanvas, num } from './util.js';

/**
 * @param {HTMLCanvasElement} canvas  a full-viewport overlay canvas
 * @param {any} wafer                 the wafer block from /api/state
 * @param {{x:number,y:number}} roi   field-of-view centre, millimetres
 * @param {{spanMm?:number, label?:string, insetPx?:number}} [opts]
 */
export function drawCallout(canvas, wafer, roi, opts = {}) {
  const [w, h] = fitCanvas(canvas);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);
  if (!wafer || !roi) return;

  const dpr = w / canvas.getBoundingClientRect().width;
  const inset = (opts.insetPx || 86) * dpr;
  const margin = 14 * dpr;
  const radius = inset / 2;
  const cx = w - margin - radius;
  const cy = margin + radius;
  const scale = radius / wafer.radius_mm;

  ctx.save();
  ctx.lineWidth = 1 * dpr;

  ctx.beginPath();
  const outline = wafer.outline || [];
  if (outline.length) {
    outline.forEach(([x, y], i) => {
      const px = cx + x * scale;
      const py = cy - y * scale;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.closePath();
  } else {
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  }
  ctx.fillStyle = '#000000';
  ctx.fill();
  ctx.strokeStyle = '#9aa2a9';
  ctx.stroke();

  const px = cx + roi.x * scale;
  const py = cy - roi.y * scale;
  const spanMm = Math.max(opts.spanMm || 0, wafer.radius_mm / 55);
  const half = Math.min(Math.max(4 * dpr, (spanMm * scale) / 2), radius * 0.82);

  // Corner crop marks rather than a closed frame: a full rectangle would run
  // straight through the inset disc.
  const frame = {
    left: margin,
    right: w - margin,
    top: margin,
    bottom: h - margin,
  };
  const tick = Math.min(26 * dpr, (frame.right - frame.left) / 6);

  ctx.strokeStyle = 'rgba(255, 176, 32, .55)';
  ctx.beginPath();
  ctx.moveTo(px - half, py + half);
  ctx.lineTo(frame.left, frame.bottom);
  ctx.moveTo(px - half, py - half);
  ctx.lineTo(frame.left, frame.top + tick);
  ctx.stroke();

  ctx.strokeStyle = 'rgba(255, 176, 32, .85)';
  ctx.lineWidth = 1.4 * dpr;
  ctx.strokeRect(px - half, py - half, half * 2, half * 2);

  ctx.strokeStyle = 'rgba(255, 176, 32, .45)';
  ctx.lineWidth = 1 * dpr;
  ctx.beginPath();
  ctx.moveTo(frame.left, frame.top + tick);
  ctx.lineTo(frame.left, frame.top);
  ctx.lineTo(frame.left + tick, frame.top);
  ctx.moveTo(frame.right - tick, frame.bottom);
  ctx.lineTo(frame.right, frame.bottom);
  ctx.lineTo(frame.right, frame.bottom - tick);
  ctx.moveTo(frame.left, frame.bottom - tick);
  ctx.lineTo(frame.left, frame.bottom);
  ctx.lineTo(frame.left + tick, frame.bottom);
  ctx.stroke();

  ctx.fillStyle = '#c3cad0';
  ctx.font = `${9.5 * dpr}px "Lucida Grande", "Lucida Sans Unicode", Geneva, Tahoma, sans-serif`;
  ctx.textAlign = 'right';
  ctx.fillText(`${num(roi.x, 4)}, ${num(roi.y, 4)} mm`,
               cx + radius, cy + radius + 11 * dpr);
  if (opts.label) {
    ctx.fillStyle = 'rgba(255, 176, 32, .9)';
    ctx.textAlign = 'right';
    ctx.fillText(opts.label, cx + radius, cy + radius + 23 * dpr);
  }
  ctx.restore();
}
