/**
 * Splits a photo of handwriting into one image per text line, in the
 * browser. First version, kept deliberately simple:
 *
 *  1. Even out the lighting: estimate the paper brightness in small blocks
 *     and divide every pixel by it. Paper becomes white, shadows disappear,
 *     and a dark table around the page also turns white.
 *  2. Mark ink pixels (darker than a threshold found with Otsu's method).
 *  3. Count ink pixels in every row. Rows with ink form bands; the white
 *     gaps between bands separate the lines.
 *  4. Join small bands (Latvian diacritics such as ā, š, ģ, ķ, ļ, ņ sit a
 *     little above or below the letters) into the nearest line, drop specks.
 *  5. Crop each line left to right to where its writing is, with a margin.
 *
 * Works for fairly straight lines with some space between them. Strongly
 * slanted or touching lines come out merged. If that happens too often, the
 * splitting should move to the server and use a stronger method.
 *
 * The pure part (findLineBoxes) has no browser dependencies so it can be
 * tested in Node.
 */

export interface LineBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface LineSplitResult {
  /** Lighting-corrected greyscale image, 0 = black, 1 = white. */
  normalized: Float32Array;
  boxes: LineBox[];
}

/** Brightness of each pixel from RGBA data (ITU-R BT.601 weights), 0..255. */
export function toGray(rgba: Uint8ClampedArray, width: number, height: number): Float32Array {
  const gray = new Float32Array(width * height);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
  }
  return gray;
}

/** Step 1: divide by the local paper brightness. */
function normalizeLighting(gray: Float32Array, width: number, height: number): Float32Array {
  const block = Math.max(8, Math.round(Math.min(width, height) / 60));
  const bw = Math.ceil(width / block);
  const bh = Math.ceil(height / block);

  // Brightest pixel per block = paper there (pen strokes are thinner than a block).
  const maxes = new Float32Array(bw * bh);
  for (let y = 0; y < height; y++) {
    const by = Math.floor(y / block) * bw;
    for (let x = 0; x < width; x++) {
      const v = gray[y * width + x];
      const b = by + Math.floor(x / block);
      if (v > maxes[b]) maxes[b] = v;
    }
  }

  // Smooth the block map (3 x 3 average) so single bright specks do not matter much.
  const bg = new Float32Array(bw * bh);
  for (let y = 0; y < bh; y++) {
    for (let x = 0; x < bw; x++) {
      let sum = 0;
      let n = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const yy = y + dy;
          const xx = x + dx;
          if (yy < 0 || yy >= bh || xx < 0 || xx >= bw) continue;
          sum += maxes[yy * bw + xx];
          n++;
        }
      }
      bg[y * bw + x] = sum / n;
    }
  }

  // Bilinear interpolation of the block map back to full size.
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    const fy = Math.min(Math.max((y + 0.5) / block - 0.5, 0), bh - 1);
    const y0 = Math.floor(fy);
    const y1 = Math.min(y0 + 1, bh - 1);
    const ty = fy - y0;
    for (let x = 0; x < width; x++) {
      const fx = Math.min(Math.max((x + 0.5) / block - 0.5, 0), bw - 1);
      const x0 = Math.floor(fx);
      const x1 = Math.min(x0 + 1, bw - 1);
      const tx = fx - x0;
      const top = bg[y0 * bw + x0] * (1 - tx) + bg[y0 * bw + x1] * tx;
      const bottom = bg[y1 * bw + x0] * (1 - tx) + bg[y1 * bw + x1] * tx;
      const paper = Math.max(top * (1 - ty) + bottom * ty, 1);
      out[y * width + x] = Math.min(1, gray[y * width + x] / paper);
    }
  }
  return out;
}

/** Step 2: Otsu threshold on the normalised image, limited to a sensible range. */
function inkThreshold(norm: Float32Array): number {
  const bins = 256;
  const hist = new Float64Array(bins);
  for (const v of norm) hist[Math.min(bins - 1, Math.floor(v * (bins - 1)))]++;
  const total = norm.length;
  let sumAll = 0;
  for (let i = 0; i < bins; i++) sumAll += i * hist[i];
  let sumB = 0;
  let wB = 0;
  let best = 0;
  let bestT = bins * 0.7;
  for (let t = 0; t < bins; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sumAll - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > best) {
      best = between;
      bestT = t;
    }
  }
  // Keep light notebook grid lines and paper texture out of the ink.
  return Math.min(0.75, Math.max(0.45, bestT / (bins - 1)));
}

/**
 * Removes dark shapes that are not writing: anything touching the photo's
 * edge (the table, the page border, fingers) and very large shapes such as a
 * page outline. Removed pixels are also painted white in the normalised
 * image, so they do not show up in the line images.
 */
function removeFrameShapes(ink: Uint8Array, normalized: Float32Array, width: number, height: number): void {
  const seen = new Uint8Array(width * height);
  const stack: number[] = [];
  const pixels: number[] = [];
  for (let start = 0; start < ink.length; start++) {
    if (!ink[start] || seen[start]) continue;
    stack.push(start);
    seen[start] = 1;
    pixels.length = 0;
    let minX = width;
    let maxX = 0;
    let minY = height;
    let maxY = 0;
    while (stack.length > 0) {
      const i = stack.pop()!;
      pixels.push(i);
      const x = i % width;
      const y = (i - x) / width;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (x > 0 && ink[i - 1] && !seen[i - 1]) (seen[i - 1] = 1), stack.push(i - 1);
      if (x < width - 1 && ink[i + 1] && !seen[i + 1]) (seen[i + 1] = 1), stack.push(i + 1);
      if (y > 0 && ink[i - width] && !seen[i - width]) (seen[i - width] = 1), stack.push(i - width);
      if (y < height - 1 && ink[i + width] && !seen[i + width]) (seen[i + width] = 1), stack.push(i + width);
    }
    const touchesEdge = minX === 0 || minY === 0 || maxX === width - 1 || maxY === height - 1;
    const huge = maxX - minX > width * 0.5 && maxY - minY > height * 0.25;
    if (touchesEdge || huge) {
      for (const i of pixels) {
        ink[i] = 0;
        normalized[i] = 1;
      }
    }
  }
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

interface Band {
  top: number;
  bottom: number; // exclusive
  ink: number;
}

export function findLineBoxes(gray: Float32Array, width: number, height: number): LineSplitResult {
  const normalized = normalizeLighting(gray, width, height);
  const t = inkThreshold(normalized);

  const ink = new Uint8Array(width * height);
  for (let i = 0; i < ink.length; i++) if (normalized[i] < t) ink[i] = 1;
  removeFrameShapes(ink, normalized, width, height);

  const rowInk = new Float64Array(height);
  for (let y = 0; y < height; y++) {
    let count = 0;
    for (let x = 0; x < width; x++) count += ink[y * width + x];
    rowInk[y] = count;
  }

  // Step 3: smooth the row profile a little and find bands of rows with ink.
  const smooth = new Float64Array(height);
  for (let y = 0; y < height; y++) {
    let s = 0;
    let n = 0;
    for (let d = -2; d <= 2; d++) {
      if (y + d >= 0 && y + d < height) {
        s += rowInk[y + d];
        n++;
      }
    }
    smooth[y] = s / n;
  }
  const sortedRows = [...smooth].sort((a, b) => a - b);
  const p95 = sortedRows[Math.floor(sortedRows.length * 0.95)] ?? 0;
  if (p95 <= 0) return { normalized, boxes: [] };
  const rowThreshold = Math.max(2, p95 * 0.08);

  let bands: Band[] = [];
  let start = -1;
  for (let y = 0; y <= height; y++) {
    const on = y < height && smooth[y] > rowThreshold;
    if (on && start < 0) start = y;
    if (!on && start >= 0) {
      let sum = 0;
      for (let k = start; k < y; k++) sum += rowInk[k];
      bands.push({ top: start, bottom: y, ink: sum });
      start = -1;
    }
  }
  if (bands.length === 0) return { normalized, boxes: [] };

  // Step 4: merge diacritics and broken pieces into their line, drop specks.
  const typical = median(bands.filter((b) => b.bottom - b.top >= 4).map((b) => b.bottom - b.top)) || 8;
  let changed = true;
  while (changed && bands.length > 1) {
    changed = false;
    for (let i = 0; i < bands.length; i++) {
      const b = bands[i];
      const h = b.bottom - b.top;
      const gapUp = i > 0 ? b.top - bands[i - 1].bottom : Infinity;
      const gapDown = i < bands.length - 1 ? bands[i + 1].top - b.bottom : Infinity;
      const small = h < typical * 0.5;
      const veryClose = Math.min(gapUp, gapDown) < typical * 0.15;
      if (!(small && Math.min(gapUp, gapDown) < typical * 0.8) && !veryClose) continue;
      const j = gapUp <= gapDown ? i - 1 : i + 1;
      const other = bands[j];
      const merged: Band = {
        top: Math.min(b.top, other.top),
        bottom: Math.max(b.bottom, other.bottom),
        ink: b.ink + other.ink,
      };
      bands = bands.filter((_, k) => k !== i && k !== j);
      bands.splice(Math.min(i, j), 0, merged);
      changed = true;
      break;
    }
  }
  const typicalInk = median(bands.map((b) => b.ink));
  bands = bands.filter((b) => b.bottom - b.top >= Math.max(4, typical * 0.35) && b.ink >= typicalInk * 0.08);

  // The band threshold keeps mostly the body of the letters. Grow each band up
  // and down while rows still have ink (ascenders, descenders, diacritics),
  // but never past halfway to the neighbouring line.
  for (let i = 0; i < bands.length; i++) {
    const b = bands[i];
    const limitUp = i > 0 ? Math.ceil((bands[i - 1].bottom + b.top) / 2) : 0;
    const limitDown = i < bands.length - 1 ? Math.floor((b.bottom + bands[i + 1].top) / 2) : height;
    while (b.top > limitUp && rowInk[b.top - 1] > 0) b.top--;
    while (b.bottom < limitDown && rowInk[b.bottom] > 0) b.bottom++;
  }

  // Step 5: horizontal extent of each line, ignoring small marks far from the writing.
  const boxes: LineBox[] = [];
  for (const b of bands) {
    const h = b.bottom - b.top;
    const colInk = new Float64Array(width);
    for (let y = b.top; y < b.bottom; y++) {
      for (let x = 0; x < width; x++) colInk[x] += ink[y * width + x];
    }
    const maxGap = Math.max(8, h * 1.5);
    const clusters: { x0: number; x1: number; ink: number }[] = [];
    let current: { x0: number; x1: number; ink: number } | null = null;
    for (let x = 0; x < width; x++) {
      if (colInk[x] === 0) continue;
      if (current && x - current.x1 <= maxGap) {
        current.x1 = x;
        current.ink += colInk[x];
      } else {
        current = { x0: x, x1: x, ink: colInk[x] };
        clusters.push(current);
      }
    }
    if (clusters.length === 0) continue;
    const biggest = Math.max(...clusters.map((c) => c.ink));
    const kept = clusters.filter((c) => c.ink >= biggest * 0.15);
    const padX = Math.round(h * 0.3);
    const padY = Math.round(h * 0.25);
    const x0 = Math.max(0, kept[0].x0 - padX);
    const x1 = Math.min(width, kept[kept.length - 1].x1 + 1 + padX);
    const y0 = Math.max(0, b.top - padY);
    const y1 = Math.min(height, b.bottom + padY);
    if (x1 - x0 < h) continue; // a lone mark, not a line of writing
    boxes.push({ x: x0, y: y0, width: x1 - x0, height: y1 - y0 });
  }

  return { normalized, boxes };
}

/**
 * Browser part: takes the rotated, scaled photo on a canvas and returns one
 * greyscale PNG per line, top line first.
 */
export async function splitCanvasIntoLines(canvas: HTMLCanvasElement): Promise<Blob[]> {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('No 2D context');
  const { width, height } = canvas;
  const rgba = ctx.getImageData(0, 0, width, height).data;
  const { normalized, boxes } = findLineBoxes(toGray(rgba, width, height), width, height);

  const blobs: Blob[] = [];
  const lineCanvas = document.createElement('canvas');
  try {
    for (const box of boxes) {
      lineCanvas.width = box.width;
      lineCanvas.height = box.height;
      const lineCtx = lineCanvas.getContext('2d');
      if (!lineCtx) throw new Error('No 2D context');
      const data = lineCtx.createImageData(box.width, box.height);
      for (let y = 0; y < box.height; y++) {
        for (let x = 0; x < box.width; x++) {
          const v = Math.round(normalized[(box.y + y) * width + (box.x + x)] * 255);
          const p = (y * box.width + x) * 4;
          data.data[p] = v;
          data.data[p + 1] = v;
          data.data[p + 2] = v;
          data.data[p + 3] = 255;
        }
      }
      lineCtx.putImageData(data, 0, 0);
      const blob = await new Promise<Blob | null>((resolve) => lineCanvas.toBlob(resolve, 'image/png'));
      if (!blob || blob.size === 0) throw new Error('Empty PNG');
      blobs.push(blob);
    }
  } finally {
    lineCanvas.width = 0;
    lineCanvas.height = 0;
  }
  return blobs;
}
