/**
 * Splits a photo of handwriting into one image per text line, in the
 * browser. The full photo never leaves the phone; only the line images are
 * sent.
 *
 * How it works:
 *
 *  1. Even out the lighting: estimate the paper brightness in small blocks
 *     and divide every pixel by it. Paper becomes white, shadows disappear.
 *  2. Mark ink pixels (darker than a threshold found with Otsu's method).
 *  3. Find the separate ink shapes (connected components). Throw away shapes
 *     that are not writing: anything at the left or right edge of the photo
 *     (table, neighbouring page, fingers), large shapes at the top or bottom
 *     edge, page outlines, long printed or ruled lines, filled patches.
 *  3b. Measure how much the writing is tilted (up to 12 degrees either way)
 *     and straighten the photo, so the lines run level. The line images are
 *     cut from the straightened photo.
 *  4. Group the shapes into lines. Two shapes belong to the same line when
 *     they sit side by side at about the same height. Shapes stacked above
 *     each other are never joined, so two lines cannot run into each other,
 *     even when the lines are close, indented or slanted.
 *  5. Add the small marks (Latvian diacritics such as ā, š, ģ, ķ, ļ, ņ, dots,
 *     commas) to the nearest line. A very tall shape that reaches into two
 *     lines (a long descender touching the line below) is cut between them.
 *  6. Cut out each line with a margin. Only the shapes of that line are
 *     drawn; parts of the lines above and below that fall inside the
 *     rectangle are painted white, so every image holds exactly one line.
 *
 * The pure part (splitIntoLines) has no browser dependencies so it can be
 * tested in Node.
 */

export interface LineBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** One cut-out line: greyscale pixels, 0 = black, 1 = white, row by row. */
export interface LineCrop {
  box: LineBox;
  pixels: Float32Array;
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

/** One connected ink shape. */
interface Shape {
  x0: number;
  x1: number; // inclusive
  y0: number;
  y1: number; // inclusive
  area: number;
  keep: boolean;
}

/** Step 3a: labels every ink pixel with its shape number (8-connected). 0 = no ink. */
function findShapes(ink: Uint8Array, width: number, height: number): { labels: Int32Array; shapes: Shape[] } {
  const labels = new Int32Array(width * height);
  const shapes: Shape[] = [];
  const stack: number[] = [];
  for (let start = 0; start < ink.length; start++) {
    if (!ink[start] || labels[start]) continue;
    const id = shapes.length + 1;
    const s: Shape = { x0: width, x1: 0, y0: height, y1: 0, area: 0, keep: true };
    labels[start] = id;
    stack.push(start);
    while (stack.length > 0) {
      const i = stack.pop()!;
      const x = i % width;
      const y = (i - x) / width;
      s.area++;
      if (x < s.x0) s.x0 = x;
      if (x > s.x1) s.x1 = x;
      if (y < s.y0) s.y0 = y;
      if (y > s.y1) s.y1 = y;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= height) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= width) continue;
          const j = yy * width + xx;
          if (ink[j] && !labels[j]) {
            labels[j] = id;
            stack.push(j);
          }
        }
      }
    }
    shapes.push(s);
  }
  return { labels, shapes };
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

/** Median where each value counts as often as its weight. */
function weightedMedian(values: number[], weights: number[]): number {
  if (values.length === 0) return 0;
  const order = values.map((_, i) => i).sort((a, b) => values[a] - values[b]);
  const total = weights.reduce((a, b) => a + b, 0);
  let acc = 0;
  for (const i of order) {
    acc += weights[i];
    if (acc >= total / 2) return values[i];
  }
  return values[order[order.length - 1]];
}

const w = (s: Shape) => s.x1 - s.x0 + 1;
const h = (s: Shape) => s.y1 - s.y0 + 1;
const cy = (s: Shape) => (s.y0 + s.y1) / 2;
const cx = (s: Shape) => (s.x0 + s.x1) / 2;

/** Step 3b: drops shapes that are not writing. Returns the typical letter height. */
function dropNonText(shapes: Shape[], width: number, height: number, checkEdges = true): number {
  for (const s of shapes) {
    if (!checkEdges) {
      if (s.area < 3) s.keep = false;
      continue;
    }
    const atSide = s.x0 === 0 || s.x1 === width - 1;
    const atTopOrBottom = s.y0 === 0 || s.y1 === height - 1;
    if (s.area < 3) s.keep = false; // noise
    else if (atSide) s.keep = false; // table, neighbouring page, fingers
    else if (atTopOrBottom && (w(s) > width * 0.2 || h(s) > height * 0.2)) s.keep = false;
    else if (w(s) > width * 0.5 && h(s) > height * 0.25) s.keep = false; // page outline
  }
  // Typical height of the writing: the median height, where bigger shapes count more (so whole
  // joined-up words count for more than dots and commas).
  const candidates = shapes.filter((s) => s.keep && s.area >= 12 && h(s) >= 4 && h(s) < height * 0.2);
  const letterHeight = weightedMedian(candidates.map(h), candidates.map((s) => s.area)) || 12;
  for (const s of shapes) {
    if (!s.keep) continue;
    if (w(s) > letterHeight * 8 && h(s) < letterHeight * 0.4) s.keep = false; // ruled or printed line, underline
    else if (h(s) > letterHeight * 5 && w(s) < letterHeight * 0.35) s.keep = false; // page edge, margin line
    else if (h(s) > letterHeight * 8) s.keep = false; // far taller than any letter: finger, shadow, drawing
    else if (checkEdges && s.y1 === height - 1 && h(s) > letterHeight * 3) s.keep = false; // a finger coming in from below
    else if (w(s) > letterHeight * 0.8 && h(s) > letterHeight * 0.8 && s.area > 0.45 * w(s) * h(s))
      s.keep = false; // a filled patch, not pen strokes
  }
  return letterHeight;
}

/** Union-find, to collect shapes into groups. */
class Groups {
  private parent: number[];
  constructor(n: number) {
    this.parent = Array.from({ length: n }, (_, i) => i);
  }
  find(i: number): number {
    while (this.parent[i] !== i) {
      this.parent[i] = this.parent[this.parent[i]];
      i = this.parent[i];
    }
    return i;
  }
  join(a: number, b: number): void {
    this.parent[this.find(a)] = this.find(b);
  }
}

/** True when two shapes sit side by side at about the same height (same line). */
function sameLine(a: Shape, b: Shape, maxGap: number): boolean {
  const gap = Math.max(a.x0, b.x0) - Math.min(a.x1, b.x1);
  if (gap > maxGap) return false;
  // Shapes that are mostly above each other belong to different lines.
  if (gap < 0 && -gap > 0.5 * Math.min(w(a), w(b))) return false;
  const lo = Math.min(h(a), h(b));
  const hi = Math.max(h(a), h(b));
  const overlap = Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0) + 1;
  return overlap >= 0.5 * lo && Math.abs(cy(a) - cy(b)) <= 0.5 * hi;
}

interface Line {
  shapes: number[]; // indexes into shapes, whole shapes of this line
  split: number[]; // tall shapes shared with other lines, cut by height
  x0: number;
  x1: number;
  /** Typical top and bottom of the letters (median of the shapes), and its middle. */
  top: number;
  bottom: number;
  mid: number;
}

function makeLine(shapes: Shape[], members: number[]): Line {
  const line: Line = { shapes: members, split: [], x0: Infinity, x1: -Infinity, top: 0, bottom: 0, mid: 0 };
  updateLine(line, shapes);
  return line;
}

function updateLine(line: Line, shapes: Shape[]): void {
  const ss = line.shapes.map((i) => shapes[i]);
  line.x0 = Math.min(...ss.map((s) => s.x0));
  line.x1 = Math.max(...ss.map((s) => s.x1));
  line.top = median(ss.map((s) => s.y0));
  line.bottom = median(ss.map((s) => s.y1));
  line.mid = (line.top + line.bottom) / 2;
}

/** True when the letter bands of two lines lie at about the same height. */
function sameHeight(a: Line, b: Line): boolean {
  return bandOverlap(a, b) >= 0.6;
}

/** How much the letter bands of two lines overlap, as a share of the lower band (0..1). */
function bandOverlap(a: Line, b: Line): number {
  const lo = Math.min(a.bottom - a.top, b.bottom - b.top) + 1;
  const overlap = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) + 1;
  return Math.max(0, overlap) / lo;
}

/** How far y is from the letter band of a line (0 inside it). */
function distanceToBand(line: Line, y: number): number {
  return y < line.top ? line.top - y : y > line.bottom ? y - line.bottom : 0;
}

/**
 * Same, for choosing between two lines: a mark above a band counts as closer
 * than a mark below one, because most loose marks belong to the line under
 * them (ā, ē, š, the top of a faint capital letter).
 */
function preferenceToBand(line: Line, y: number): number {
  return y < line.top ? (line.top - y) * 0.6 : distanceToBand(line, y);
}

/** Steps 4 and 5: groups the kept shapes into lines, top line first. */
function groupIntoLines(shapes: Shape[], letterHeight: number): Line[] {
  const maxGap = letterHeight * 2.5;
  const kept = shapes.map((_, i) => i).filter((i) => shapes[i].keep);
  const isSmall = (s: Shape) => h(s) < letterHeight * 0.5 && w(s) < letterHeight;
  const isTall = (s: Shape) => h(s) > letterHeight * 2.2;
  const small = kept.filter((i) => isSmall(shapes[i]));
  const tall = kept.filter((i) => !isSmall(shapes[i]) && isTall(shapes[i]));
  const regular = kept.filter((i) => !isSmall(shapes[i]) && !isTall(shapes[i]));

  // Step 4: join regular shapes that sit side by side.
  const cluster = (members: number[]): number[][] => {
    const groups = new Groups(members.length);
    const order = members.map((_, k) => k).sort((a, b) => shapes[members[a]].x0 - shapes[members[b]].x0);
    for (let p = 0; p < order.length; p++) {
      const a = shapes[members[order[p]]];
      for (let q = p + 1; q < order.length; q++) {
        const b = shapes[members[order[q]]];
        if (b.x0 - a.x1 > maxGap) break; // sorted by x0: nothing further right can be close
        if (sameLine(a, b, maxGap)) groups.join(order[p], order[q]);
      }
    }
    const byRoot = new Map<number, number[]>();
    members.forEach((m, k) => {
      const r = groups.find(k);
      byRoot.set(r, [...(byRoot.get(r) ?? []), m]);
    });
    return [...byRoot.values()];
  };
  let lines = cluster(regular).map((m) => makeLine(shapes, m));

  // Tall shapes (big capitals, long loops, a descender touching the next line).
  const orphanTall: number[] = [];
  for (const i of tall) {
    const s = shapes[i];
    const near = lines.filter(
      (l) => l.mid >= s.y0 && l.mid <= s.y1 && s.x1 >= l.x0 - maxGap && s.x0 <= l.x1 + maxGap,
    );
    if (near.length === 0) {
      orphanTall.push(i);
      continue;
    }
    // Lines at the same height are parts of one line that the tall shape connects: join them.
    const first = near[0];
    for (const other of near.slice(1)) {
      if (!sameHeight(first, other)) continue;
      first.shapes.push(...other.shapes);
      first.split.push(...other.split.filter((k) => !first.split.includes(k)));
      lines = lines.filter((l) => l !== other);
      updateLine(first, shapes);
    }
    const reached = near.filter((l) => lines.includes(l));
    if (reached.length === 1) reached[0].shapes.push(i);
    else reached.forEach((l) => l.split.push(i)); // reaches into stacked lines: cut between them
  }
  // Tall shapes with no line nearby (a large title): they form lines of their own.
  lines.push(...cluster(orphanTall).map((m) => makeLine(shapes, m)));
  lines.forEach((l) => updateLine(l, shapes));

  // A line with a wide gap between words can come out as two parts at the same height: join them.
  // The height is compared where the two parts meet (the last letters of one, the first letters
  // of the other), so a line that bends or slopes a little still joins up.
  // Returns the middle height and the letter size of the 3 letters at that end of the line.
  const lineEnd = (line: Line, side: 'left' | 'right') => {
    const letters = line.shapes.map((i) => shapes[i]).filter((s) => h(s) >= letterHeight * 0.5);
    const pool = letters.length > 0 ? letters : line.shapes.map((i) => shapes[i]);
    pool.sort((a, b) => (side === 'right' ? b.x1 - a.x1 : a.x0 - b.x0));
    const end = pool.slice(0, 3);
    return { mid: median(end.map(cy)), size: median(end.map(h)) };
  };
  const joinParts = () => {
    lines.sort((a, b) => a.x0 - b.x0);
    for (let changed = true; changed; ) {
      changed = false;
      outer: for (let p = 0; p < lines.length; p++) {
        for (let q = 0; q < lines.length; q++) {
          if (p === q) continue;
          const a = lines[p];
          const b = lines[q];
          if (b.x1 <= a.x1) continue; // b must continue a to the right
          const endA = lineEnd(a, 'right');
          const startB = lineEnd(b, 'left');
          // Measured in the size of the letters where the parts meet (large in a title).
          const size = Math.max(letterHeight, endA.size, startB.size);
          const gap = b.x0 - a.x1;
          if (gap < -size || gap > size * 8) continue;
          const step = Math.abs(endA.mid - startB.mid);
          // Same height, or right next to each other and nearly the same height
          // (a raised question mark, a word ending in a long tail).
          if (step > size * 0.6 && !(gap <= size * 2.5 && step <= size * 0.9)) continue;
          // Never join across another line lying in the gap at the same height.
          const between = lines.some(
            (c) =>
              c !== a &&
              c !== b &&
              c.x0 >= a.x1 &&
              c.x1 <= b.x0 &&
              c.mid >= Math.min(a.top, b.top) &&
              c.mid <= Math.max(a.bottom, b.bottom),
          );
          if (between) continue;
          a.shapes.push(...b.shapes);
          a.split.push(...b.split.filter((i) => !a.split.includes(i)));
          updateLine(a, shapes);
          lines.splice(q, 1);
          changed = true;
          break outer;
        }
      }
    }
  };
  joinParts();

  // Pieces too small to be a line of their own (the loose top of a faint capital letter, a
  // stray stroke) are handed to the nearest line, like the small marks below.
  const inkOf = (l: Line) => l.shapes.reduce((sum, i) => sum + shapes[i].area, 0);
  const pieceLimit = median(lines.map(inkOf)) * 0.2;
  const pieces = lines.filter((l) => inkOf(l) < pieceLimit && l.split.length === 0);
  const homeFor = (s: Shape, others: Line[], limit = letterHeight * 1.2): Line | null => {
    let best: Line | null = null;
    let bestDist = Infinity;
    for (const l of others) {
      if (cx(s) < l.x0 - letterHeight * 2 || cx(s) > l.x1 + letterHeight * 2) continue;
      if (distanceToBand(l, cy(s)) > limit) continue;
      const d = preferenceToBand(l, cy(s));
      if (d < bestDist) {
        bestDist = d;
        best = l;
      }
    }
    return best;
  };
  for (const piece of pieces) {
    const others = lines.filter((l) => l !== piece && !pieces.includes(l));
    const homes = piece.shapes.map((i) => homeFor(shapes[i], others));
    if (homes.some((home) => home === null)) continue; // not near another line: keep it as a line
    piece.shapes.forEach((i, k) => homes[k]!.shapes.push(i));
    lines = lines.filter((l) => l !== piece);
  }
  lines.forEach((l) => updateLine(l, shapes));

  // Step 5: small marks go to the line whose letter band is closest.
  for (const i of small) homeFor(shapes[i], lines, letterHeight * 1.5)?.shapes.push(i);
  lines.forEach((l) => updateLine(l, shapes));
  joinParts(); // again, now that every line has all its letters and marks

  // Drop lines that are only a few specks, then sort top to bottom
  // (lines at the same height, left to right).
  const typicalInk = median(lines.map(inkOf));
  lines = lines.filter((l) => {
    updateLine(l, shapes);
    const lineWidth = l.x1 - l.x0 + 1;
    const lineTall = Math.max(...l.shapes.map((i) => shapes[i].y1)) - Math.min(...l.shapes.map((i) => shapes[i].y0)) + 1;
    const streak = lineWidth < letterHeight * 2 && lineTall > letterHeight * 3; // a vertical streak, not writing
    return inkOf(l) >= typicalInk * 0.08 && lineWidth >= letterHeight && !streak;
  });
  // Bits of writing completely outside the main text (the edge of a neighbouring page): drop them.
  const main = lines.filter((l) => inkOf(l) >= typicalInk * 0.3);
  if (main.length > 0) {
    const left = Math.min(...main.map((l) => l.x0));
    const right = Math.max(...main.map((l) => l.x1));
    lines = lines.filter((l) => l.x1 > left && l.x0 < right);
  }
  lines.sort((a, b) => a.mid - b.mid);
  const sorted: Line[] = [];
  for (const l of lines) {
    let k = sorted.length;
    while (k > 0) {
      const prev = sorted[k - 1];
      const overlap = Math.min(prev.bottom, l.bottom) - Math.max(prev.top, l.top);
      if (overlap > 0.5 * Math.min(prev.bottom - prev.top, l.bottom - l.top) && prev.x0 > l.x0) k--;
      else break;
    }
    sorted.splice(k, 0, l);
  }
  return sorted;
}

function inkMask(normalized: Float32Array, t: number): Uint8Array {
  const ink = new Uint8Array(normalized.length);
  for (let i = 0; i < ink.length; i++) if (normalized[i] < t) ink[i] = 1;
  return ink;
}

/** Paints every shape that is not writing white, so it is gone for the next steps. */
function whiteOutDropped(normalized: Float32Array, labels: Int32Array, shapes: Shape[]): void {
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] && !shapes[labels[i] - 1].keep) normalized[i] = 1;
  }
}

/**
 * Step 3b: the tilt of the writing in radians (positive = lines go down to
 * the right). Tries angles from -12 to 12 degrees and keeps the one where the
 * ink, counted row by row along that angle, forms the sharpest stripes.
 */
function estimateTilt(labels: Int32Array, shapes: Shape[], width: number, height: number, letterHeight: number): number {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let y = 0; y < height; y += 2) {
    for (let x = 0; x < width; x += 2) {
      const label = labels[y * width + x];
      if (label && shapes[label - 1].keep) {
        xs.push(x);
        ys.push(y);
      }
    }
  }
  if (xs.length < 50) return 0;
  const bin = Math.max(2, letterHeight / 4);
  const diag = Math.hypot(width, height);
  const bins = Math.ceil((2 * diag) / bin) + 1;
  const counts = new Float64Array(bins);
  let bestAngle = 0;
  let bestScore = -1;
  for (let deg = -12; deg <= 12.001; deg += 0.25) {
    const a = (deg * Math.PI) / 180;
    const sin = Math.sin(a);
    const cos = Math.cos(a);
    counts.fill(0);
    for (let k = 0; k < xs.length; k++) {
      const yr = ys[k] * cos - xs[k] * sin + diag;
      counts[Math.floor(yr / bin)]++;
    }
    let score = 0;
    for (let b = 0; b < bins; b++) score += counts[b] * counts[b];
    // Prefer level when angles score the same (a tiny bonus for small angles).
    score *= 1 - Math.abs(deg) * 0.0005;
    if (score > bestScore) {
      bestScore = score;
      bestAngle = a;
    }
  }
  return bestAngle;
}

/** Rotates the image by -angle (so lines tilted by angle become level), on a canvas big enough for all of it. White fill. */
function straighten(src: Float32Array, width: number, height: number, angle: number) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const outW = Math.ceil(Math.abs(width * cos) + Math.abs(height * sin));
  const outH = Math.ceil(Math.abs(width * sin) + Math.abs(height * cos));
  const out = new Float32Array(outW * outH).fill(1);
  const cxIn = width / 2;
  const cyIn = height / 2;
  const cxOut = outW / 2;
  const cyOut = outH / 2;
  for (let y = 0; y < outH; y++) {
    const dy = y - cyOut;
    for (let x = 0; x < outW; x++) {
      const dx = x - cxOut;
      // Where this output pixel comes from in the tilted photo.
      const sx = dx * cos - dy * sin + cxIn;
      const sy = dx * sin + dy * cos + cyIn;
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      if (x0 < 0 || y0 < 0 || x0 >= width - 1 || y0 >= height - 1) continue;
      const fx = sx - x0;
      const fy = sy - y0;
      const i = y0 * width + x0;
      const top = src[i] * (1 - fx) + src[i + 1] * fx;
      const bottom = src[i + width] * (1 - fx) + src[i + width + 1] * fx;
      out[y * outW + x] = top * (1 - fy) + bottom * fy;
    }
  }
  return { data: out, width: outW, height: outH };
}

/**
 * The pure part: greyscale photo in (0..255), one cut-out per text line out,
 * top line first.
 */
export function splitIntoLines(gray: Float32Array, photoWidth: number, photoHeight: number): LineCrop[] {
  let normalized = normalizeLighting(gray, photoWidth, photoHeight);
  let width = photoWidth;
  let height = photoHeight;
  const t = inkThreshold(normalized);

  let { labels, shapes } = findShapes(inkMask(normalized, t), width, height);
  let letterHeight = dropNonText(shapes, width, height);
  whiteOutDropped(normalized, labels, shapes);

  // Straighten a tilted photo and find the shapes again on the level version.
  const tilt = estimateTilt(labels, shapes, width, height, letterHeight);
  if (Math.abs(tilt) > (0.4 * Math.PI) / 180) {
    ({ data: normalized, width, height } = straighten(normalized, width, height, tilt));
    ({ labels, shapes } = findShapes(inkMask(normalized, t), width, height));
    letterHeight = dropNonText(shapes, width, height, false);
  }
  const lines = groupIntoLines(shapes, letterHeight);

  // Which line owns a pixel of a shape that is cut between lines: the one whose band is nearest.
  const ownerOfSplitPixel = (shape: number, y: number): Line | null => {
    let best: Line | null = null;
    let bestDist = Infinity;
    for (const l of lines) {
      if (!l.split.includes(shape)) continue;
      const d = Math.abs(y - l.mid);
      if (d < bestDist) {
        bestDist = d;
        best = l;
      }
    }
    return best;
  };

  const crops: LineCrop[] = [];
  for (const line of lines) {
    const own = new Set(line.shapes.map((i) => i + 1)); // label numbers
    const shared = new Set(line.split.map((i) => i + 1));
    const owns = (label: number, y: number) =>
      own.has(label) || (shared.has(label) && ownerOfSplitPixel(label - 1, y) === line);

    // Bounding box of the line's own ink.
    let x0 = width;
    let x1 = -1;
    let y0 = height;
    let y1 = -1;
    for (const i of [...line.shapes, ...line.split]) {
      const s = shapes[i];
      for (let y = s.y0; y <= s.y1; y++) {
        for (let x = s.x0; x <= s.x1; x++) {
          const label = labels[y * width + x];
          if (label !== i + 1 || !owns(label, y)) continue;
          if (x < x0) x0 = x;
          if (x > x1) x1 = x;
          if (y < y0) y0 = y;
          if (y > y1) y1 = y;
        }
      }
    }
    if (x1 < 0) continue;

    // Margin, like the training images.
    const lineHeight = y1 - y0 + 1;
    const padX = Math.round(lineHeight * 0.3);
    const padY = Math.round(lineHeight * 0.25);
    const box: LineBox = {
      x: Math.max(0, x0 - padX),
      y: Math.max(0, y0 - padY),
      width: 0,
      height: 0,
    };
    box.width = Math.min(width, x1 + 1 + padX) - box.x;
    box.height = Math.min(height, y1 + 1 + padY) - box.y;

    // Mask of the line's own ink, grown by 2 pixels to keep the soft stroke edges.
    const bw = box.width;
    const bh = box.height;
    let mask = new Uint8Array(bw * bh);
    for (let y = 0; y < bh; y++) {
      for (let x = 0; x < bw; x++) {
        const label = labels[(box.y + y) * width + box.x + x];
        if (label && owns(label, box.y + y)) mask[y * bw + x] = 1;
      }
    }
    for (let pass = 0; pass < 2; pass++) {
      const grown = new Uint8Array(bw * bh);
      for (let y = 0; y < bh; y++) {
        for (let x = 0; x < bw; x++) {
          let on = 0;
          for (let dy = -1; dy <= 1 && !on; dy++) {
            const yy = y + dy;
            if (yy < 0 || yy >= bh) continue;
            for (let dx = -1; dx <= 1; dx++) {
              const xx = x + dx;
              if (xx >= 0 && xx < bw && mask[yy * bw + xx]) {
                on = 1;
                break;
              }
            }
          }
          grown[y * bw + x] = on;
        }
      }
      mask = grown;
    }
    // Do not let the grown edge reach into other lines' strokes.
    const pixels = new Float32Array(bw * bh);
    for (let y = 0; y < bh; y++) {
      for (let x = 0; x < bw; x++) {
        const k = y * bw + x;
        const label = labels[(box.y + y) * width + box.x + x];
        const foreign = label !== 0 && !owns(label, box.y + y);
        pixels[k] = mask[k] && !foreign ? normalized[(box.y + y) * width + box.x + x] : 1;
      }
    }
    crops.push({ box, pixels });
  }
  return crops;
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
  const crops = splitIntoLines(toGray(rgba, width, height), width, height);

  const blobs: Blob[] = [];
  const lineCanvas = document.createElement('canvas');
  try {
    for (const { box, pixels } of crops) {
      lineCanvas.width = box.width;
      lineCanvas.height = box.height;
      const lineCtx = lineCanvas.getContext('2d');
      if (!lineCtx) throw new Error('No 2D context');
      const data = lineCtx.createImageData(box.width, box.height);
      for (let i = 0; i < pixels.length; i++) {
        const v = Math.round(pixels[i] * 255);
        const p = i * 4;
        data.data[p] = v;
        data.data[p + 1] = v;
        data.data[p + 2] = v;
        data.data[p + 3] = 255;
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
