/**
 * DOM helpers for the result editor (a contenteditable element).
 *
 * The editor holds plain text nodes with "\n" line breaks (rendered with
 * white-space: pre-wrap) plus <mark> elements around uncertain words. Each
 * mark remembers its original word; once the user changes it, the mark is
 * removed, which is how the highlight disappears after editing.
 */
import type { TextSegment } from '../recognition/segments';

const MARK_ATTR = 'data-original';
const SENTINEL_ATTR = 'data-sentinel';

/** Replaces the editor content with the given segments. */
export function renderSegments(container: HTMLElement, segments: TextSegment[]): void {
  container.textContent = '';
  const doc = container.ownerDocument;
  for (const segment of segments) {
    if (segment.uncertain) {
      const mark = doc.createElement('mark');
      mark.className = 'uncertain';
      mark.setAttribute(MARK_ATTR, segment.text);
      mark.textContent = segment.text;
      container.appendChild(mark);
    } else {
      container.appendChild(doc.createTextNode(segment.text));
    }
  }
  ensureSentinel(container);
}

/**
 * A trailing <br> lets the caret sit on an empty last line (a lone trailing
 * "\n" does not render in pre-wrap). It contributes no text.
 */
export function ensureSentinel(container: HTMLElement): void {
  const last = container.lastChild;
  if (last instanceof HTMLBRElement && last.hasAttribute(SENTINEL_ATTR)) return;
  container.querySelectorAll(`br[${SENTINEL_ATTR}]`).forEach((br) => br.remove());
  const br = container.ownerDocument.createElement('br');
  br.setAttribute(SENTINEL_ATTR, '');
  container.appendChild(br);
}

/** Removes the highlight from every uncertain word the user has changed. Returns true if anything changed. */
export function unwrapEditedMarks(container: HTMLElement): boolean {
  let changed = false;
  container.querySelectorAll<HTMLElement>(`mark[${MARK_ATTR}]`).forEach((mark) => {
    const content = mark.textContent ?? '';
    if (content === mark.getAttribute(MARK_ATTR)) return;
    changed = true;
    if (content) mark.replaceWith(container.ownerDocument.createTextNode(content));
    else mark.remove();
  });
  if (changed) container.normalize();
  return changed;
}

/** Plain text of the editor, with line breaks as "\n". */
export function extractText(container: HTMLElement): string {
  let out = '';
  const walk = (node: Node) => {
    node.childNodes.forEach((child) => {
      if (child.nodeType === Node.TEXT_NODE) {
        out += (child as Text).data;
      } else if (child instanceof HTMLBRElement) {
        if (!child.hasAttribute(SENTINEL_ATTR)) out += '\n';
      } else if (child instanceof HTMLElement) {
        // Some browsers wrap new lines in <div>/<p>; treat them as line breaks.
        const isBlock = child.tagName === 'DIV' || child.tagName === 'P';
        if (isBlock && out !== '' && !out.endsWith('\n')) out += '\n';
        walk(child);
      }
    });
  };
  walk(container);
  return out;
}

/** Caret position as a character offset from the start of the editor, or null. */
export function getCaretOffset(container: HTMLElement): number | null {
  const selection = container.ownerDocument.getSelection();
  if (!selection || selection.rangeCount === 0) return null;
  const range = selection.getRangeAt(0);
  if (!container.contains(range.endContainer)) return null;
  const before = container.ownerDocument.createRange();
  before.selectNodeContents(container);
  before.setEnd(range.endContainer, range.endOffset);
  return before.toString().length;
}

/** Places the caret at a character offset from the start of the editor. */
export function setCaretOffset(container: HTMLElement, offset: number): void {
  const doc = container.ownerDocument;
  const selection = doc.getSelection();
  if (!selection) return;
  const walker = doc.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  let remaining = offset;
  let node = walker.nextNode() as Text | null;
  const range = doc.createRange();
  while (node) {
    if (remaining <= node.data.length) {
      range.setStart(node, remaining);
      range.collapse(true);
      selection.removeAllRanges();
      selection.addRange(range);
      return;
    }
    remaining -= node.data.length;
    node = walker.nextNode() as Text | null;
  }
  range.selectNodeContents(container);
  range.collapse(false);
  selection.removeAllRanges();
  selection.addRange(range);
}

/** Inserts plain text at the caret (used for Enter and paste so no HTML gets in). */
export function insertPlainText(container: HTMLElement, text: string): void {
  const doc = container.ownerDocument;
  const selection = doc.getSelection();
  if (!selection || selection.rangeCount === 0) return;
  const range = selection.getRangeAt(0);
  if (!container.contains(range.commonAncestorContainer)) return;
  range.deleteContents();
  const node = doc.createTextNode(text);
  range.insertNode(node);
  range.setStartAfter(node);
  range.collapse(true);
  selection.removeAllRanges();
  selection.addRange(range);
}
