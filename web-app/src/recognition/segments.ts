import type { RecognizedWord } from './types';

export interface TextSegment {
  text: string;
  uncertain: boolean;
}

/**
 * Splits the recognised text into plain and uncertain parts for highlighting.
 * Words are matched in reading order; a word that cannot be found in the text
 * is skipped, so a mismatch never changes the text itself.
 */
export function buildSegments(text: string, words: RecognizedWord[], threshold: number): TextSegment[] {
  const segments: TextSegment[] = [];
  const pushPlain = (value: string) => {
    if (!value) return;
    const last = segments[segments.length - 1];
    if (last && !last.uncertain) last.text += value;
    else segments.push({ text: value, uncertain: false });
  };

  let cursor = 0;
  for (const word of words) {
    const index = text.indexOf(word.text, cursor);
    if (index === -1) continue;
    pushPlain(text.slice(cursor, index));
    if (word.confidence < threshold) {
      segments.push({ text: word.text, uncertain: true });
    } else {
      pushPlain(word.text);
    }
    cursor = index + word.text.length;
  }
  pushPlain(text.slice(cursor));
  return segments;
}
