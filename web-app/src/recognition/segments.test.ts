import { buildSegments } from './segments';

describe('buildSegments', () => {
  const text = 'Labdien, pasaule!\nOtrā rinda.';

  it('marks only words below the confidence threshold', () => {
    const segments = buildSegments(
      text,
      [
        { text: 'Labdien', confidence: 0.9 },
        { text: 'pasaule', confidence: 0.5 },
        { text: 'Otrā', confidence: 0.69 },
        { text: 'rinda', confidence: 0.7 },
      ],
      0.7,
    );
    expect(segments).toEqual([
      { text: 'Labdien, ', uncertain: false },
      { text: 'pasaule', uncertain: true },
      { text: '!\n', uncertain: false },
      { text: 'Otrā', uncertain: true },
      { text: ' rinda.', uncertain: false },
    ]);
  });

  it('keeps the full text, including line breaks, whatever the words say', () => {
    const segments = buildSegments(text, [{ text: 'nav tekstā', confidence: 0.1 }], 0.7);
    expect(segments.map((s) => s.text).join('')).toBe(text);
    expect(segments.every((s) => !s.uncertain)).toBe(true);
  });

  it('works without word data', () => {
    expect(buildSegments(text, [], 0.7)).toEqual([{ text, uncertain: false }]);
  });
});
