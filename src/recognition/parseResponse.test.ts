import { AppError } from './errors';
import { errorCodeFromResponse, parseRecognitionResponse } from './parseResponse';

describe('parseRecognitionResponse', () => {
  it('accepts text with words and normalises line endings', () => {
    const result = parseRecognitionResponse({
      text: 'Rinda viens\r\nRinda divi',
      words: [{ text: 'Rinda', confidence: 0.8 }, { bad: true }, { text: 'divi', confidence: 1.4 }],
    });
    expect(result.text).toBe('Rinda viens\nRinda divi');
    expect(result.words).toEqual([
      { text: 'Rinda', confidence: 0.8 },
      { text: 'divi', confidence: 1 },
    ]);
  });

  it('treats empty text as "no handwriting found"', () => {
    expect(() => parseRecognitionResponse({ text: '   ' })).toThrowError(expect.objectContaining({ code: 'NO_TEXT_FOUND' }));
  });

  it('rejects a response without text', () => {
    expect(() => parseRecognitionResponse({ result: 'x' })).toThrow(AppError);
  });
});

describe('errorCodeFromResponse', () => {
  it('uses the error code sent by the server when known', () => {
    expect(errorCodeFromResponse(422, { error: { code: 'POOR_QUALITY' } })).toBe('POOR_QUALITY');
  });

  it('falls back to the HTTP status', () => {
    expect(errorCodeFromResponse(503, null)).toBe('SERVICE_UNAVAILABLE');
    expect(errorCodeFromResponse(413, null)).toBe('INVALID_IMAGE');
    expect(errorCodeFromResponse(504, null)).toBe('TIMEOUT');
  });
});
