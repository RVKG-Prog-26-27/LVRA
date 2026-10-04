import { AppError, type AppErrorCode } from './errors';
import type { RecognitionResult, RecognizedWord } from './types';

/**
 * Validates and normalises a JSON response from the recognition server.
 * The expected shape is documented in docs/API.md.
 */
export function parseRecognitionResponse(data: unknown): RecognitionResult {
  if (!data || typeof data !== 'object' || typeof (data as { text?: unknown }).text !== 'string') {
    throw new AppError('UNKNOWN', 'Response has no "text" string');
  }

  const raw = data as { text: string; words?: unknown };
  const text = raw.text.replace(/\r\n?/g, '\n');

  if (text.trim() === '') {
    throw new AppError('NO_TEXT_FOUND');
  }

  const words: RecognizedWord[] = Array.isArray(raw.words)
    ? raw.words
        .filter(
          (w): w is RecognizedWord =>
            !!w &&
            typeof (w as RecognizedWord).text === 'string' &&
            (w as RecognizedWord).text.length > 0 &&
            typeof (w as RecognizedWord).confidence === 'number' &&
            Number.isFinite((w as RecognizedWord).confidence),
        )
        .map((w) => ({ text: w.text, confidence: Math.min(1, Math.max(0, w.confidence)) }))
    : [];

  return { text, words };
}

const SERVER_ERROR_CODES: AppErrorCode[] = ['NO_TEXT_FOUND', 'POOR_QUALITY', 'INVALID_IMAGE'];

/** Maps an unsuccessful HTTP response (status plus optional JSON body) to an error code. */
export function errorCodeFromResponse(status: number, body: unknown): AppErrorCode {
  const code = (body as { error?: { code?: unknown } } | null)?.error?.code;
  if (typeof code === 'string' && (SERVER_ERROR_CODES as string[]).includes(code)) {
    return code as AppErrorCode;
  }
  if (status === 413 || status === 415 || status === 400) return 'INVALID_IMAGE';
  if (status === 408 || status === 504) return 'TIMEOUT';
  if (status === 422) return 'NO_TEXT_FOUND';
  if (status >= 500 || status === 404 || status === 429) return 'SERVICE_UNAVAILABLE';
  return 'UNKNOWN';
}
