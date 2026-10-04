/**
 * TEMPORARY MOCK. NOT THE REAL AI MODEL.
 *
 * Stands in for the unfinished handwriting recognition model so the rest of
 * the app can be built and tested. It ignores the image content and returns
 * a fixed Latvian sample text. While this service is active the UI shows a
 * visible "demonstration mode" notice, so users are never told it is real.
 *
 * Remove this file (and mockSettings.ts, DevPanel.tsx) once the real model
 * is connected through ApiRecognitionService or another implementation.
 */
import { AppError } from './errors';
import { mockSettings, type MockSettings } from './mockSettings';
import type { HandwritingRecognitionService, RecognitionResult, RecognizeOptions } from './types';

export const MOCK_SAMPLE_TEXT =
  'Šodien no rīta devos uz tirgu.\nNopirku maizi, pienu un ābolus.\nVakarā ciemos atbrauks vecmāmiņa.';

const UNCERTAIN_WORDS: Record<string, number> = { tirgu: 0.62, ābolus: 0.55, vecmāmiņa: 0.68 };

function buildWords(text: string, allConfident: boolean) {
  return (text.match(/[\p{L}\p{N}]+/gu) ?? []).map((word) => ({
    text: word,
    confidence: allConfident ? 0.97 : (UNCERTAIN_WORDS[word] ?? 0.95),
  }));
}

export class MockRecognitionService implements HandwritingRecognitionService {
  constructor(private readonly settings: MockSettings = mockSettings) {}

  recognize(image: Blob, { signal }: RecognizeOptions): Promise<RecognitionResult> {
    return new Promise((resolve, reject) => {
      if (signal.aborted) {
        reject(new AppError('CANCELLED'));
        return;
      }
      const { scenario, delayMs } = this.settings;

      const onAbort = () => {
        clearTimeout(timer);
        reject(new AppError('CANCELLED'));
      };
      signal.addEventListener('abort', onAbort, { once: true });

      const timer = setTimeout(
        () => {
          signal.removeEventListener('abort', onAbort);
          if (!image || image.size === 0) {
            reject(new AppError('INVALID_IMAGE', 'Empty image'));
            return;
          }
          switch (scenario) {
            case 'success':
            case 'success-confident':
              resolve({
                text: MOCK_SAMPLE_TEXT,
                words: buildWords(MOCK_SAMPLE_TEXT, scenario === 'success-confident'),
              });
              return;
            case 'hang':
              // Never settles; the caller's timeout or cancel ends the attempt.
              signal.addEventListener('abort', onAbort, { once: true });
              return;
            default:
              reject(new AppError(scenario, 'Simulated by mock'));
          }
        },
        scenario === 'hang' ? 0 : delayMs,
      );
    });
  }
}
