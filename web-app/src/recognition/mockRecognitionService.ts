/**
 * TEMPORARY MOCK
 *
<<<<<<< HEAD
 * For development without the server: it ignores the line images and fills
 * the text column of labels.csv with Latvian sample lines, one per PNG.
 * Selected only from the developer panel (?dev=1); while it is active the UI
 * shows the "Demonstrācijas režīms" notice.
=======
 * Stands in for the unfinished handwriting recognition model so the rest of
 * the app can be built and tested. It ignores the image content and returns
 * a fixed Latvian sample text. While this service is active the UI shows a
 * visible "demonstration mode" notice, so users are never told it is real
 *
>>>>>>> origin/web-app
 */
import { AppError } from './errors';
import { buildLabelsCsv, parseLabelsCsv } from './labelsCsv';
import { mockSettings, type MockSettings } from './mockSettings';
import type { HandwritingRecognitionService, RecognitionRequest, RecognizeOptions } from './types';

<<<<<<< HEAD
export const MOCK_SAMPLE_LINES = [
  'Šodien no rīta devos uz tirgu.',
  'Nopirku maizi, pienu un ābolus.',
  'Vakarā ciemos atbrauks vecmāmiņa.',
];
=======
export const MOCK_SAMPLE_TEXT =
  'Šodien no rīta devos uz tirgu.\nNopirku maizi, pienu un ābolus.\nVakarā ciemos atbrauks vecmāmiņa.';

const UNCERTAIN_WORDS: Record<string, number> = { devos: 0.62, ābolus: 0.55, vecmāmiņa: 0.68 };

function buildWords(text: string, allConfident: boolean) {
  return (text.match(/[\p{L}\p{N}]+/gu) ?? []).map((word) => ({
    text: word,
    confidence: allConfident ? 0.97 : (UNCERTAIN_WORDS[word] ?? 0.95),
  }));
}
>>>>>>> origin/web-app

export class MockRecognitionService implements HandwritingRecognitionService {
  constructor(private readonly settings: MockSettings = mockSettings) {}

  recognize(request: RecognitionRequest, { signal }: RecognizeOptions): Promise<string> {
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
          if (request.lines.length === 0) {
            reject(new AppError('INVALID_IMAGE', 'No line images'));
            return;
          }
          switch (scenario) {
            case 'success': {
              const rows = parseLabelsCsv(request.labelsCsv).map((row, i) => ({
                filename: row.filename,
                text: MOCK_SAMPLE_LINES[i % MOCK_SAMPLE_LINES.length],
              }));
              resolve(buildLabelsCsv(rows));
              return;
            }
            case 'hang':
<<<<<<< HEAD
              return; // Never answers; the timeout in recognizeHandwriting takes over.
=======
              // Never settles; the caller's timeout or cancel ends the attempt
              signal.addEventListener('abort', onAbort, { once: true });
              return;
>>>>>>> origin/web-app
            default:
              reject(new AppError(scenario, 'Mock scenario'));
          }
        },
        scenario === 'hang' ? 0 : delayMs,
      );
    });
  }
}
