/**
 * TEMPORARY MOCK. NOT THE REAL AI MODEL.
 *
 * For development without the server: it ignores the line images and fills
 * the text column of labels.csv with Latvian sample lines, one per PNG.
 * Selected only from the developer panel (?dev=1); while it is active the UI
 * shows the "Demonstrācijas režīms" notice.
 */
import { AppError } from './errors';
import { buildLabelsCsv, parseLabelsCsv } from './labelsCsv';
import { mockSettings, type MockSettings } from './mockSettings';
import type { HandwritingRecognitionService, RecognitionRequest, RecognizeOptions } from './types';

export const MOCK_SAMPLE_LINES = [
  'Šodien no rīta devos uz tirgu.',
  'Nopirku maizi, pienu un ābolus.',
  'Vakarā ciemos atbrauks vecmāmiņa.',
];

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
              return; // Never answers; the timeout in recognizeHandwriting takes over.
            default:
              reject(new AppError(scenario, 'Mock scenario'));
          }
        },
        scenario === 'hang' ? 0 : delayMs,
      );
    });
  }
}
