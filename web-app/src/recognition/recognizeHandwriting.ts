import { devInspector } from './devInspector';
import { AppError, toAppError } from './errors';
import { buildLabelsCsv, parseLabelsCsv } from './labelsCsv';
import { mockSettings } from './mockSettings';
import type { HandwritingRecognitionService, LineImage, RecognitionResult } from './types';

export interface RecognizeHandwritingOptions {
  timeoutMs: number;
  trainingConsent: boolean;
  /** Aborted when the user presses "Atcelt". */
  signal?: AbortSignal;
}

/**
 * The single entry point the UI uses: line images in, recognised text out.
 * Builds labels.csv, adds the timeout and cancellation on top of whichever
 * service is active, and turns the returned CSV into text for the editor.
 */
export async function recognizeHandwriting(
  service: HandwritingRecognitionService,
  lines: LineImage[],
  { timeoutMs, trainingConsent, signal }: RecognizeHandwritingOptions,
): Promise<RecognitionResult> {
  if (lines.length === 0) throw new AppError('NO_TEXT_FOUND');
  if (signal?.aborted) throw new AppError('CANCELLED');

  const labelsCsv = buildLabelsCsv(lines.map((l) => ({ filename: l.filename, text: '' })));
  devInspector.recordRequest(mockSettings.backend, lines, labelsCsv, trainingConsent);

  const controller = new AbortController();
  let timedOut = false;

  const onUserAbort = () => controller.abort();
  signal?.addEventListener('abort', onUserAbort, { once: true });

  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
      reject(new AppError('TIMEOUT'));
    }, timeoutMs);
  });

  let csv: string;
  try {
    csv = await Promise.race([
      service.recognize({ lines, labelsCsv, trainingConsent }, { signal: controller.signal }),
      timeout,
    ]);
  } catch (error) {
    if (timedOut) throw new AppError('TIMEOUT');
    throw toAppError(error);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onUserAbort);
  }
  devInspector.recordResponse(csv);

  // Put the text back in the order of the line images (1.png = first line).
  let rows;
  try {
    rows = parseLabelsCsv(csv);
  } catch (error) {
    throw new AppError('UNKNOWN', error instanceof Error ? error.message : 'Bad CSV');
  }
  const byName = new Map(rows.map((r) => [r.filename, r.text]));
  const textLines = lines.map((l) => (byName.get(l.filename) ?? '').replace(/\r?\n/g, ' ').trim());
  if (textLines.every((t) => t === '')) throw new AppError('NO_TEXT_FOUND', 'Empty result');

  return { text: textLines.join('\n'), words: [] };
}
