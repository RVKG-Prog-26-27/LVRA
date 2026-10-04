import { AppError, toAppError } from './errors';
import type { HandwritingRecognitionService, RecognitionResult } from './types';

export interface RecognizeHandwritingOptions {
  timeoutMs: number;
  /** Aborted when the user presses "Atcelt". */
  signal?: AbortSignal;
}

/**
 * The single entry point the UI uses: image in, recognised text out.
 * Adds the timeout and cancellation on top of whichever service is active,
 * so every backend gets identical behaviour.
 */
export async function recognizeHandwriting(
  service: HandwritingRecognitionService,
  image: Blob | null,
  { timeoutMs, signal }: RecognizeHandwritingOptions,
): Promise<RecognitionResult> {
  if (!image) throw new AppError('NO_IMAGE');
  if (signal?.aborted) throw new AppError('CANCELLED');

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

  try {
    return await Promise.race([service.recognize(image, { signal: controller.signal }), timeout]);
  } catch (error) {
    if (timedOut) throw new AppError('TIMEOUT');
    throw toAppError(error);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onUserAbort);
  }
}
