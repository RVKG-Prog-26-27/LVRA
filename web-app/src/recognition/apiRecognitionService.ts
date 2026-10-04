import { AppError } from './errors';
import { errorCodeFromResponse, parseRecognitionResponse } from './parseResponse';
import type { HandwritingRecognitionService, RecognitionResult, RecognizeOptions } from './types';

type FetchLike = (input: string, init: RequestInit) => Promise<Response>;

/**
 * Talks to the real recognition server over HTTP.
 * Request and response formats are documented in docs/API.md.
 */
export class ApiRecognitionService implements HandwritingRecognitionService {
  private readonly endpoint: string;

  constructor(
    baseUrl: string,
    private readonly fetchImpl: FetchLike = (input, init) => fetch(input, init),
  ) {
    this.endpoint = baseUrl ? `${baseUrl.replace(/\/+$/, '')}/recognize` : '';
  }

  async recognize(image: Blob, { signal }: RecognizeOptions): Promise<RecognitionResult> {
    if (!this.endpoint) {
      throw new AppError('SERVICE_UNAVAILABLE', 'Recognition API URL is not configured');
    }
    if (typeof navigator !== 'undefined' && navigator.onLine === false) {
      throw new AppError('OFFLINE');
    }

    const body = new FormData();
    body.append('image', image, 'image.jpg');

    let response: Response;
    try {
      response = await this.fetchImpl(this.endpoint, { method: 'POST', body, signal });
    } catch (error) {
      if (signal.aborted) throw new AppError('CANCELLED');
      if (typeof navigator !== 'undefined' && navigator.onLine === false) throw new AppError('OFFLINE');
      throw new AppError('SERVICE_UNAVAILABLE', error instanceof Error ? error.message : undefined);
    }

    if (!response.ok) {
      const errorBody = await response.json().catch(() => null);
      throw new AppError(errorCodeFromResponse(response.status, errorBody), `HTTP ${response.status}`);
    }

    let data: unknown;
    try {
      data = await response.json();
    } catch {
      throw new AppError('UNKNOWN', 'Response is not valid JSON');
    }
    return parseRecognitionResponse(data);
  }
}
