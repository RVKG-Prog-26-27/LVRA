/**
 * The real AI: sends the line images and labels.csv to our server
 * (web-app/server/app.py, POST /api/recognize) and returns the filled CSV.
 *
 * Request (multipart/form-data):
 *   labels            labels.csv, header "filename,text", text column empty
 *   images            one file per line, named 0001.png, 0002.png, ...
 *   training_consent  "true" or "false"
 * Response:
 *   200  text/csv, the same labels.csv with the text column filled in
 *   4xx/5xx  JSON {"error": "<AppErrorCode>"}
 */
import { config } from '../config/config';
import { AppError, type AppErrorCode } from './errors';
import type { HandwritingRecognitionService, RecognitionRequest, RecognizeOptions } from './types';

const SERVER_CODES: AppErrorCode[] = ['NO_TEXT_FOUND', 'POOR_QUALITY', 'INVALID_IMAGE', 'SERVICE_UNAVAILABLE'];

export class ServerRecognitionService implements HandwritingRecognitionService {
  constructor(private readonly url: string = config.recognition.apiUrl) {}

  async recognize(request: RecognitionRequest, { signal }: RecognizeOptions): Promise<string> {
    if (typeof navigator !== 'undefined' && navigator.onLine === false) throw new AppError('OFFLINE');

    const form = new FormData();
    form.append('labels', new Blob([request.labelsCsv], { type: 'text/csv' }), 'labels.csv');
    for (const line of request.lines) form.append('images', line.blob, line.filename);
    form.append('training_consent', request.trainingConsent ? 'true' : 'false');

    let response: Response;
    try {
      response = await fetch(this.url, { method: 'POST', body: form, signal });
    } catch (error) {
      if (signal.aborted) throw new AppError('CANCELLED');
      if (typeof navigator !== 'undefined' && navigator.onLine === false) throw new AppError('OFFLINE');
      throw new AppError('SERVICE_UNAVAILABLE', error instanceof Error ? error.message : undefined);
    }

    if (response.ok) return response.text();

    let code: AppErrorCode = response.status >= 500 ? 'SERVICE_UNAVAILABLE' : 'INVALID_IMAGE';
    try {
      const body = (await response.json()) as { error?: string };
      if (body.error && SERVER_CODES.includes(body.error as AppErrorCode)) code = body.error as AppErrorCode;
    } catch {
      // Not JSON (for example a hosting error page): keep the status-based code.
    }
    throw new AppError(code, `HTTP ${response.status}`);
  }
}
