/**
 * Error codes shared by the image layer and the recognition layer.
 * The UI maps each code to a plain Latvian message (see src/i18n/lv.ts),
 * so no technical error text ever reaches the user.
 */
export type AppErrorCode =
  // Image input and processing
  | 'NO_IMAGE'
  | 'NOT_AN_IMAGE'
  | 'UNSUPPORTED_FORMAT'
  | 'IMAGE_PROCESSING_FAILED'
  // Recognition results reported by the service
  | 'NO_TEXT_FOUND'
  | 'POOR_QUALITY'
  | 'INVALID_IMAGE'
  // Transport and availability
  | 'SERVICE_UNAVAILABLE'
  | 'TIMEOUT'
  | 'OFFLINE'
  | 'CANCELLED'
  | 'UNKNOWN';

export class AppError extends Error {
  readonly code: AppErrorCode;

  constructor(code: AppErrorCode, detail?: string) {
    super(detail ? `${code}: ${detail}` : code);
    this.name = 'AppError';
    this.code = code;
  }
}

/** Normalises anything thrown into an AppError. */
export function toAppError(error: unknown): AppError {
  if (error instanceof AppError) return error;
  if (error instanceof DOMException && error.name === 'AbortError') return new AppError('CANCELLED');
  return new AppError('UNKNOWN', error instanceof Error ? error.message : String(error));
}
