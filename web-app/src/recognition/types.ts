/**
 * The contract between the app and any handwriting recognition backend.
 *
 * The UI only ever talks to HandwritingRecognitionService. To plug in the real
 * model, write a new class that implements this interface (or configure the
 * existing ApiRecognitionService) and select it in createRecognitionService.ts.
 */

export interface RecognizedWord {
  /** The word exactly as it appears in RecognitionResult.text. */
  text: string;
  /** Model confidence between 0 and 1. */
  confidence: number;
}

export interface RecognitionResult {
  /** Recognised text with line breaks preserved ("\n"). Not spell-corrected. */
  text: string;
  /** Optional per-word confidence, in reading order. Used to highlight uncertain words. */
  words: RecognizedWord[];
}

export interface RecognizeOptions {
  /** Aborted when the user cancels or the timeout is reached. */
  signal: AbortSignal;
}

export interface HandwritingRecognitionService {
  /**
   * Recognises handwritten Latvian text in a JPEG image.
   * Must reject with an AppError (see errors.ts) on failure.
   */
  recognize(image: Blob, options: RecognizeOptions): Promise<RecognitionResult>;
}
