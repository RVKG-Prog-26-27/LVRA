/**
 * The contract between the app and any handwriting recognition backend.
 *
 * The browser splits the photo into line images (0001.png = first line,
 * 0002.png = second line, ...) and a labels.csv with the header
 * "filename,text" and an empty text column. This is the same format the
 * training code in latvian_htr reads. The backend fills in the text column
 * and sends the CSV back.
 *
 * The UI only ever talks to HandwritingRecognitionService. Backends:
 *   serverRecognitionService.ts  the real AI on our server (web-app/server)
 *   mockRecognitionService.ts    TEMPORARY sample text for development
 */

export interface LineImage {
  /** 0001.png, 0002.png, ... in reading order, top line first. */
  filename: string;
  /** Greyscale PNG of one text line. */
  blob: Blob;
}

export interface RecognitionRequest {
  lines: LineImage[];
  /** labels.csv with the header "filename,text" and the text column left empty. */
  labelsCsv: string;
  /** The user's choice in the terms window. false = the server must not keep the files. */
  trainingConsent: boolean;
}

export interface RecognizedWord {
  /** The word exactly as it appears in RecognitionResult.text. */
  text: string;
  /** Model confidence between 0 and 1. */
  confidence: number;
}

export interface RecognitionResult {
  /** Recognised text, one line per line image, joined with "\n". Not spell-corrected. */
  text: string;
  /** Optional per-word confidence, in reading order. Empty while the AI returns text only. */
  words: RecognizedWord[];
}

export interface RecognizeOptions {
  /** Aborted when the user cancels or the timeout is reached. */
  signal: AbortSignal;
}

export interface HandwritingRecognitionService {
  /**
   * Sends the line images and labels.csv, returns the filled labels.csv as text.
   * Must reject with an AppError (see errors.ts) on failure.
   */
  recognize(request: RecognitionRequest, options: RecognizeOptions): Promise<string>;
}
