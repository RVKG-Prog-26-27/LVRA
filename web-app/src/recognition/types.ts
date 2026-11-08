/**
 * The contract between the app and any handwriting recognition backend.
 *
<<<<<<< HEAD
 * The browser splits the photo into line images (0001.png = first line,
 * 0002.png = second line, ...) and a labels.csv with the header
 * "filename,text" and an empty text column. This is the same format the
 * training code in latvian_htr reads. The backend fills in the text column
 * and sends the CSV back.
 *
 * The UI only ever talks to HandwritingRecognitionService. Backends:
 *   serverRecognitionService.ts  the real AI on our server (web-app/server)
 *   mockRecognitionService.ts    TEMPORARY sample text for development
=======
 * The UI only ever talks to HandwritingRecognitionService. To plug in the real
 * model, write a new class that implements this interface and select it in
 * createRecognitionService.ts
>>>>>>> origin/web-app
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
  /** The word exactly as it appears in RecognitionResult.text */
  text: string;
  /** Model confidence between 0 and 1 */
  confidence: number;
}

export interface RecognitionResult {
<<<<<<< HEAD
  /** Recognised text, one line per line image, joined with "\n". Not spell-corrected. */
  text: string;
  /** Optional per-word confidence, in reading order. Empty while the AI returns text only. */
=======
  /** Recognised text with line breaks preserved ("\n"); Not spell-corrected */
  text: string;
  /** Optional per-word confidence, in reading order - used to highlight uncertain words */
>>>>>>> origin/web-app
  words: RecognizedWord[];
}

export interface RecognizeOptions {
  /** Aborted when the user cancels or the timeout is reached */
  signal: AbortSignal;
}

export interface HandwritingRecognitionService {
  /**
<<<<<<< HEAD
   * Sends the line images and labels.csv, returns the filled labels.csv as text.
   * Must reject with an AppError (see errors.ts) on failure.
=======
   * Recognises handwritten Latvian text in a JPEG image
   * Must reject with an AppError (see errors.ts) on failure4
>>>>>>> origin/web-app
   */
  recognize(request: RecognitionRequest, options: RecognizeOptions): Promise<string>;
}
