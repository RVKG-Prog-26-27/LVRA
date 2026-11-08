/**
 * Central app configuration: fixed product decisions in one place.
 */
export const config = {
  /** After this long the recognition attempt is cancelled and a timeout error is shown. */
  recognitionTimeoutMs: 60_000,

  recognition: {
    /**
     * Which backend the app uses when it opens.
     * 'server' = the real AI on our PythonAnywhere server (web-app/server/app.py).
     * 'mock'   = the temporary sample-text service, for development without a server.
     * The developer panel (?dev=1) can switch between them while testing.
     */
    defaultBackend: 'server' as 'server' | 'mock',
    /** Address of the recognition endpoint. Relative, because the website and the AI run on the same server. */
    apiUrl: '/api/recognize',
  },

  /**
   * Highlighting of uncertain words in the editor. Switched off for now: the
   * AI returns only text per line (labels.csv), without confidence values.
   * The highlighting code is kept so it can be switched on again later.
   */
  highlightUncertainWords: false,

  /** Words with confidence below this value are highlighted as uncertain (when highlighting is on). */
  uncertainConfidenceThreshold: 0.7,

  image: {
    /** The processed image must fit inside 1920 x 1080 (or 1080 x 1920 for portrait). */
    maxLongSide: 1920,
    maxShortSide: 1080,
  },

  lines: {
    /** More lines than this in one photo is treated as unreadable (protects the server). */
    maxLines: 100,
  },

  terms: {
    /**
     * Version of the terms of use. When it changes, every user sees the terms
     * window again and has to choose again. Use the "Spēkā no" date.
     */
    version: '2026-11-08',
    /** Key in the browser's localStorage where the user's choice is remembered. */
    storageKey: 'lvra.terms',
  },

  /** Contact email shown in the terms of use and the privacy notice. */
  privacyContactEmail: 'slepie@gmail.com',
} as const;
