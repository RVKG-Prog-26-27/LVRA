/**
 * Central app configuration. Values that differ between environments come
 * from Vite environment variables (see .env.example); the rest are fixed
 * product decisions.
 */

export type RecognitionMode = 'mock' | 'api';

const env = import.meta.env;

export const config = {
  /** Which recognition service to use. Defaults to the temporary mock. */
  recognitionMode: (env.VITE_RECOGNITION_MODE === 'api' ? 'api' : 'mock') as RecognitionMode,

  /** Base URL of the recognition server (used only in "api" mode). */
  recognitionApiUrl: (env.VITE_RECOGNITION_API_URL ?? '').trim(),

  /** After this long the recognition attempt is cancelled and a timeout error is shown. */
  recognitionTimeoutMs: 60_000,

  /** Words with confidence below this value are highlighted as uncertain. */
  uncertainConfidenceThreshold: 0.7,

  image: {
    /** The processed image must fit inside 1920 x 1080 (or 1080 x 1920 for portrait). */
    maxLongSide: 1920,
    maxShortSide: 1080,
    /** JPEG quality used for the image sent to the recognition service. */
    jpegQuality: 0.92,
  },

  /**
   * Contact email shown in the privacy notice.
   * PLACEHOLDER: set VITE_PRIVACY_CONTACT_EMAIL before publishing.
   */
  privacyContactEmail: (env.VITE_PRIVACY_CONTACT_EMAIL ?? '').trim() || 'kontakti@example.com',
} as const;
