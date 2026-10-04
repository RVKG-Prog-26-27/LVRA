/**
 * Central app configuration: fixed product decisions in one place.
 */
export const config = {
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

  /** Contact email shown in the privacy notice. */
  privacyContactEmail: 'slepie@gmail.com',
} as const;
