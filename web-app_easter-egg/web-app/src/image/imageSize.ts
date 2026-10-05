/** Pure size calculations, kept separate from browser APIs so they can be unit tested. */

export type Rotation = 0 | 90 | 180 | 270;

export interface Size {
  width: number;
  height: number;
}

export function rotateBy(rotation: Rotation, delta: 90 | -90): Rotation {
  return ((((rotation + delta) % 360) + 360) % 360) as Rotation;
}

/** Size of the image after rotation (90 and 270 swap width and height). */
export function rotatedSize({ width, height }: Size, rotation: Rotation): Size {
  return rotation === 90 || rotation === 270 ? { width: height, height: width } : { width, height };
}

/**
 * Scales a size down so it fits inside maxLong x maxShort in its own
 * orientation (1920 x 1080 landscape, 1080 x 1920 portrait), keeping the
 * aspect ratio. Images that already fit are never enlarged.
 */
export function fitWithin({ width, height }: Size, maxLong: number, maxShort: number): Size {
  const long = Math.max(width, height);
  const short = Math.min(width, height);
  const scale = Math.min(1, maxLong / long, maxShort / short);
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}
