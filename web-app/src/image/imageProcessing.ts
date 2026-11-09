/**
 * Browser-side image handling: open the user's file, draw it rotated and
 * scaled, and cut it into line images for recognition.
 * Everything happens in memory; nothing is written to disk or storage.
 */
import { AppError } from '../recognition/errors';
import type { LineImage } from '../recognition/types';
import { splitCanvasIntoLines } from './lineSegmentation';
import { fitWithin, rotatedSize, type Rotation, type Size } from './imageSize';

export interface LoadedImage {
  source: CanvasImageSource;
  size: Size;
  /** Frees the decoded image from memory. */
  release: () => void;
}

/**
 * Opens a user-selected file and keeps it at most maxLong x maxShort, the
 * size recognition uses anyway. Several full-size phone photos would
 * otherwise take hundreds of MB of memory. Browsers apply EXIF orientation
 * automatically.
 */
export async function loadImageFile(file: File | null | undefined, maxLong: number, maxShort: number): Promise<LoadedImage> {
  return shrinkToFit(await decodeImageFile(file), maxLong, maxShort);
}

async function decodeImageFile(file: File | null | undefined): Promise<LoadedImage> {
  if (!file) throw new AppError('NO_IMAGE');
  if (file.type && !file.type.startsWith('image/')) throw new AppError('NOT_AN_IMAGE', file.type);
  if (file.size === 0) throw new AppError('UNSUPPORTED_FORMAT', 'Empty file');

  if (typeof createImageBitmap === 'function') {
    try {
      const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
      return {
        source: bitmap,
        size: { width: bitmap.width, height: bitmap.height },
        release: () => bitmap.close(),
      };
    } catch {
      // Fall through to the <img> decoder, which some browsers handle better.
    }
  }

  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.decoding = 'async';
    img.src = url;
    await img.decode();
    if (!img.naturalWidth || !img.naturalHeight) throw new Error('No dimensions');
    return {
      source: img,
      size: { width: img.naturalWidth, height: img.naturalHeight },
      release: () => {
        img.src = '';
      },
    };
  } catch (error) {
    throw new AppError('UNSUPPORTED_FORMAT', error instanceof Error ? error.message : undefined);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function shrinkToFit(image: LoadedImage, maxLong: number, maxShort: number): LoadedImage {
  const target = fitWithin(image.size, maxLong, maxShort);
  if (target.width === image.size.width && target.height === image.size.height) return image;
  const canvas = document.createElement('canvas');
  canvas.width = target.width;
  canvas.height = target.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return image; // Keep the full-size image rather than fail.
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, target.width, target.height);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(image.source, 0, 0, target.width, target.height);
  image.release();
  return {
    source: canvas,
    size: target,
    release: () => {
      canvas.width = 0;
      canvas.height = 0;
    },
  };
}

/** Draws the image rotated and scaled to fit maxLong x maxShort onto the given canvas. */
export function drawRotated(
  canvas: HTMLCanvasElement,
  image: LoadedImage,
  rotation: Rotation,
  maxLong: number,
  maxShort: number,
): void {
  const target = fitWithin(rotatedSize(image.size, rotation), maxLong, maxShort);
  canvas.width = target.width;
  canvas.height = target.height;

  const ctx = canvas.getContext('2d');
  if (!ctx) throw new AppError('IMAGE_PROCESSING_FAILED', 'No 2D context');

  // Width and height of the unrotated image at the target scale.
  const swap = rotation === 90 || rotation === 270;
  const drawWidth = swap ? target.height : target.width;
  const drawHeight = swap ? target.width : target.height;

  ctx.fillStyle = '#ffffff'; // White paper behind transparent PNGs.
  ctx.fillRect(0, 0, target.width, target.height);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.save();
  ctx.translate(target.width / 2, target.height / 2);
  ctx.rotate((rotation * Math.PI) / 180);
  ctx.drawImage(image.source, -drawWidth / 2, -drawHeight / 2, drawWidth, drawHeight);
  ctx.restore();
}

/**
 * Rotates and scales the photo (max 1920 x 1080 in its orientation), then
 * splits it into line images 0001.png, 0002.png, ... (top line first).
 * The full photo itself is never sent anywhere.
 */
export async function prepareLineImages(
  image: LoadedImage,
  rotation: Rotation,
  options: { maxLongSide: number; maxShortSide: number; maxLines: number },
): Promise<LineImage[]> {
  const canvas = document.createElement('canvas');
  let blobs: Blob[];
  try {
    drawRotated(canvas, image, rotation, options.maxLongSide, options.maxShortSide);
    blobs = await splitCanvasIntoLines(canvas);
  } catch (error) {
    throw new AppError('IMAGE_PROCESSING_FAILED', error instanceof Error ? error.message : undefined);
  } finally {
    // Release canvas memory right away (important on phones).
    canvas.width = 0;
    canvas.height = 0;
  }
  if (blobs.length === 0) throw new AppError('NO_TEXT_FOUND', 'No text lines found');
  if (blobs.length > options.maxLines) throw new AppError('POOR_QUALITY', `${blobs.length} lines found`);
  return blobs.map((blob, i) => ({ filename: lineFilename(i), blob }));
}

/** 0 -> 0001.png, the naming used by latvian_htr's labels.csv. */
export function lineFilename(index: number): string {
  return `${String(index + 1).padStart(4, '0')}.png`;
}
