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

/** Opens a user-selected file. Browsers apply EXIF orientation automatically. */
export async function loadImageFile(file: File | null | undefined): Promise<LoadedImage> {
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

/** Draws the image - rotated and scaled to fit maxLong x maxShort onto the canvas */
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

<<<<<<< HEAD
  ctx.fillStyle = '#ffffff'; // White paper behind transparent PNGs.
=======
  ctx.fillStyle = '#ffffff'; // draws background to PNG without one
>>>>>>> origin/web-app
  ctx.fillRect(0, 0, target.width, target.height);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.save();
  ctx.translate(target.width / 2, target.height / 2);
  ctx.rotate((rotation * Math.PI) / 180);
  ctx.drawImage(image.source, -drawWidth / 2, -drawHeight / 2, drawWidth, drawHeight);
  ctx.restore();
}

<<<<<<< HEAD
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
=======
/** Produces the PNG (max 1920 x 1080) that is sent for recognition. */
export async function prepareImageForRecognition(
  image: LoadedImage,
  rotation: Rotation,
  options: { maxLongSide: number; maxShortSide: number; pngQuality: number },
): Promise<Blob> {
>>>>>>> origin/web-app
  const canvas = document.createElement('canvas');
  let blobs: Blob[];
  try {
    drawRotated(canvas, image, rotation, options.maxLongSide, options.maxShortSide);
<<<<<<< HEAD
    blobs = await splitCanvasIntoLines(canvas);
=======
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, 'image/png', options.pngQuality),
    );
    if (!blob || blob.size === 0) throw new AppError('IMAGE_PROCESSING_FAILED', 'Empty PNG');
    return blob;
>>>>>>> origin/web-app
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
