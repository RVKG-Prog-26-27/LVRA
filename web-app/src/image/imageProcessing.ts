/**
 * Browser-side image handling: open the user's file, draw it rotated and
 * scaled, and export the JPEG that is sent for recognition.
 * Everything happens in memory; nothing is written to disk or storage.
 */
import { AppError } from '../recognition/errors';
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

  ctx.fillStyle = '#ffffff'; // JPEG has no transparency; use white paper behind transparent PNGs.
  ctx.fillRect(0, 0, target.width, target.height);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.save();
  ctx.translate(target.width / 2, target.height / 2);
  ctx.rotate((rotation * Math.PI) / 180);
  ctx.drawImage(image.source, -drawWidth / 2, -drawHeight / 2, drawWidth, drawHeight);
  ctx.restore();
}

/** Produces the JPEG (max 1920 x 1080 in its orientation) that is sent for recognition. */
export async function prepareImageForRecognition(
  image: LoadedImage,
  rotation: Rotation,
  options: { maxLongSide: number; maxShortSide: number; jpegQuality: number },
): Promise<Blob> {
  const canvas = document.createElement('canvas');
  try {
    drawRotated(canvas, image, rotation, options.maxLongSide, options.maxShortSide);
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, 'image/jpeg', options.jpegQuality),
    );
    if (!blob || blob.size === 0) throw new AppError('IMAGE_PROCESSING_FAILED', 'Empty JPEG');
    return blob;
  } catch (error) {
    if (error instanceof AppError) throw error;
    throw new AppError('IMAGE_PROCESSING_FAILED', error instanceof Error ? error.message : undefined);
  } finally {
    // Release canvas memory right away (important on phones).
    canvas.width = 0;
    canvas.height = 0;
  }
}
