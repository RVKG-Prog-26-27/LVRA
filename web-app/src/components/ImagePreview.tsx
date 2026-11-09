import { useEffect, useRef } from 'react';
import { drawRotated, type LoadedImage } from '../image/imageProcessing';
import type { Rotation } from '../image/imageSize';
import { lv } from '../i18n/lv';

interface Props {
  image: LoadedImage;
  rotation: Rotation;
  busy: boolean;
  /** Longest side of the preview canvas. Smaller when many photos are shown, to save memory. */
  maxSize?: number;
}

/** Shows the image exactly as it will be sent (same rotation, drawn by the same code). */
export function ImagePreview({ image, rotation, busy, maxSize = 1400 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    try {
      drawRotated(canvas, image, rotation, maxSize, maxSize);
    } catch {
      // A failed preview is not fatal; the real processing reports its own error.
    }
    return () => {
      canvas.width = 0;
      canvas.height = 0;
    };
  }, [image, rotation, maxSize]);

  return (
    <div className={`preview${busy ? ' is-busy' : ''}`}>
      <canvas ref={canvasRef} role="img" aria-label={lv.previewAlt} />
    </div>
  );
}
