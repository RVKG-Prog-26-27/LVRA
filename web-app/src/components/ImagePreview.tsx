import { useEffect, useRef } from 'react';
import { drawRotated, type LoadedImage } from '../image/imageProcessing';
import type { Rotation } from '../image/imageSize';
import { lv } from '../i18n/lv';

interface Props {
  image: LoadedImage;
  rotation: Rotation;
  busy: boolean;
}

const PREVIEW_MAX = 1400;

/** Shows the image exactly as it will be sent (same rotation, drawn by the same code). */
export function ImagePreview({ image, rotation, busy }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    try {
      drawRotated(canvas, image, rotation, PREVIEW_MAX, PREVIEW_MAX);
    } catch {
      // A failed preview is not fatal; the real processing reports its own error.
    }
    return () => {
      canvas.width = 0;
      canvas.height = 0;
    };
  }, [image, rotation]);

  return (
    <div className={`preview${busy ? ' is-busy' : ''}`}>
      <canvas ref={canvasRef} role="img" aria-label={lv.previewAlt} />
    </div>
  );
}
