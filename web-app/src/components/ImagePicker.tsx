import { useEffect, useRef } from 'react';
import { lv } from '../i18n/lv';
import { CameraIcon, ImageIcon } from './Icons';

interface Props {
  onFile: (file: File | null) => void;
  disabled?: boolean;
}

/**
 * Two ways to provide an image. Both use the browser's file input: the
 * camera button adds capture="environment", which opens the rear camera on
 * phones. Permissions are requested by the phone's own system dialog.
 */
export function ImagePicker({ onFile, disabled }: Props) {
  const libraryRef = useRef<HTMLInputElement>(null);
  const cameraRef = useRef<HTMLInputElement>(null);

  // The "cancel" event (user closed the picker without choosing) is not wired by React 18.
  useEffect(() => {
    const inputs = [libraryRef.current, cameraRef.current].filter(Boolean) as HTMLInputElement[];
    const onCancel = () => onFile(null);
    inputs.forEach((input) => input.addEventListener('cancel', onCancel));
    return () => inputs.forEach((input) => input.removeEventListener('cancel', onCancel));
  }, [onFile]);

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    event.target.value = ''; // Drop the browser's reference to the file and allow picking the same file again.
    onFile(file);
  };

  return (
    <div className="picker">
      <button type="button" className="btn btn-primary" onClick={() => libraryRef.current?.click()} disabled={disabled}>
        <ImageIcon />
        {lv.chooseImage}
      </button>
      <button type="button" className="btn btn-secondary" onClick={() => cameraRef.current?.click()} disabled={disabled}>
        <CameraIcon />
        {lv.takePhoto}
      </button>
      <input ref={libraryRef} type="file" accept="image/*" hidden onChange={handleChange} />
      <input ref={cameraRef} type="file" accept="image/*" capture="environment" hidden onChange={handleChange} />
    </div>
  );
}
