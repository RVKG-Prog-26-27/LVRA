import { useEffect, useRef } from 'react';
import { lv } from '../i18n/lv';
import { CameraIcon, ImageIcon } from './Icons';

interface Props {
  onFiles: (files: File[] | null) => void;
  /** 'start' on the first screen, 'add' under the photos in the preview. */
  variant?: 'start' | 'add';
  disabled?: boolean;
}

/**
 * Two ways to provide photos. Both use the browser's file input: the gallery
 * button allows several photos at once, the camera button adds
 * capture="environment", which opens the rear camera on phones (one photo per
 * press; more can be added from the preview). Permissions are requested by
 * the phone's own system dialog.
 */
export function ImagePicker({ onFiles, variant = 'start', disabled }: Props) {
  const libraryRef = useRef<HTMLInputElement>(null);
  const cameraRef = useRef<HTMLInputElement>(null);

  // The "cancel" event (user closed the picker without choosing) is not wired by React 18.
  useEffect(() => {
    const inputs = [libraryRef.current, cameraRef.current].filter(Boolean) as HTMLInputElement[];
    const onCancel = () => onFiles(null);
    inputs.forEach((input) => input.addEventListener('cancel', onCancel));
    return () => inputs.forEach((input) => input.removeEventListener('cancel', onCancel));
  }, [onFiles]);

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    event.target.value = ''; // Drop the browser's reference to the files and allow picking the same file again.
    onFiles(files.length > 0 ? files : null);
  };

  const isStart = variant === 'start';
  return (
    <div className={isStart ? 'picker' : 'picker picker-add'}>
      {isStart ? (
        <>
          <button type="button" className="btn btn-primary" onClick={() => libraryRef.current?.click()} disabled={disabled}>
            <ImageIcon />
            {lv.chooseImage}
          </button>
          <button type="button" className="btn btn-secondary" onClick={() => cameraRef.current?.click()} disabled={disabled}>
            <CameraIcon />
            {lv.takePhoto}
          </button>
        </>
      ) : (
        <>
          <button type="button" className="btn btn-secondary" onClick={() => cameraRef.current?.click()} disabled={disabled}>
            <CameraIcon />
            {lv.takeAnotherPhoto}
          </button>
          <button type="button" className="btn btn-secondary" onClick={() => libraryRef.current?.click()} disabled={disabled}>
            <ImageIcon />
            {lv.addFromGallery}
          </button>
        </>
      )}
      <input ref={libraryRef} type="file" accept="image/*" multiple hidden onChange={handleChange} />
      <input ref={cameraRef} type="file" accept="image/*" capture="environment" hidden onChange={handleChange} />
    </div>
  );
}
