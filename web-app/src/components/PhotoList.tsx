import type { PhotoItem } from '../app/appState';
import { lv } from '../i18n/lv';
import { RotateLeftIcon, RotateRightIcon } from './Icons';
import { ImagePreview } from './ImagePreview';

interface Props {
  items: PhotoItem[];
  busy: boolean;
  onRotate: (id: number, delta: 90 | -90) => void;
  onRemove: (id: number) => void;
}

/** The chosen photos in the order they will be recognised, each with its own rotate and remove buttons. */
export function PhotoList({ items, busy, onRotate, onRemove }: Props) {
  const many = items.length > 1;
  return (
    <ol className={`photos${many ? ' photos-many' : ''}`}>
      {items.map((item, index) => (
        <li key={item.id} className="photo">
          {many && <p className="photo-label">{lv.photoLabel(index + 1)}</p>}
          <ImagePreview image={item.image} rotation={item.rotation} busy={busy} maxSize={many ? 800 : 1400} />
          {!busy && (
            <div className="actions photo-actions">
              <button type="button" className="btn btn-quiet" onClick={() => onRotate(item.id, -90)}>
                <RotateLeftIcon />
                {lv.rotateLeft}
              </button>
              <button type="button" className="btn btn-quiet" onClick={() => onRotate(item.id, 90)}>
                <RotateRightIcon />
                {lv.rotateRight}
              </button>
              <button type="button" className="btn btn-quiet" onClick={() => onRemove(item.id)}>
                {lv.remove}
              </button>
            </div>
          )}
        </li>
      ))}
    </ol>
  );
}
