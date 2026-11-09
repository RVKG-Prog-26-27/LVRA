import { lv } from '../i18n/lv';

interface Props {
  onCancel: () => void;
  /** 1-based number of the photo being processed. */
  current: number;
  total: number;
}

export function ProcessingStatus({ onCancel, current, total }: Props) {
  return (
    <div className="processing processing-inline" role="status" aria-live="polite">
      <span className="ink-line" aria-hidden="true" />
      <p className="processing-title">{total > 1 ? lv.processingMany(current, total) : lv.processing}</p>
      <p className="muted">{lv.processingHint}</p>
      <button type="button" className="btn btn-quiet" onClick={onCancel}>
        {lv.cancel}
      </button>
    </div>
  );
}
