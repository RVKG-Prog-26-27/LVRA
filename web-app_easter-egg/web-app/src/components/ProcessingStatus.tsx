import { lv } from '../i18n/lv';

export function ProcessingStatus({ onCancel }: { onCancel: () => void }) {
  return (
    <div className="processing" role="status" aria-live="polite">
      <span className="ink-line" aria-hidden="true" />
      <p className="processing-title">{lv.processing}</p>
      <p className="muted">{lv.processingHint}</p>
      <button type="button" className="btn btn-quiet" onClick={onCancel}>
        {lv.cancel}
      </button>
    </div>
  );
}
