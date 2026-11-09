import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { PhotoResult } from '../app/appState';
import { lv } from '../i18n/lv';
import { copyToClipboard } from '../utils/clipboard';
import { ErrorMessage } from './ErrorMessage';
import { CopyIcon } from './Icons';
import { ResultEditor } from './ResultEditor';

interface Props {
  results: PhotoResult[];
  onStartOver: () => void;
}

type CopyState = 'idle' | 'copied' | 'failed';

/**
 * The recognised text. One photo: a single editor. Several photos: one tab
 * per photo ("1. attēls", "2. attēls", ...); a photo that failed shows its
 * error in its tab. All editors stay mounted, so edits survive switching tabs.
 */
export function ResultTabs({ results, onStartOver }: Props) {
  const many = results.length > 1;
  const [active, setActive] = useState(0);
  const textsRef = useRef<string[]>(results.map(() => ''));
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const [copyAllState, setCopyAllState] = useState<CopyState>('idle');

  // One stable callback per photo, so editors do not re-render their text on every change.
  const onTextChange = useMemo(
    () =>
      results.map((_, i) => (text: string) => {
        textsRef.current[i] = text;
      }),
    [results],
  );

  const anyText = results.some((r) => r.ok);

  const handleCopyAll = async () => {
    const parts = results
      .map((r, i) => (r.ok ? textsRef.current[i].replace(/\s+$/, '') : null))
      .filter((t): t is string => t !== null && t.trim() !== '');
    const ok = await copyToClipboard(parts.join('\n\n'));
    setCopyAllState(ok ? 'copied' : 'failed');
  };

  useEffect(() => {
    if (copyAllState === 'idle') return;
    const timer = setTimeout(() => setCopyAllState('idle'), 2500);
    return () => clearTimeout(timer);
  }, [copyAllState]);

  // Left and right arrow keys move between tabs (standard tab behaviour).
  const onTabKey = useCallback(
    (event: React.KeyboardEvent, index: number) => {
      if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
      event.preventDefault();
      const next = (index + (event.key === 'ArrowRight' ? 1 : -1) + results.length) % results.length;
      setActive(next);
      tabRefs.current[next]?.focus();
    },
    [results.length],
  );

  return (
    <section className="result" aria-labelledby="result-title">
      <h2 id="result-title">{lv.resultTitle}</h2>
      <p className="muted">{lv.resultHint}</p>

      {many && (
        <div className="tabs" role="tablist" aria-label={lv.resultTitle}>
          {results.map((r, i) => (
            <button
              key={i}
              ref={(el) => {
                tabRefs.current[i] = el;
              }}
              type="button"
              role="tab"
              id={`photo-tab-${i}`}
              aria-selected={active === i}
              aria-controls={`photo-panel-${i}`}
              tabIndex={active === i ? 0 : -1}
              className={`tab${active === i ? ' is-active' : ''}${r.ok ? '' : ' has-error'}`}
              onClick={() => setActive(i)}
              onKeyDown={(e) => onTabKey(e, i)}
            >
              {lv.photoLabel(i + 1)}
            </button>
          ))}
        </div>
      )}

      {results.map((r, i) => (
        <div
          key={i}
          id={`photo-panel-${i}`}
          role={many ? 'tabpanel' : undefined}
          aria-labelledby={many ? `photo-tab-${i}` : undefined}
          hidden={active !== i}
          className="tab-panel"
        >
          {r.ok ? (
            <ResultEditor
              segments={r.segments}
              label={many ? `${lv.editorLabel}, ${lv.photoLabel(i + 1)}` : lv.editorLabel}
              onTextChange={onTextChange[i]}
            />
          ) : (
            <ErrorMessage code={r.error} withTitle />
          )}
        </div>
      ))}

      <div className="actions result-actions">
        {many && anyText && (
          <button type="button" className="btn btn-secondary" onClick={handleCopyAll}>
            <CopyIcon />
            {lv.copyAll}
          </button>
        )}
        <button type="button" className="btn btn-quiet" onClick={onStartOver}>
          {lv.startOver}
        </button>
      </div>
      {many && (
        <p className={`toast${copyAllState === 'failed' ? ' toast-error' : ''}`} role="status" aria-live="polite">
          {copyAllState === 'copied' ? lv.copied : copyAllState === 'failed' ? lv.copyFailed : ''}
        </p>
      )}
    </section>
  );
}
