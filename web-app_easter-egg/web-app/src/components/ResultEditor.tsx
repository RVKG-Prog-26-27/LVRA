import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import {
  ensureSentinel,
  extractText,
  getCaretOffset,
  insertPlainText,
  renderSegments,
  setCaretOffset,
  unwrapEditedMarks,
} from '../editor/editorDom';
import { lv } from '../i18n/lv';
import type { TextSegment } from '../recognition/segments';
import { copyToClipboard } from '../utils/clipboard';
import { CopyIcon } from './Icons';

interface Props {
  segments: TextSegment[];
  onStartOver: () => void;
}

type CopyState = 'idle' | 'copied' | 'failed';

export function ResultEditor({ segments, onStartOver }: Props) {
  const editorRef = useRef<HTMLDivElement>(null);
  const [isEmpty, setIsEmpty] = useState(false);
  const [hasUncertain, setHasUncertain] = useState(segments.some((s) => s.uncertain));
  const [copyState, setCopyState] = useState<CopyState>('idle');

  // The editor content is managed directly in the DOM (not by React) so the caret is never reset.
  useLayoutEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;
    renderSegments(editor, segments);
    setHasUncertain(segments.some((s) => s.uncertain));
    setIsEmpty(extractText(editor).trim() === '');
  }, [segments]);

  const syncAfterEdit = useCallback(() => {
    const editor = editorRef.current;
    if (!editor) return;
    const caret = getCaretOffset(editor);
    if (unwrapEditedMarks(editor) && caret !== null) setCaretOffset(editor, caret);
    ensureSentinel(editor);
    setHasUncertain(editor.querySelector('mark.uncertain') !== null);
    setIsEmpty(extractText(editor).trim() === '');
    setCopyState('idle');
  }, []);

  // Keep the content plain text: Enter inserts "\n", formatting and dropped HTML are blocked.
  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;
    const onBeforeInput = (event: InputEvent) => {
      if (event.inputType === 'insertParagraph' || event.inputType === 'insertLineBreak') {
        event.preventDefault();
        insertPlainText(editor, '\n');
        syncAfterEdit();
      } else if (event.inputType.startsWith('format') || event.inputType === 'insertFromDrop') {
        event.preventDefault();
      }
    };
    const onPaste = (event: ClipboardEvent) => {
      event.preventDefault();
      const text = (event.clipboardData?.getData('text/plain') ?? '').replace(/\r\n?/g, '\n');
      insertPlainText(editor, text);
      syncAfterEdit();
    };
    editor.addEventListener('beforeinput', onBeforeInput);
    editor.addEventListener('paste', onPaste);
    return () => {
      editor.removeEventListener('beforeinput', onBeforeInput);
      editor.removeEventListener('paste', onPaste);
    };
  }, [syncAfterEdit]);

  useEffect(() => {
    if (copyState === 'idle') return;
    const timer = setTimeout(() => setCopyState('idle'), 2500);
    return () => clearTimeout(timer);
  }, [copyState]);

  const handleCopy = async () => {
    const editor = editorRef.current;
    if (!editor) return;
    const ok = await copyToClipboard(extractText(editor));
    setCopyState(ok ? 'copied' : 'failed');
  };

  return (
    <section className="result" aria-labelledby="result-title">
      <h2 id="result-title">{lv.resultTitle}</h2>
      <p className="muted">{lv.resultHint}</p>
      {hasUncertain && (
        <p className="legend">
          <mark className="uncertain legend-swatch">vārds</mark>
          {lv.uncertainLegend}
        </p>
      )}

      <div
        ref={editorRef}
        className="editor"
        contentEditable
        suppressContentEditableWarning
        role="textbox"
        aria-multiline="true"
        aria-label={lv.editorLabel}
        spellCheck={false}
        lang="lv"
        onInput={syncAfterEdit}
      />
      {isEmpty && <p className="muted">{lv.emptyText}</p>}

      <div className="actions">
        <button type="button" className="btn btn-primary" onClick={handleCopy} disabled={isEmpty}>
          <CopyIcon />
          {lv.copy}
        </button>
        <button type="button" className="btn btn-quiet" onClick={onStartOver}>
          {lv.startOver}
        </button>
      </div>
      <p className={`toast${copyState === 'failed' ? ' toast-error' : ''}`} role="status" aria-live="polite">
        {copyState === 'copied' ? lv.copied : copyState === 'failed' ? lv.copyFailed : ''}
      </p>
    </section>
  );
}
