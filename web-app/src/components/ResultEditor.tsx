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
  /** Accessible name of the editor, e.g. "Atpazītais teksts, ko var labot, 2. attēls". */
  label: string;
  /** Called with the current text whenever it changes, so "Kopēt visu" can use edited text. */
  onTextChange: (text: string) => void;
}

type CopyState = 'idle' | 'copied' | 'failed';

/** The editable text of one photo with its own "Kopēt tekstu" button. */
export function ResultEditor({ segments, label, onTextChange }: Props) {
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
    const text = extractText(editor);
    setIsEmpty(text.trim() === '');
    onTextChange(text);
  }, [segments, onTextChange]);

  const syncAfterEdit = useCallback(() => {
    const editor = editorRef.current;
    if (!editor) return;
    const caret = getCaretOffset(editor);
    if (unwrapEditedMarks(editor) && caret !== null) setCaretOffset(editor, caret);
    ensureSentinel(editor);
    setHasUncertain(editor.querySelector('mark.uncertain') !== null);
    const text = extractText(editor);
    setIsEmpty(text.trim() === '');
    onTextChange(text);
    setCopyState('idle');
  }, [onTextChange]);

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
    <>
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
        aria-label={label}
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
      </div>
      <p className={`toast${copyState === 'failed' ? ' toast-error' : ''}`} role="status" aria-live="polite">
        {copyState === 'copied' ? lv.copied : copyState === 'failed' ? lv.copyFailed : ''}
      </p>
    </>
  );
}
