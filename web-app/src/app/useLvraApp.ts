import { useCallback, useEffect, useMemo, useReducer, useRef } from 'react';
import { config } from '../config/config';
import { loadImageFile, prepareLineImages } from '../image/imageProcessing';
import { createRecognitionService } from '../recognition/createRecognitionService';
import { devInspector } from '../recognition/devInspector';
import { toAppError, type AppErrorCode } from '../recognition/errors';
import { mockSettings } from '../recognition/mockSettings';
import { recognizeHandwriting } from '../recognition/recognizeHandwriting';
import { buildSegments } from '../recognition/segments';
import { appReducer, initialState, type PhotoItem, type PhotoResult } from './appState';

/**
 * Connects the UI to the image and recognition layers.
 * trainingConsent is the user's choice in the terms window; it is sent to the
 * server, which keeps the line images for training only when it is true.
 */
export function useLvraApp(trainingConsent: boolean) {
  const service = useMemo(() => createRecognitionService(), []);
  const [state, dispatch] = useReducer(appReducer, initialState);
  /** The photos currently held in memory, so they can always be released. */
  const itemsRef = useRef<PhotoItem[]>([]);
  const nextIdRef = useRef(1);
  const abortRef = useRef<AbortController | null>(null);

  const releaseItems = useCallback((items: PhotoItem[]) => {
    for (const item of items) item.image.release();
    const ids = new Set(items.map((i) => i.id));
    itemsRef.current = itemsRef.current.filter((i) => !ids.has(i.id));
  }, []);

  /** Deletes every photo from memory. Called after recognition and whenever the user starts over. */
  const releaseAll = useCallback(() => releaseItems([...itemsRef.current]), [releaseItems]);

  useEffect(
    () => () => {
      abortRef.current?.abort();
      releaseAll();
    },
    [releaseAll],
  );

  /** Adds the chosen photos (one from the camera, or several from the gallery) in the order given. */
  const selectFiles = useCallback(
    async (files: File[] | null) => {
      if (!files || files.length === 0) {
        dispatch({ type: 'SELECTION_CANCELLED' });
        return;
      }
      dispatch({ type: 'LOAD_START' });

      let total = itemsRef.current.reduce((sum, item) => sum + item.bytes, 0);
      const added: PhotoItem[] = [];
      let skippedForSize = false;
      let loadError: AppErrorCode | null = null;

      for (const file of files) {
        if (total + file.size > config.photos.maxTotalBytes) {
          skippedForSize = true;
          continue;
        }
        try {
          const image = await loadImageFile(file, config.image.maxLongSide, config.image.maxShortSide);
          const item: PhotoItem = { id: nextIdRef.current++, image, rotation: 0, bytes: file.size };
          added.push(item);
          itemsRef.current.push(item);
          total += file.size;
        } catch (error) {
          loadError = toAppError(error).code;
        }
      }
      dispatch({ type: 'LOAD_DONE', added, skippedForSize, loadError });
    },
    [],
  );

  const rotate = useCallback((id: number, delta: 90 | -90) => dispatch({ type: 'ROTATE', id, delta }), []);

  const remove = useCallback(
    (id: number) => {
      const item = itemsRef.current.find((i) => i.id === id);
      if (item) releaseItems([item]);
      dispatch({ type: 'REMOVE', id });
    },
    [releaseItems],
  );

  const recognize = useCallback(async () => {
    if (state.step !== 'preview' || state.items.length === 0) return;
    const { items } = state;
    dispatch({ type: 'RECOGNIZE_START' });

    const controller = new AbortController();
    abortRef.current = controller;
    devInspector.startBatch();
    const timeoutMs = mockSettings.timeoutOverrideMs || config.recognitionTimeoutMs;
    const results: PhotoResult[] = [];

    try {
      // One photo after another, each as its own request with its own 0001.png, 0002.png, ...
      for (let i = 0; i < items.length; i++) {
        dispatch({ type: 'RECOGNIZE_PROGRESS', current: i });
        try {
          const { image, rotation } = items[i];
          const lines = await prepareLineImages(image, rotation, { ...config.image, maxLines: config.lines.maxLines });
          const result = await recognizeHandwriting(service, lines, {
            timeoutMs,
            trainingConsent,
            signal: controller.signal,
          });
          const words = config.highlightUncertainWords ? result.words : [];
          results.push({ ok: true, segments: buildSegments(result.text, words, config.uncertainConfidenceThreshold) });
        } catch (error) {
          const { code } = toAppError(error);
          if (code === 'CANCELLED') {
            dispatch({ type: 'RECOGNIZE_CANCELLED' });
            return;
          }
          // This photo failed; the others are still recognised.
          results.push({ ok: false, error: code });
        }
      }

      const firstError = results.find((r): r is { ok: false; error: AppErrorCode } => !r.ok);
      if (firstError && results.every((r) => !r.ok)) {
        // Nothing worked (for example no internet): keep the photos so the user can retry.
        dispatch({ type: 'RECOGNIZE_FAILURE', error: firstError.error });
        return;
      }
      // Recognition done: delete the photos right away, keep only the text.
      releaseAll();
      dispatch({ type: 'RECOGNIZE_DONE', results });
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  }, [state, service, releaseAll, trainingConsent]);

  const cancel = useCallback(() => abortRef.current?.abort(), []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    releaseAll();
    dispatch({ type: 'RESET' });
  }, [releaseAll]);

  return { state, selectFiles, rotate, remove, recognize, cancel, reset };
}
