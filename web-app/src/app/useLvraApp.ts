import { useCallback, useEffect, useMemo, useReducer, useRef } from 'react';
import { config } from '../config/config';
import { loadImageFile, prepareLineImages, type LoadedImage } from '../image/imageProcessing';
import { createRecognitionService } from '../recognition/createRecognitionService';
import { toAppError } from '../recognition/errors';
import { mockSettings } from '../recognition/mockSettings';
import { recognizeHandwriting } from '../recognition/recognizeHandwriting';
import { buildSegments } from '../recognition/segments';
import { appReducer, initialState } from './appState';

/**
 * Connects the UI to the image and recognition layers.
 * trainingConsent is the user's choice in the terms window; it is sent to the
 * server, which keeps the line images for training only when it is true.
 */
export function useLvraApp(trainingConsent: boolean) {
  const service = useMemo(() => createRecognitionService(), []);
  const [state, dispatch] = useReducer(appReducer, initialState);
  const imageRef = useRef<LoadedImage | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  /** Deletes the image from memory. Called after recognition and whenever the user starts over. */
  const releaseImage = useCallback(() => {
    imageRef.current?.release();
    imageRef.current = null;
  }, []);

  useEffect(
    () => () => {
      abortRef.current?.abort();
      releaseImage();
    },
    [releaseImage],
  );

  const selectFile = useCallback(
    async (file: File | null) => {
      if (!file) {
        dispatch({ type: 'SELECTION_CANCELLED' });
        return;
      }
      releaseImage();
      dispatch({ type: 'LOAD_START' });
      try {
        const image = await loadImageFile(file);
        imageRef.current = image;
        dispatch({ type: 'LOAD_SUCCESS', image });
      } catch (error) {
        dispatch({ type: 'LOAD_FAILURE', error: toAppError(error).code });
      }
    },
    [releaseImage],
  );

  const rotate = useCallback((delta: 90 | -90) => dispatch({ type: 'ROTATE', delta }), []);

  const recognize = useCallback(async () => {
    if (state.step !== 'preview') return;
    const { image, rotation } = state;
    dispatch({ type: 'RECOGNIZE_START' });

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const lines = await prepareLineImages(image, rotation, { ...config.image, maxLines: config.lines.maxLines });
      const timeoutMs = mockSettings.timeoutOverrideMs || config.recognitionTimeoutMs;
      const result = await recognizeHandwriting(service, lines, {
        timeoutMs,
        trainingConsent,
        signal: controller.signal,
      });

      // Recognition succeeded: delete the image right away, keep only the text.
      releaseImage();
      const words = config.highlightUncertainWords ? result.words : [];
      dispatch({
        type: 'RECOGNIZE_SUCCESS',
        segments: buildSegments(result.text, words, config.uncertainConfidenceThreshold),
      });
    } catch (error) {
      const { code } = toAppError(error);
      if (code === 'CANCELLED') dispatch({ type: 'RECOGNIZE_CANCELLED' });
      else dispatch({ type: 'RECOGNIZE_FAILURE', error: code });
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  }, [state, service, releaseImage, trainingConsent]);

  const cancel = useCallback(() => abortRef.current?.abort(), []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    releaseImage();
    dispatch({ type: 'RESET' });
  }, [releaseImage]);

  return { state, selectFile, rotate, recognize, cancel, reset };
}
