import { useCallback, useEffect, useMemo, useReducer, useRef } from 'react';
import { config } from '../config/config';
import { loadImageFile, prepareImageForRecognition, type LoadedImage } from '../image/imageProcessing';
import { createRecognitionService, isMockMode } from '../recognition/createRecognitionService';
import { toAppError } from '../recognition/errors';
import { mockSettings } from '../recognition/mockSettings';
import { recognizeHandwriting } from '../recognition/recognizeHandwriting';
import { buildSegments } from '../recognition/segments';
import { appReducer, initialState } from './appState';

/** Connects the UI to the image and recognition layers. */
export function useLvraApp() {
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
      const jpeg = await prepareImageForRecognition(image, rotation, config.image);
      const timeoutMs = (isMockMode && mockSettings.timeoutOverrideMs) || config.recognitionTimeoutMs;
      const result = await recognizeHandwriting(service, jpeg, { timeoutMs, signal: controller.signal });

      // Recognition succeeded: delete the image right away, keep only the text.
      releaseImage();
      dispatch({
        type: 'RECOGNIZE_SUCCESS',
        segments: buildSegments(result.text, result.words, config.uncertainConfidenceThreshold),
      });
    } catch (error) {
      const { code } = toAppError(error);
      if (code === 'CANCELLED') dispatch({ type: 'RECOGNIZE_CANCELLED' });
      else dispatch({ type: 'RECOGNIZE_FAILURE', error: code });
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  }, [state, service, releaseImage]);

  const cancel = useCallback(() => abortRef.current?.abort(), []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    releaseImage();
    dispatch({ type: 'RESET' });
  }, [releaseImage]);

  return { state, selectFile, rotate, recognize, cancel, reset };
}
