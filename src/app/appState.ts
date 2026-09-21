/**
 * The app's screens as a small state machine. Pure and side-effect free, so
 * it can be unit tested; releasing images and calling services happens in
 * useLvraApp.ts.
 *
 *   pick -> loading -> preview -> recognizing -> result
 *                        ^            |
 *                        +-- error ---+   (image kept for "Mēģināt vēlreiz")
 */
import type { LoadedImage } from '../image/imageProcessing';
import { rotateBy, type Rotation } from '../image/imageSize';
import type { AppErrorCode } from '../recognition/errors';
import type { TextSegment } from '../recognition/segments';

export type AppState =
  | { step: 'pick'; notice: 'cancelled' | null; error: AppErrorCode | null }
  | { step: 'loading' }
  | { step: 'preview'; image: LoadedImage; rotation: Rotation; error: AppErrorCode | null }
  | { step: 'recognizing'; image: LoadedImage; rotation: Rotation }
  | { step: 'result'; segments: TextSegment[] };

export type AppAction =
  | { type: 'SELECTION_CANCELLED' }
  | { type: 'LOAD_START' }
  | { type: 'LOAD_SUCCESS'; image: LoadedImage }
  | { type: 'LOAD_FAILURE'; error: AppErrorCode }
  | { type: 'ROTATE'; delta: 90 | -90 }
  | { type: 'RECOGNIZE_START' }
  | { type: 'RECOGNIZE_SUCCESS'; segments: TextSegment[] }
  | { type: 'RECOGNIZE_FAILURE'; error: AppErrorCode }
  | { type: 'RECOGNIZE_CANCELLED' }
  | { type: 'RESET' };

export const initialState: AppState = { step: 'pick', notice: null, error: null };

export function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SELECTION_CANCELLED':
      // Only relevant on the start screen; on later screens the current image stays.
      return state.step === 'pick' ? { step: 'pick', notice: 'cancelled', error: null } : state;
    case 'LOAD_START':
      return { step: 'loading' };
    case 'LOAD_SUCCESS':
      return { step: 'preview', image: action.image, rotation: 0, error: null };
    case 'LOAD_FAILURE':
      return { step: 'pick', notice: null, error: action.error };
    case 'ROTATE':
      return state.step === 'preview'
        ? { ...state, rotation: rotateBy(state.rotation, action.delta), error: null }
        : state;
    case 'RECOGNIZE_START':
      return state.step === 'preview'
        ? { step: 'recognizing', image: state.image, rotation: state.rotation }
        : state;
    case 'RECOGNIZE_SUCCESS':
      return state.step === 'recognizing' ? { step: 'result', segments: action.segments } : state;
    case 'RECOGNIZE_FAILURE':
      return state.step === 'recognizing'
        ? { step: 'preview', image: state.image, rotation: state.rotation, error: action.error }
        : state;
    case 'RECOGNIZE_CANCELLED':
      return state.step === 'recognizing'
        ? { step: 'preview', image: state.image, rotation: state.rotation, error: null }
        : state;
    case 'RESET':
      return initialState;
    default:
      return state;
  }
}
