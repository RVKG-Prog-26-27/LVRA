/**
 * The app's screens as a small state machine. Pure and side-effect free, so
 * it can be unit tested; releasing images and calling services happens in
 * useLvraApp.ts.
 *
 *   pick -> loading -> preview -> recognizing -> result
 *             ^  |        ^            |
 *             +--+        +-- error ---+   (all photos failed: photos kept for "Mēģināt vēlreiz")
 *        (adding more photos from the preview goes through loading and back)
 */
import type { LoadedImage } from '../image/imageProcessing';
import { rotateBy, type Rotation } from '../image/imageSize';
import type { AppErrorCode } from '../recognition/errors';
import type { TextSegment } from '../recognition/segments';

export interface PhotoItem {
  id: number;
  image: LoadedImage;
  rotation: Rotation;
  /** Size of the original file, for the total size limit. */
  bytes: number;
}

/** The result for one photo: its text, or the error that stopped it. */
export type PhotoResult = { ok: true; segments: TextSegment[] } | { ok: false; error: AppErrorCode };

export type Notice = 'cancelled' | 'sizeLimit' | null;

export type AppState =
  | { step: 'pick'; notice: Notice; error: AppErrorCode | null }
  | { step: 'loading'; items: PhotoItem[] }
  | {
      step: 'preview';
      items: PhotoItem[];
      /** Recognition failed for every photo (shown with "Mēģināt vēlreiz"). */
      error: AppErrorCode | null;
      /** A photo that was being added could not be opened. */
      loadError: AppErrorCode | null;
      notice: Notice;
    }
  | { step: 'recognizing'; items: PhotoItem[]; current: number }
  | { step: 'result'; results: PhotoResult[] };

export type AppAction =
  | { type: 'SELECTION_CANCELLED' }
  | { type: 'LOAD_START' }
  | { type: 'LOAD_DONE'; added: PhotoItem[]; skippedForSize: boolean; loadError: AppErrorCode | null }
  | { type: 'ROTATE'; id: number; delta: 90 | -90 }
  | { type: 'REMOVE'; id: number }
  | { type: 'RECOGNIZE_START' }
  | { type: 'RECOGNIZE_PROGRESS'; current: number }
  | { type: 'RECOGNIZE_DONE'; results: PhotoResult[] }
  | { type: 'RECOGNIZE_FAILURE'; error: AppErrorCode }
  | { type: 'RECOGNIZE_CANCELLED' }
  | { type: 'RESET' };

export const initialState: AppState = { step: 'pick', notice: null, error: null };

function currentItems(state: AppState): PhotoItem[] {
  return state.step === 'loading' || state.step === 'preview' || state.step === 'recognizing' ? state.items : [];
}

function preview(items: PhotoItem[], extra: Partial<Extract<AppState, { step: 'preview' }>> = {}): AppState {
  return { step: 'preview', items, error: null, loadError: null, notice: null, ...extra };
}

export function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SELECTION_CANCELLED':
      // On the start screen explain how to allow camera/gallery access; elsewhere nothing changes.
      return state.step === 'pick' ? { step: 'pick', notice: 'cancelled', error: null } : state;
    case 'LOAD_START':
      return state.step === 'pick' || state.step === 'preview' ? { step: 'loading', items: currentItems(state) } : state;
    case 'LOAD_DONE': {
      if (state.step !== 'loading') return state;
      const items = [...state.items, ...action.added];
      const notice: Notice = action.skippedForSize ? 'sizeLimit' : null;
      if (items.length === 0) return { step: 'pick', notice, error: action.loadError };
      return preview(items, { loadError: action.loadError, notice });
    }
    case 'ROTATE':
      return state.step === 'preview'
        ? preview(
            state.items.map((item) =>
              item.id === action.id ? { ...item, rotation: rotateBy(item.rotation, action.delta) } : item,
            ),
          )
        : state;
    case 'REMOVE': {
      if (state.step !== 'preview') return state;
      const items = state.items.filter((item) => item.id !== action.id);
      return items.length === 0 ? initialState : preview(items);
    }
    case 'RECOGNIZE_START':
      return state.step === 'preview' && state.items.length > 0
        ? { step: 'recognizing', items: state.items, current: 0 }
        : state;
    case 'RECOGNIZE_PROGRESS':
      return state.step === 'recognizing' ? { ...state, current: action.current } : state;
    case 'RECOGNIZE_DONE':
      return state.step === 'recognizing' ? { step: 'result', results: action.results } : state;
    case 'RECOGNIZE_FAILURE':
      return state.step === 'recognizing' ? preview(state.items, { error: action.error }) : state;
    case 'RECOGNIZE_CANCELLED':
      return state.step === 'recognizing' ? preview(state.items) : state;
    case 'RESET':
      return initialState;
    default:
      return state;
  }
}
