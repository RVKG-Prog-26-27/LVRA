import type { LoadedImage } from '../image/imageProcessing';
import { appReducer, initialState, type AppState } from './appState';

const image = { size: { width: 10, height: 10 }, source: {} as CanvasImageSource, release: () => {} } as LoadedImage;
const preview: AppState = { step: 'preview', image, rotation: 0, error: null };

describe('appReducer', () => {
  it('goes from loading to preview with no rotation', () => {
    const state = appReducer(appReducer(initialState, { type: 'LOAD_START' }), { type: 'LOAD_SUCCESS', image });
    expect(state).toEqual(preview);
  });

  it('keeps the image after a failed recognition so the user can retry', () => {
    const recognizing = appReducer(preview, { type: 'RECOGNIZE_START' });
    const failed = appReducer(recognizing, { type: 'RECOGNIZE_FAILURE', error: 'TIMEOUT' });
    expect(failed).toEqual({ ...preview, error: 'TIMEOUT' });
  });

  it('drops the image once text is recognised', () => {
    const recognizing = appReducer(preview, { type: 'RECOGNIZE_START' });
    const result = appReducer(recognizing, { type: 'RECOGNIZE_SUCCESS', segments: [{ text: 'a', uncertain: false }] });
    expect(result).toEqual({ step: 'result', segments: [{ text: 'a', uncertain: false }] });
    expect('image' in result).toBe(false);
  });

  it('rotates only on the preview screen', () => {
    expect(appReducer(preview, { type: 'ROTATE', delta: -90 })).toMatchObject({ rotation: 270 });
    expect(appReducer(initialState, { type: 'ROTATE', delta: 90 })).toBe(initialState);
  });

  it('shows a hint when the picker is closed without a file', () => {
    expect(appReducer(initialState, { type: 'SELECTION_CANCELLED' })).toMatchObject({ notice: 'cancelled' });
  });
});
