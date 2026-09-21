import { MockRecognitionService } from './mockRecognitionService';
import type { MockSettings } from './mockSettings';
import { recognizeHandwriting } from './recognizeHandwriting';

const image = new Blob([new Uint8Array([1])], { type: 'image/jpeg' });
const settings = (change: Partial<MockSettings>): MockSettings => ({
  scenario: 'success',
  delayMs: 10,
  timeoutOverrideMs: null,
  ...change,
});

describe('recognizeHandwriting with the temporary mock', () => {
  it('returns the sample text with uncertain words', async () => {
    const result = await recognizeHandwriting(new MockRecognitionService(settings({})), image, { timeoutMs: 1000 });
    expect(result.text).toContain('\n');
    expect(result.words.some((w) => w.confidence < 0.7)).toBe(true);
  });

  it('passes simulated errors through', async () => {
    const service = new MockRecognitionService(settings({ scenario: 'POOR_QUALITY' }));
    await expect(recognizeHandwriting(service, image, { timeoutMs: 1000 })).rejects.toMatchObject({
      code: 'POOR_QUALITY',
    });
  });

  it('stops with a timeout error when the service never answers', async () => {
    const service = new MockRecognitionService(settings({ scenario: 'hang' }));
    await expect(recognizeHandwriting(service, image, { timeoutMs: 30 })).rejects.toMatchObject({ code: 'TIMEOUT' });
  });

  it('reports cancellation when the user presses cancel', async () => {
    const controller = new AbortController();
    const service = new MockRecognitionService(settings({ delayMs: 500 }));
    const promise = recognizeHandwriting(service, image, { timeoutMs: 1000, signal: controller.signal });
    controller.abort();
    await expect(promise).rejects.toMatchObject({ code: 'CANCELLED' });
  });

  it('refuses to run without an image', async () => {
    await expect(
      recognizeHandwriting(new MockRecognitionService(settings({})), null, { timeoutMs: 1000 }),
    ).rejects.toMatchObject({ code: 'NO_IMAGE' });
  });
});
