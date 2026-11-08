import { MockRecognitionService } from './mockRecognitionService';
import { mockSettings } from './mockSettings';
import { ServerRecognitionService } from './serverRecognitionService';
import type { HandwritingRecognitionService } from './types';

/**
 * The one place that decides which recognition backend the app uses.
 * Normally the real AI server (config.recognition.defaultBackend). The
 * developer panel can switch to the mock at any time, so the choice is read
 * on every call.
 */
export function createRecognitionService(): HandwritingRecognitionService {
  const server = new ServerRecognitionService();
  const mock = new MockRecognitionService();
  return {
    recognize: (request, options) =>
      (mockSettings.backend === 'mock' ? mock : server).recognize(request, options),
  };
}
