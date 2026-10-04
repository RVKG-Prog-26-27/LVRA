import { MockRecognitionService } from './mockRecognitionService';
import type { HandwritingRecognitionService } from './types';

/**
 * The one place that decides which recognition backend the app uses.
 * Alpha version: the handwriting model is not connected yet, so the app
 * always uses the temporary mock. To connect the real model later, return
 * another implementation of HandwritingRecognitionService here.
 */
export function createRecognitionService(): HandwritingRecognitionService {
  return new MockRecognitionService();
}
