import { MockRecognitionService } from './mockRecognitionService';
import type { HandwritingRecognitionService } from './types';

// decides which recognition module the app uses
export function createRecognitionService(): HandwritingRecognitionService {
  return new MockRecognitionService();
}
