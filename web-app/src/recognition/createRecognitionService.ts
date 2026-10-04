import { config } from '../config/config';
import { ApiRecognitionService } from './apiRecognitionService';
import { MockRecognitionService } from './mockRecognitionService';
import type { HandwritingRecognitionService } from './types';

/**
 * The one place that decides which recognition backend the app uses.
 * To integrate the real model: set VITE_RECOGNITION_MODE=api and
 * VITE_RECOGNITION_API_URL in .env, or return your own implementation here.
 */
export function createRecognitionService(): HandwritingRecognitionService {
  if (config.recognitionMode === 'api') {
    return new ApiRecognitionService(config.recognitionApiUrl);
  }
  return new MockRecognitionService();
}

export const isMockMode = config.recognitionMode === 'mock';
