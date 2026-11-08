/**
<<<<<<< HEAD
 * Settings changed from the developer panel (open the app with ?dev=1).
 * Development only: normal users always get config.recognition.defaultBackend.
=======
 * TEMPORARY
 * Settings for the mock recognition service, changed from the developer
 * panel (open the app with ?dev=1). Delete together with the mock
>>>>>>> origin/web-app
 */
import { config } from '../config/config';

export type Backend = 'server' | 'mock';

export type MockScenario =
  | 'success'
  | 'NO_TEXT_FOUND'
  | 'POOR_QUALITY'
  | 'INVALID_IMAGE'
  | 'SERVICE_UNAVAILABLE'
  | 'hang'
  | 'UNKNOWN';

export interface MockSettings {
  /** Which recognition backend is used: the real AI server or the mock. */
  backend: Backend;
  scenario: MockScenario;
  /** Simulated processing time */
  delayMs: number;
  /** When set, overrides the normal 60 s timeout so the timeout error can be tested quickly */
  timeoutOverrideMs: number | null;
}

export const mockSettings: MockSettings = {
  backend: config.recognition.defaultBackend,
  scenario: 'success',
  delayMs: 1500,
  timeoutOverrideMs: null,
};

export const MOCK_SCENARIO_LABELS: Record<MockScenario, string> = {
  success: 'Success (sample text, one line per PNG)',
  NO_TEXT_FOUND: 'Error: no handwriting found',
  POOR_QUALITY: 'Error: poor image quality',
  INVALID_IMAGE: 'Error: image rejected',
  SERVICE_UNAVAILABLE: 'Error: service unavailable',
  hang: 'Never answers (tests timeout)',
  UNKNOWN: 'Error: unexpected',
};
