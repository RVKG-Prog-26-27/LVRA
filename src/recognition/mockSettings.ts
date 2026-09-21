/**
 * TEMPORARY, for development only.
 * Settings for the mock recognition service, changed from the developer
 * panel (open the app with ?dev=1). Delete together with the mock.
 */

export type MockScenario =
  | 'success'
  | 'success-confident'
  | 'NO_TEXT_FOUND'
  | 'POOR_QUALITY'
  | 'INVALID_IMAGE'
  | 'SERVICE_UNAVAILABLE'
  | 'hang'
  | 'UNKNOWN';

export interface MockSettings {
  scenario: MockScenario;
  /** Simulated processing time. */
  delayMs: number;
  /** When set, overrides the normal 60 s timeout so the timeout error can be tested quickly. */
  timeoutOverrideMs: number | null;
}

export const mockSettings: MockSettings = {
  scenario: 'success',
  delayMs: 1500,
  timeoutOverrideMs: null,
};

export const MOCK_SCENARIO_LABELS: Record<MockScenario, string> = {
  success: 'Success with uncertain words',
  'success-confident': 'Success, all words confident',
  NO_TEXT_FOUND: 'Error: no handwriting found',
  POOR_QUALITY: 'Error: poor image quality',
  INVALID_IMAGE: 'Error: image rejected',
  SERVICE_UNAVAILABLE: 'Error: service unavailable',
  hang: 'Never answers (tests timeout)',
  UNKNOWN: 'Error: unexpected',
};
