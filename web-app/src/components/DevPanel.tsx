/**
 * TEMPORARY developer panel for testing with the mock service.
 * Visible only when the address contains ?dev=1. Intentionally in English:
 * it is a developer tool, not part of the user-facing app.
 * Remove together with the mock once the real model is connected.
 */
import { useState } from 'react';
import { config } from '../config/config';
import { isMockMode } from '../recognition/createRecognitionService';
import { MOCK_SCENARIO_LABELS, mockSettings, type MockScenario } from '../recognition/mockSettings';

export function DevPanel() {
  const [, forceRender] = useState(0);
  const update = (change: Partial<typeof mockSettings>) => {
    Object.assign(mockSettings, change);
    forceRender((n) => n + 1);
  };

  return (
    <details className="devpanel" lang="en">
      <summary>Developer panel</summary>
      <p>
        Mode: <strong>{config.recognitionMode}</strong>
        {config.recognitionMode === 'api' && <> ({config.recognitionApiUrl || 'API URL not set'})</>}
      </p>
      {isMockMode ? (
        <>
          <label>
            Mock scenario
            <select
              value={mockSettings.scenario}
              onChange={(e) => update({ scenario: e.target.value as MockScenario })}
            >
              {Object.entries(MOCK_SCENARIO_LABELS).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Simulated delay (ms)
            <input
              type="number"
              min={0}
              step={250}
              value={mockSettings.delayMs}
              onChange={(e) => update({ delayMs: Math.max(0, Number(e.target.value) || 0) })}
            />
          </label>
          <label>
            Timeout override (s, empty = 60)
            <input
              type="number"
              min={1}
              value={mockSettings.timeoutOverrideMs ? mockSettings.timeoutOverrideMs / 1000 : ''}
              onChange={(e) =>
                update({ timeoutOverrideMs: e.target.value ? Math.max(1, Number(e.target.value)) * 1000 : null })
              }
            />
          </label>
        </>
      ) : (
        <p>Mock controls are disabled because the real API is active.</p>
      )}
    </details>
  );
}
