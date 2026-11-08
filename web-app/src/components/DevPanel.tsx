/**
 * Developer panel, visible only when the address contains ?dev=1.
 * Intentionally in English: it is a developer tool, not part of the
 * user-facing app.
 *
 * - Switch between the real AI server and the mock.
 * - Mock scenarios, simulated delay, short timeout.
 * - The last request: every line PNG the browser cut out, the labels.csv
 *   that was sent and the labels.csv that came back.
 */
import { useState, useSyncExternalStore } from 'react';
import { devInspector } from '../recognition/devInspector';
import { MOCK_SCENARIO_LABELS, mockSettings, type Backend, type MockScenario } from '../recognition/mockSettings';

interface Props {
  backend: Backend;
  onBackendChange: (backend: Backend) => void;
}

export function DevPanel({ backend, onBackendChange }: Props) {
  const [, forceRender] = useState(0);
  const inspection = useSyncExternalStore(devInspector.subscribe, devInspector.get);

  const update = (change: Partial<typeof mockSettings>) => {
    Object.assign(mockSettings, change);
    if (change.backend) onBackendChange(change.backend);
    forceRender((n) => n + 1);
  };

  return (
    <details className="devpanel" lang="en" open>
      <summary>Developer panel</summary>

      <fieldset>
        <legend>Recognition backend</legend>
        <label className="inline">
          <input
            type="radio"
            name="backend"
            checked={backend === 'server'}
            onChange={() => update({ backend: 'server' })}
          />
          Real AI server (/api/recognize)
        </label>
        <label className="inline">
          <input type="radio" name="backend" checked={backend === 'mock'} onChange={() => update({ backend: 'mock' })} />
          Mock (sample text)
        </label>
      </fieldset>

      {backend === 'mock' && (
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
        </>
      )}
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

      <h3>Last request</h3>
      {!inspection ? (
        <p>Nothing sent yet. Choose an image and press "Atpazīt tekstu".</p>
      ) : (
        <>
          <p>
            Backend: <strong>{inspection.backend}</strong> · lines: <strong>{inspection.lines.length}</strong> ·
            training_consent: <strong>{String(inspection.trainingConsent)}</strong>
          </p>
          <ol className="devlines">
            {inspection.lines.map((line) => (
              <li key={line.filename}>
                <span>{line.filename}</span>
                <img src={line.url} alt={line.filename} />
              </li>
            ))}
          </ol>
          <p>labels.csv sent:</p>
          <pre>{inspection.sentCsv}</pre>
          <p>labels.csv received:</p>
          <pre>{inspection.receivedCsv ?? '(waiting or failed)'}</pre>
        </>
      )}
    </details>
  );
}
