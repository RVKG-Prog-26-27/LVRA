/**
 * Remembers the user's answer in the terms window, so it is not shown on
 * every visit. Stored in the browser's localStorage together with the terms
 * version: when the terms change, the stored answer no longer counts and the
 * window opens again.
 *
 * Storage can be blocked (private mode, cleared site data). Every access is
 * wrapped in try/catch; without storage the window simply opens each visit.
 */
import { config } from '../config/config';

export interface ConsentChoice {
  /** true = "Piekrītu visiem noteikumiem", false = "Piekrītu tikai obligātajiem". */
  trainingConsent: boolean;
}

interface StoredChoice extends ConsentChoice {
  version: string;
  decidedAt: string;
}

export function loadConsent(): ConsentChoice | null {
  try {
    const raw = window.localStorage.getItem(config.terms.storageKey);
    if (!raw) return null;
    const stored = JSON.parse(raw) as Partial<StoredChoice>;
    if (stored.version !== config.terms.version || typeof stored.trainingConsent !== 'boolean') return null;
    return { trainingConsent: stored.trainingConsent };
  } catch {
    return null;
  }
}

export function saveConsent(choice: ConsentChoice): void {
  const stored: StoredChoice = {
    ...choice,
    version: config.terms.version,
    decidedAt: new Date().toISOString(),
  };
  try {
    window.localStorage.setItem(config.terms.storageKey, JSON.stringify(stored));
  } catch {
    // Storage blocked: the choice still applies for this visit.
  }
}
