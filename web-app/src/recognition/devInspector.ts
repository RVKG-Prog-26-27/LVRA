/**
 * Development only. Keeps the last request (line PNGs and labels.csv) and the
 * CSV the backend returned, so the developer panel (?dev=1) can show them.
 * Nothing is recorded unless the panel is enabled, so normal users' line
 * images are not kept in memory after recognition.
 */

export interface InspectedLine {
  filename: string;
  url: string;
}

export interface Inspection {
  backend: string;
  lines: InspectedLine[];
  sentCsv: string;
  receivedCsv: string | null;
  trainingConsent: boolean;
}

let enabled = false;
let current: Inspection | null = null;
const listeners = new Set<() => void>();

function emit() {
  for (const listener of listeners) listener();
}

export const devInspector = {
  enable() {
    enabled = true;
  },
  get(): Inspection | null {
    return current;
  },
  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  recordRequest(backend: string, lines: { filename: string; blob: Blob }[], sentCsv: string, trainingConsent: boolean) {
    if (!enabled) return;
    current?.lines.forEach((l) => URL.revokeObjectURL(l.url));
    current = {
      backend,
      lines: lines.map((l) => ({ filename: l.filename, url: URL.createObjectURL(l.blob) })),
      sentCsv,
      receivedCsv: null,
      trainingConsent,
    };
    emit();
  },
  recordResponse(receivedCsv: string) {
    if (!enabled || !current) return;
    current = { ...current, receivedCsv };
    emit();
  },
};
