/**
 * Development only. Keeps the requests of the last "Atpazīt tekstu" press
 * (one per photo: line PNGs and labels.csv) and the CSV each one returned,
 * so the developer panel (?dev=1) can show them. Nothing is recorded unless
 * the panel is enabled, so normal users' line images are not kept in memory
 * after recognition.
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
let batch: Inspection[] = [];
const listeners = new Set<() => void>();

function emit() {
  for (const listener of listeners) listener();
}

export const devInspector = {
  enable() {
    enabled = true;
  },
  get(): Inspection[] {
    return batch;
  },
  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  /** Called once per "Atpazīt tekstu" press, before the first photo is sent. */
  startBatch() {
    if (!enabled) return;
    batch.forEach((inspection) => inspection.lines.forEach((l) => URL.revokeObjectURL(l.url)));
    batch = [];
    emit();
  },
  recordRequest(backend: string, lines: { filename: string; blob: Blob }[], sentCsv: string, trainingConsent: boolean) {
    if (!enabled) return;
    batch = [
      ...batch,
      {
        backend,
        lines: lines.map((l) => ({ filename: l.filename, url: URL.createObjectURL(l.blob) })),
        sentCsv,
        receivedCsv: null,
        trainingConsent,
      },
    ];
    emit();
  },
  recordResponse(receivedCsv: string) {
    if (!enabled || batch.length === 0) return;
    batch = [...batch.slice(0, -1), { ...batch[batch.length - 1], receivedCsv }];
    emit();
  },
};
