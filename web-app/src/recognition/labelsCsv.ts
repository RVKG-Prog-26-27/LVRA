/**
 * labels.csv in the format the training code reads (latvian_htr/dataset.py):
 *
 *   filename,text
 *   0001.png,Labdien! Kā jums iet?
 *   0002.png,"Rīgā, šodien līst."
 *
 * UTF-8, first row is the header. A value with a comma, quote or line break
 * is wrapped in double quotes and inner quotes are doubled (standard CSV, the
 * same as Python's csv module writes and reads).
 */

export interface LabelRow {
  filename: string;
  text: string;
}

export const LABELS_HEADER = ['filename', 'text'] as const;

function escapeCell(value: string): string {
  return /[",\r\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

export function buildLabelsCsv(rows: LabelRow[]): string {
  const lines = [LABELS_HEADER.join(',')];
  for (const row of rows) lines.push(`${escapeCell(row.filename)},${escapeCell(row.text)}`);
  return `${lines.join('\r\n')}\r\n`;
}

/** Parses CSV text into rows of cells. Handles quotes, doubled quotes and line breaks inside quotes. */
function parseCsv(input: string): string[][] {
  const text = input.replace(/^﻿/, '');
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = '';
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          cell += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        cell += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ',') {
      row.push(cell);
      cell = '';
    } else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      row.push(cell);
      rows.push(row);
      row = [];
      cell = '';
    } else {
      cell += ch;
    }
  }
  if (cell !== '' || row.length > 0) {
    row.push(cell);
    rows.push(row);
  }
  return rows.filter((r) => !(r.length === 1 && r[0] === ''));
}

/** Reads labels.csv. Throws if the header is not "filename,text". */
export function parseLabelsCsv(input: string): LabelRow[] {
  const [header, ...body] = parseCsv(input);
  const filenameCol = header?.indexOf('filename') ?? -1;
  const textCol = header?.indexOf('text') ?? -1;
  if (filenameCol === -1 || textCol === -1) throw new Error('labels.csv must have the columns filename,text');
  return body.map((cells) => ({ filename: cells[filenameCol] ?? '', text: cells[textCol] ?? '' }));
}
