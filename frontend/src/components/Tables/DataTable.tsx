import type { ReactNode } from "react";

interface Props {
  headers: string[];
  rows: ReactNode[][];
}

export function DataTable({ headers, rows }: Props) {
  return (
    <div className="overflow-x-auto rounded-lg border border-obs-border">
      <table className="min-w-full divide-y divide-obs-border text-sm">
        <thead className="bg-obs-card2 text-xs uppercase text-obs-muted">
          <tr>
            {headers.map((header) => (
              <th key={header} className="whitespace-nowrap px-3 py-3 text-left font-medium">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-obs-border bg-obs-card">
          {rows.map((row, idx) => (
            <tr key={idx}>
              {row.map((cell, cellIdx) => (
                <td key={cellIdx} className="whitespace-nowrap px-3 py-3 text-obs-text">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

