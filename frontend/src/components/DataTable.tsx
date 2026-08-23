import { useState } from "react";
import { ArrowDown, ArrowUp } from "lucide-react";

export interface Column<T> {
  key: string;
  header: string;
  align?: "left" | "right";
  mono?: boolean;
  render?: (row: T) => React.ReactNode;
  sortValue?: (row: T) => number | string;
}

export function DataTable<T>({
  columns,
  rows,
  rowKey,
  highlightRow,
}: {
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  /** predicate for a row to call out (e.g. the weakest subgroup) */
  highlightRow?: (row: T) => boolean;
}) {
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null);

  const sorted = (() => {
    if (!sort) return rows;
    const col = columns.find((c) => c.key === sort.key);
    if (!col?.sortValue) return rows;
    return [...rows].sort((a, b) => {
      const av = col.sortValue!(a);
      const bv = col.sortValue!(b);
      if (av < bv) return -1 * sort.dir;
      if (av > bv) return 1 * sort.dir;
      return 0;
    });
  })();

  return (
    <div className="w-full overflow-x-auto rounded-table border border-line">
      <table className="w-full border-collapse font-body text-sm">
        <thead className="sticky top-0 bg-ink-800">
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                className={`border-b border-line px-3 py-2 text-xs font-medium uppercase tracking-wide text-slate-400 ${
                  col.align === "right" ? "text-right" : "text-left"
                } ${col.sortValue ? "cursor-pointer select-none hover:text-paper" : ""}`}
                onClick={() =>
                  col.sortValue &&
                  setSort((prev) =>
                    prev?.key === col.key ? { key: col.key, dir: prev.dir === 1 ? -1 : 1 } : { key: col.key, dir: -1 }
                  )
                }
              >
                <span className="inline-flex items-center gap-1">
                  {col.header}
                  {sort?.key === col.key &&
                    (sort.dir === 1 ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => (
            <tr
              key={rowKey(row)}
              className={`h-9 ${i % 2 === 1 ? "bg-ink-800/40" : ""} ${
                highlightRow?.(row) ? "outline outline-1 outline-flag/50" : ""
              } border-b border-line/60 last:border-b-0`}
            >
              {columns.map((col) => (
                <td
                  key={col.key}
                  className={`px-3 ${col.align === "right" ? "text-right" : "text-left"} ${
                    col.mono ? "font-mono" : "font-body"
                  }`}
                >
                  {col.render ? col.render(row) : String((row as Record<string, unknown>)[col.key] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
