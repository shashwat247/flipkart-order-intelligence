import { useState } from "react";

/** Shared by the GridSearchCV heatmap (Part 1) and the confusion matrix
 * (Part 2). Colour intensity is relative to this matrix's own min/max —
 * never a fixed absolute scale, since a 6-cell CV grid and a 10x10 confusion
 * matrix have completely different ranges. */
export function Heatmap({
  rowLabels,
  colLabels,
  values,
  bestCell,
  onCellClick,
  formatValue = (v) => v.toFixed(4),
  diagonalIsIdentity = false,
}: {
  rowLabels: string[];
  colLabels: string[];
  values: number[][];
  bestCell?: [number, number];
  onCellClick?: (row: number, col: number, value: number) => void;
  formatValue?: (v: number) => string;
  /** true for a confusion matrix — the diagonal (correct predictions) is
   * visually distinguished from the off-diagonal (errors) rather than sharing
   * one colour scale, since a correct prediction and an error are not the
   * same kind of thing even at equal magnitude. */
  diagonalIsIdentity?: boolean;
}) {
  const [hover, setHover] = useState<[number, number] | null>(null);
  const flat = values.flat();
  const offDiagonal = diagonalIsIdentity
    ? values.flatMap((row, r) => row.filter((_, c) => c !== r))
    : flat;
  const min = Math.min(...offDiagonal, 0);
  const max = Math.max(...offDiagonal, 1);

  const intensity = (r: number, c: number, v: number) => {
    if (diagonalIsIdentity && r === c) return 1;
    return max === min ? 0 : (v - min) / (max - min);
  };

  const cellColor = (r: number, c: number, v: number) => {
    const t = intensity(r, c, v);
    if (diagonalIsIdentity && r === c) {
      // correct predictions: verdant scale
      return `rgba(49, 207, 163, ${0.15 + 0.65 * Math.min(1, v / max)})`;
    }
    // errors / generic values: signal-to-tape scale
    return `rgba(255, 180, 61, ${0.06 + 0.85 * t})`;
  };

  return (
    <div className="w-full overflow-x-auto">
      <table className="border-collapse font-mono text-xs">
        <thead>
          <tr>
            <th className="p-1" />
            {colLabels.map((c, ci) => (
              <th
                key={c}
                className={`whitespace-nowrap p-1 text-slate-400 ${hover?.[1] === ci ? "text-paper" : ""}`}
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rowLabels.map((rLabel, ri) => (
            <tr key={rLabel}>
              <th className={`whitespace-nowrap p-1 text-right text-slate-400 ${hover?.[0] === ri ? "text-paper" : ""}`}>
                {rLabel}
              </th>
              {values[ri].map((v, ci) => {
                const isBest = bestCell && bestCell[0] === ri && bestCell[1] === ci;
                return (
                  <td
                    key={ci}
                    onMouseEnter={() => setHover([ri, ci])}
                    onMouseLeave={() => setHover(null)}
                    onClick={() => onCellClick?.(ri, ci, v)}
                    className={`h-8 w-14 cursor-default border border-ink-900 text-center align-middle ${
                      onCellClick ? "cursor-pointer" : ""
                    } ${isBest ? "outline outline-2 outline-signal" : ""}`}
                    style={{ backgroundColor: cellColor(ri, ci, v) }}
                    title={`${rLabel} × ${colLabels[ci]}: ${formatValue(v)}`}
                  >
                    <span className={v === 0 ? "text-slate-400/40" : "text-paper"}>
                      {v === 0 ? "·" : formatValue(v)}
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {hover && (
        <p className="mt-2 font-body text-xs text-slate-400">
          <span className="text-paper">{rowLabels[hover[0]]}</span> ×{" "}
          <span className="text-paper">{colLabels[hover[1]]}</span>:{" "}
          <span className="font-mono text-paper">{formatValue(values[hover[0]][hover[1]])}</span>
        </p>
      )}
    </div>
  );
}
