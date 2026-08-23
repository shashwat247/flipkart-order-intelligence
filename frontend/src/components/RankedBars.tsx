export interface RankedBarItem {
  label: string;
  value: number;
  rank?: number;
  flag?: boolean; // e.g. "collapsed under permutation" or "below overall average"
}

/** Shared by Part 1's impurity/permutation importance lists and its subgroup
 * recall/precision deltas. Bars are horizontal so long feature names stay
 * legible; a `flag`ged row renders in --flag rather than the default --signal. */
export function RankedBars({
  items,
  diverging,
}: {
  items: RankedBarItem[];
  /** true for a delta-from-average chart (can go negative); false for a plain
   * ranked magnitude list */
  diverging?: boolean;
}) {
  const max = Math.max(...items.map((i) => Math.abs(i.value)), 1e-9);

  return (
    <div className="flex flex-col gap-2">
      {items.map((item) => {
        const pct = (Math.abs(item.value) / max) * 100;
        const negative = diverging && item.value < 0;
        return (
          <div key={item.label} className="grid grid-cols-[minmax(0,140px)_1fr_80px] items-center gap-2">
            <span className="truncate font-body text-xs text-slate-400" title={item.label}>
              {item.rank !== undefined && <span className="text-slate-400/70">#{item.rank} </span>}
              {item.label}
            </span>
            <div className="relative h-4 w-full rounded-sm bg-ink-900">
              <div
                className={`h-4 rounded-sm ${item.flag ? "bg-flag" : negative ? "bg-slate-400" : "bg-signal"}`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="text-right font-mono text-xs text-paper">
              {item.value >= 0 ? "+" : ""}
              {item.value.toFixed(4)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
