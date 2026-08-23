export function MetricTile({
  label,
  value,
  unit,
  delta,
  deltaGood,
  reference,
  hero,
}: {
  label: string;
  value: string;
  unit?: string;
  delta?: string;
  deltaGood?: boolean;
  reference?: string;
  /** the ONE hero number per screen — larger, display face for the label only */
  hero?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1 rounded-panel border border-line bg-ink-800 p-4">
      <span className={hero ? "font-display text-sm text-slate-400" : "font-body text-xs text-slate-400"}>
        {label}
      </span>
      <span
        className={
          hero
            ? "font-mono text-[40px] leading-none text-paper"
            : "font-mono text-2xl leading-none text-paper"
        }
      >
        {value}
        {unit && <span className="ml-1 text-base text-slate-400">{unit}</span>}
      </span>
      {delta && (
        <span className={`font-mono text-xs ${deltaGood ? "text-verdant" : "text-flag"}`}>{delta}</span>
      )}
      {reference && <span className="font-mono text-xs text-slate-400">{reference}</span>}
    </div>
  );
}
