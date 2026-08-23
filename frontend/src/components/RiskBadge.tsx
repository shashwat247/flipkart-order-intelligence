import { AlertTriangle, CheckCircle2, TriangleAlert } from "lucide-react";

export type RiskBucket = "Low" | "Medium" | "High";

const CONFIG: Record<RiskBucket, { color: string; bg: string; Icon: typeof CheckCircle2 }> = {
  Low: { color: "text-verdant", bg: "bg-verdant/10", Icon: CheckCircle2 },
  Medium: { color: "text-tape", bg: "bg-tape/10", Icon: TriangleAlert },
  High: { color: "text-flag", bg: "bg-flag/10", Icon: AlertTriangle },
};

/** Colour is never the sole carrier of meaning — every bucket pairs a glyph
 * and a label, and the numeric bound that produced it is always shown
 * alongside, anchored to t*_rf (never a hardcoded 0.3/0.6 split). */
export function RiskBadge({
  bucket,
  thresholdRf,
  probability,
}: {
  bucket: RiskBucket;
  thresholdRf: number;
  probability?: number;
}) {
  const { color, bg, Icon } = CONFIG[bucket];
  const mediumHigh = thresholdRf + 0.15;
  const bound =
    bucket === "Low"
      ? `< ${thresholdRf.toFixed(2)}`
      : bucket === "Medium"
        ? `${thresholdRf.toFixed(2)}–${mediumHigh.toFixed(2)}`
        : `≥ ${mediumHigh.toFixed(2)}`;

  return (
    <div className={`inline-flex items-center gap-2 rounded-control border border-line px-3 py-1.5 ${bg}`}>
      <Icon size={16} className={color} aria-hidden />
      <span className={`font-body text-sm font-semibold ${color}`}>{bucket.toUpperCase()} RISK</span>
      {probability !== undefined && (
        <span className="font-mono text-sm text-paper">{(probability * 100).toFixed(2)}%</span>
      )}
      <span className="font-mono text-xs text-slate-400">(t*_rf {bound})</span>
    </div>
  );
}
