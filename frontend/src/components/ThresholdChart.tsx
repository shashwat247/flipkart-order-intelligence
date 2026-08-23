import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ThresholdPoint } from "../lib/types";

const LINE_COLORS = { precision: "#8A99B5", recall: "#2E7CF6", f1: "#FFB43D" };

/**
 * "Live recompute" here is a lookup into the already-exported full 0.02-step
 * sweep grid — never an in-browser refit of the model. Dragging the marker
 * snaps to the nearest pre-computed threshold.
 */
export function ThresholdChart({
  points,
  threshold,
  onThresholdChange,
  bestThreshold,
}: {
  points: ThresholdPoint[];
  threshold: number;
  onThresholdChange: (t: number) => void;
  bestThreshold: number;
}) {
  const nearest = points.reduce((best, p) =>
    Math.abs(p.threshold - threshold) < Math.abs(best.threshold - threshold) ? p : best
  );

  return (
    <div className="flex flex-col gap-3">
      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#26324A" strokeDasharray="2 4" />
            <XAxis
              dataKey="threshold"
              stroke="#8A99B5"
              tick={{ fill: "#8A99B5", fontFamily: "IBM Plex Mono", fontSize: 11 }}
              tickFormatter={(v: number) => v.toFixed(2)}
            />
            <YAxis
              stroke="#8A99B5"
              tick={{ fill: "#8A99B5", fontFamily: "IBM Plex Mono", fontSize: 11 }}
              domain={[0, 1]}
            />
            <Tooltip
              contentStyle={{ background: "#101827", border: "1px solid #26324A", borderRadius: 8 }}
              labelStyle={{ color: "#E9EEF7", fontFamily: "IBM Plex Mono" }}
              itemStyle={{ fontFamily: "IBM Plex Mono", fontSize: 12 }}
              labelFormatter={(v) => `threshold ${Number(v).toFixed(2)}`}
            />
            <Legend wrapperStyle={{ fontFamily: "Public Sans", fontSize: 12, color: "#8A99B5" }} />
            <Line type="monotone" dataKey="precision" stroke={LINE_COLORS.precision} dot={false} strokeWidth={1.5} />
            <Line type="monotone" dataKey="recall" stroke={LINE_COLORS.recall} dot={false} strokeWidth={1.5} />
            <Line type="monotone" dataKey="f1" stroke={LINE_COLORS.f1} dot={false} strokeWidth={2} />
            <ReferenceLine x={bestThreshold} stroke="#FFB43D" strokeDasharray="4 2" label={{ value: "F1-max", fill: "#FFB43D", fontSize: 11, position: "top" }} />
            <ReferenceLine x={threshold} stroke="#2E7CF6" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="flex items-center gap-3">
        <span className="font-mono text-xs text-slate-400 shrink-0">0.10</span>
        <input
          type="range"
          min={0.1}
          max={0.9}
          step={0.02}
          value={threshold}
          onChange={(e) => onThresholdChange(Number(e.target.value))}
          className="w-full accent-signal"
          aria-label="Decision threshold"
        />
        <span className="font-mono text-xs text-slate-400 shrink-0">0.90</span>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricChip label="accuracy" value={nearest.accuracy} />
        <MetricChip label="F1" value={nearest.f1} />
        <MetricChip label="precision" value={nearest.precision} />
        <MetricChip label="recall" value={nearest.recall} />
      </div>

      <div className="grid w-fit grid-cols-2 gap-px overflow-hidden rounded-control border border-line font-mono text-xs">
        <ConfusionCell label="TN" value={nearest.tn} />
        <ConfusionCell label="FP" value={nearest.fp} tone="flag" />
        <ConfusionCell label="FN" value={nearest.fn} tone="flag" />
        <ConfusionCell label="TP" value={nearest.tp} tone="verdant" />
      </div>
    </div>
  );
}

function MetricChip({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-control border border-line bg-ink-800 px-2 py-1.5">
      <div className="font-body text-[11px] text-slate-400">{label}</div>
      <div className="font-mono text-sm text-paper">{value.toFixed(4)}</div>
    </div>
  );
}

function ConfusionCell({ label, value, tone }: { label: string; value: number; tone?: "flag" | "verdant" }) {
  return (
    <div className={`flex flex-col items-center justify-center gap-0.5 bg-ink-800 px-4 py-2 ${
      tone === "flag" ? "text-flag" : tone === "verdant" ? "text-verdant" : "text-slate-400"
    }`}>
      <span className="text-[10px]">{label}</span>
      <span className="text-sm text-paper">{value}</span>
    </div>
  );
}
