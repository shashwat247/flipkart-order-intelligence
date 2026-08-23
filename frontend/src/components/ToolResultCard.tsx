import { Boxes, Cpu } from "lucide-react";
import { RiskBadge, type RiskBucket } from "./RiskBadge";
import type { ToolResult } from "../lib/agent";

/** Renders whatever the real tool returned.
 *
 * Both cards read their numbers straight off the tool payload — the return
 * probability comes from Part 1's saved Random Forest and the class confidence
 * from Part 2's saved ResNet-18. Nothing here is computed in the browser.
 */
export function ToolResultCard({ tool }: { tool: ToolResult }) {
  if (tool.status !== "ok") return null;

  if (tool.return_probability !== undefined && tool.risk_bucket) {
    return (
      <section
        aria-label="Return-risk prediction"
        className="mt-3 rounded-panel border border-line bg-ink-900/60 p-4"
      >
        <header className="mb-3 flex items-center gap-2">
          <Cpu size={14} className="text-signal" aria-hidden />
          <h3 className="font-body text-xs font-semibold uppercase tracking-wide text-slate-400">
            Return-risk model
          </h3>
        </header>

        <RiskBadge
          bucket={tool.risk_bucket as RiskBucket}
          thresholdRf={tool.threshold_rf ?? 0.5}
          probability={tool.return_probability}
        />

        <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-ink-700" aria-hidden>
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              tool.risk_bucket === "High"
                ? "bg-flag"
                : tool.risk_bucket === "Medium"
                  ? "bg-tape"
                  : "bg-verdant"
            }`}
            style={{ width: `${(tool.return_probability * 100).toFixed(1)}%` }}
          />
        </div>

        {tool.bucket_cut_points && (
          <dl className="mt-3 grid grid-cols-1 gap-1 font-mono text-xs text-slate-400 sm:grid-cols-3">
            {Object.entries(tool.bucket_cut_points).map(([bucket, rule]) => (
              <div key={bucket} className="flex gap-1.5">
                <dt className="text-paper">{bucket}:</dt>
                <dd>{rule}</dd>
              </div>
            ))}
          </dl>
        )}
        <p className="mt-2 font-mono text-xs text-slate-400">model: {tool.model}</p>
      </section>
    );
  }

  if (tool.predicted_class) {
    return (
      <section
        aria-label="Product category prediction"
        className="mt-3 rounded-panel border border-line bg-ink-900/60 p-4"
      >
        <header className="mb-3 flex items-center gap-2">
          <Boxes size={14} className="text-signal" aria-hidden />
          <h3 className="font-body text-xs font-semibold uppercase tracking-wide text-slate-400">
            Image classifier
          </h3>
        </header>

        <p className="font-display text-xl font-bold text-paper">{tool.predicted_class}</p>
        <p className="font-mono text-sm text-verdant">
          {(tool.confidence_percent ?? (tool.confidence ?? 0) * 100).toFixed(2)}% confidence
        </p>

        {tool.top3 && (
          <ul className="mt-3 flex flex-col gap-1.5">
            {tool.top3.map((t) => (
              <li key={t.label} className="flex items-center gap-2">
                <span className="w-24 shrink-0 font-mono text-xs text-slate-400">{t.label}</span>
                <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-ink-700">
                  <span
                    className="block h-full rounded-full bg-signal"
                    style={{ width: `${(t.probability * 100).toFixed(1)}%` }}
                  />
                </span>
                <span className="w-16 shrink-0 text-right font-mono text-xs text-paper">
                  {(t.probability * 100).toFixed(2)}%
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
    );
  }

  return null;
}
