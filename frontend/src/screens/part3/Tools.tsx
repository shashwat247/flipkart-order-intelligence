import { useState } from "react";
import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { CodeBlock } from "../../components/CodeBlock";
import { RiskBadge } from "../../components/RiskBadge";
import { useReport } from "../../lib/reports";

export default function Tools() {
  const tools = useReport("part3_tools.json");
  const p2 = useReport("part2_artifact.json");
  const [orderIndex, setOrderIndex] = useState(0);
  const [imageIndex, setImageIndex] = useState(0);

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Tools" description="The agent's two real tools — both load the actual trained artifacts.">
        <ProvenanceStrip source="part3/tools.py" />
      </ScreenHeader>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="flex flex-col gap-3 rounded-panel border border-line bg-ink-800 p-4">
          <h2 className="font-body text-sm font-semibold text-paper">check_return_risk</h2>
          {tools.status !== "ready" ? (
            <EmptyState file="part3_tools.json" />
          ) : (
            <>
              <CodeBlock code={tools.data.function_signature} language="python" />
              <p className="font-mono text-xs text-slate-400">artifact: {tools.data.artifact_path}</p>
              <select
                value={orderIndex}
                onChange={(e) => setOrderIndex(Number(e.target.value))}
                className="rounded-control border border-line bg-ink-900 px-2 py-1.5 font-mono text-xs text-paper"
              >
                {tools.data.return_risk_examples.map((ex, i) => (
                  <option key={ex.order_id} value={i}>
                    Order {ex.order_id} — a real held-out test-split order
                  </option>
                ))}
              </select>
              {(() => {
                const ex = tools.data.return_risk_examples[orderIndex];
                const bucket = ex.return_probability < tools.data.threshold_rf
                  ? "Low"
                  : ex.return_probability < tools.data.threshold_rf + 0.15
                    ? "Medium"
                    : "High";
                return (
                  <>
                    <div className="grid grid-cols-2 gap-1 font-mono text-[11px] text-slate-400">
                      {Object.entries(ex.features).map(([k, v]) => (
                        <div key={k}>
                          {k}: <span className="text-paper">{String(v)}</span>
                        </div>
                      ))}
                    </div>
                    <RiskBadge bucket={bucket} thresholdRf={tools.data.threshold_rf} probability={ex.return_probability} />
                    <p className="font-mono text-[11px] text-slate-400">actually returned: {String(ex.actual_returned)}</p>
                  </>
                );
              })()}
            </>
          )}
        </div>

        <div className="flex flex-col gap-3 rounded-panel border border-line bg-ink-800 p-4">
          <h2 className="font-body text-sm font-semibold text-paper">classify_product_image</h2>
          {p2.status !== "ready" ? (
            <EmptyState file="part2_artifact.json" />
          ) : (
            <>
              <CodeBlock code={p2.data.load_snippet.split(";").slice(0, 2).join(";") + "\nclassify_product_image(path)"} language="python" />
              <p className="font-mono text-xs text-slate-400">artifact: {p2.data.path}</p>
              <select
                value={imageIndex}
                onChange={(e) => setImageIndex(Number(e.target.value))}
                className="rounded-control border border-line bg-ink-900 px-2 py-1.5 font-mono text-xs text-paper"
              >
                {p2.data.sample_images.map((s, i) => (
                  <option key={s.file} value={i}>
                    {s.file}
                  </option>
                ))}
              </select>
              {(() => {
                const s = p2.data.sample_images[imageIndex];
                return (
                  <div className="flex items-center gap-3">
                    <img src={`/samples/${s.file}`} alt={s.true_label} className="h-20 w-20 rounded-control border border-line bg-ink-900 object-contain" />
                    <div>
                      <p className="font-body text-sm text-paper">predicted: {s.predicted_class}</p>
                      <p className="font-mono text-xs text-slate-400">confidence {(s.confidence * 100).toFixed(2)}%</p>
                      <p className={`font-mono text-xs ${s.agrees_with_true_label ? "text-verdant" : "text-flag"}`}>
                        true label: {s.true_label}
                      </p>
                    </div>
                  </div>
                );
              })()}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
