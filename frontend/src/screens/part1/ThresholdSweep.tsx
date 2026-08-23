import { useState } from "react";
import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { ThresholdChart } from "../../components/ThresholdChart";
import { useReport } from "../../lib/reports";

export default function ThresholdSweep() {
  const report = useReport("part1_threshold_sweep.json");
  const [threshold, setThreshold] = useState<number | null>(null);
  if (report.status !== "ready") return <EmptyState file="part1_threshold_sweep.json" />;
  const s = report.data;
  const t = threshold ?? s.best_threshold;

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader
        title="Threshold Sweep"
        description="Task 5 — F1 vs. threshold for the tuned Random Forest's own predict_proba, 0.10 to 0.90."
      >
        <ProvenanceStrip source="reports/part1_threshold_sweep_rf.csv" script="python3 -m part1.train_return_risk" />
      </ScreenHeader>

      <ThresholdChart points={s.points} threshold={t} onThresholdChange={setThreshold} bestThreshold={s.best_threshold} />

      <div className="max-w-2xl rounded-panel border border-line bg-ink-800 p-4">
        <h2 className="mb-2 font-body text-sm font-semibold text-paper">The business trade-off</h2>
        <p className="font-body text-sm text-slate-400">{s.tradeoff_paragraph}</p>
      </div>
    </div>
  );
}
