import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { MetricTile } from "../../components/MetricTile";
import { useReport } from "../../lib/reports";

export default function Baseline() {
  const report = useReport("part1_baseline.json");
  if (report.status !== "ready") return <EmptyState file="part1_baseline.json" />;
  const b = report.data;

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Baseline" description="Task 4 — DummyClassifier(strategy='most_frequent').">
        <ProvenanceStrip source="reports/part1_model_report.md" script="python3 -m part1.train_return_risk" />
      </ScreenHeader>

      <div className="flex gap-4">
        <MetricTile label="Accuracy" value={`${(b.accuracy * 100).toFixed(2)}%`} hero />
        <MetricTile label="F1 (returned=1)" value={b.f1_positive.toFixed(4)} />
      </div>

      <div className="max-w-xl rounded-panel border border-flag bg-flag/10 p-4">
        <p className="font-body text-sm text-paper">
          <span className="font-semibold text-flag">The trap:</span> high accuracy, zero recall —
          this classifier predicts "not returned" for every order and is never once right about a return.
        </p>
        <p className="mt-2 font-body text-sm text-slate-400">{b.note}</p>
      </div>
    </div>
  );
}
