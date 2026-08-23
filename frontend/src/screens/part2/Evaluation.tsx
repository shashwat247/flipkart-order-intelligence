import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { MetricTile } from "../../components/MetricTile";
import { DataTable } from "../../components/DataTable";
import { useReport } from "../../lib/reports";
import type { PerClassMetric } from "../../lib/types";

export default function Evaluation() {
  const report = useReport("part2_eval.json");
  if (report.status !== "ready") return <EmptyState file="part2_eval.json" />;
  const e = report.data;
  const weakest = [...e.per_class].sort((a, b) => a.f1 - b.f1)[0];

  const columns = [
    { key: "class", header: "class" },
    { key: "precision", header: "precision", align: "right" as const, mono: true, sortValue: (r: PerClassMetric) => r.precision, render: (r: PerClassMetric) => r.precision.toFixed(4) },
    { key: "recall", header: "recall", align: "right" as const, mono: true, sortValue: (r: PerClassMetric) => r.recall, render: (r: PerClassMetric) => r.recall.toFixed(4) },
    { key: "f1", header: "F1", align: "right" as const, mono: true, sortValue: (r: PerClassMetric) => r.f1, render: (r: PerClassMetric) => r.f1.toFixed(4) },
    { key: "support", header: "support", align: "right" as const, mono: true, sortValue: (r: PerClassMetric) => r.support },
  ];

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Evaluation" description="Final test-set accuracy and per-class precision/recall.">
        <ProvenanceStrip source="reports/part2_per_class_metrics.csv" script="python3 -m part2.evaluate_product_classifier" />
      </ScreenHeader>

      <MetricTile label="Test-set accuracy" value={`${(e.test_accuracy * 100).toFixed(2)}%`} reference={`80% reference line — ${e.test_accuracy >= e.reference_accuracy ? "cleared" : "missed"}`} hero />

      <DataTable columns={columns} rows={e.per_class} rowKey={(r) => r.class} highlightRow={(r) => r.class === weakest.class} />
    </div>
  );
}
