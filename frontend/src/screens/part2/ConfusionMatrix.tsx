import { useState } from "react";
import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { Heatmap } from "../../components/Heatmap";
import { useReport } from "../../lib/reports";

export default function ConfusionMatrix() {
  const report = useReport("part2_eval.json");
  const [selected, setSelected] = useState<[number, number] | null>(null);
  if (report.status !== "ready") return <EmptyState file="part2_eval.json" />;
  const e = report.data;
  const short = (n: string) => (n.length > 6 ? n.slice(0, 5) + "…" : n);

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Confusion Matrix" description="10×10, real predicted counts on the held-out test split.">
        <ProvenanceStrip source="reports/part2_confusion_matrix.csv" script="python3 -m part2.evaluate_product_classifier" />
      </ScreenHeader>

      <Heatmap
        rowLabels={e.class_names.map(short)}
        colLabels={e.class_names.map(short)}
        values={e.confusion_matrix}
        diagonalIsIdentity
        formatValue={(v) => String(v)}
        onCellClick={(r, c) => setSelected([r, c])}
      />

      {selected && (
        <div className="max-w-md rounded-panel border border-line bg-ink-800 p-4">
          <p className="font-body text-sm text-paper">
            True: <span className="font-semibold">{e.class_names[selected[0]]}</span> · Predicted:{" "}
            <span className="font-semibold">{e.class_names[selected[1]]}</span>
          </p>
          <p className="font-mono text-sm text-paper">count = {e.confusion_matrix[selected[0]][selected[1]]}</p>
          {selected[0] === selected[1] ? (
            <p className="mt-1 font-body text-xs text-verdant">Correct prediction.</p>
          ) : (
            <p className="mt-1 font-body text-xs text-flag">Misclassification — see Confusion Patterns for the visual-similarity explanation.</p>
          )}
        </div>
      )}
    </div>
  );
}
