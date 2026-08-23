import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { MetricTile } from "../../components/MetricTile";
import { useReport } from "../../lib/reports";

export default function Finetuning() {
  const report = useReport("part2_training.json");
  if (report.status !== "ready") return <EmptyState file="part2_training.json" />;
  const t = report.data;

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Fine-tuning" description="Conditional — only triggered if feature-extraction validation accuracy misses the bar.">
        <ProvenanceStrip source="reports/part2_training_log.json" script="python3 -m part2.train_product_classifier" />
      </ScreenHeader>

      <div className="flex gap-4">
        <MetricTile label="Before fine-tuning" value={`${(t.val_accuracy_before_finetuning * 100).toFixed(2)}%`} hero />
        <MetricTile label="After fine-tuning" value={`${(t.val_accuracy_after_finetuning * 100).toFixed(2)}%`} />
        <MetricTile label="Trigger threshold" value={`${(t.finetune_trigger_threshold * 100).toFixed(0)}%`} />
      </div>

      {t.finetune_triggered ? (
        <p className="font-body text-sm text-paper">Fine-tuning was triggered — layer4 was unfrozen at a reduced learning rate.</p>
      ) : (
        <div className="max-w-2xl rounded-panel border border-verdant bg-verdant/10 p-4">
          <p className="font-body text-sm text-paper">
            <span className="font-semibold text-verdant">Feature extraction alone was sufficient.</span> Validation
            accuracy reached {(t.val_accuracy_before_finetuning * 100).toFixed(2)}%, above the{" "}
            {(t.finetune_trigger_threshold * 100).toFixed(0)}% bar, so the conditional fine-tuning stage was{" "}
            <strong>not</strong> triggered and layer4 was never unfrozen. This is stated explicitly rather than left
            ambiguous — the before/after numbers are identical because no second stage ran.
          </p>
        </div>
      )}
    </div>
  );
}
