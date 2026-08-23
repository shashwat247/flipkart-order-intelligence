import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { MetricTile } from "../../components/MetricTile";
import { useReport } from "../../lib/reports";

export default function Dataset() {
  const report = useReport("part2_dataset.json");
  if (report.status !== "ready") return <EmptyState file="part2_dataset.json" />;
  const d = report.data;

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Dataset" description="Fashion-MNIST (Zalando Research), via torchvision.">
        <ProvenanceStrip source={d.source} script="python3 -m part2.cache_features" />
      </ScreenHeader>

      <div className="grid grid-cols-3 gap-3">
        <MetricTile label="Head-train split" value={String(d.split_sizes.train)} hero />
        <MetricTile label="Validation split" value={String(d.split_sizes.val)} />
        <MetricTile label="Test split (untouched)" value={String(d.split_sizes.test)} />
      </div>

      <p className="max-w-2xl font-body text-sm text-slate-400">{d.test_untouched_note}</p>

      <div className="flex flex-wrap gap-2">
        {d.classes.map((c) => (
          <div key={c.class_name} className="flex flex-col gap-1 rounded-panel border border-line bg-ink-800 p-3">
            <span className="font-body text-sm font-semibold text-paper">{c.class_name}</span>
            <span className="font-mono text-xs text-slate-400">train {c.head_train_count} · val {c.val_count} · test {c.test_count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
