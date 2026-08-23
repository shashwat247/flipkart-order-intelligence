import { ArrowRight } from "lucide-react";
import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { useReport } from "../../lib/reports";

function Step({ label }: { label: string }) {
  return (
    <div className="rounded-panel border border-line bg-ink-800 px-3 py-2 font-mono text-xs text-paper">{label}</div>
  );
}

export default function Preprocessing() {
  const report = useReport("part2_training.json");
  if (report.status !== "ready") return <EmptyState file="part2_training.json" />;
  const t = report.data;

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Preprocessing" description="The transform chain a pretrained ImageNet backbone requires.">
        <ProvenanceStrip source="part2/config.py, models/product_classifier_metadata.json" />
      </ScreenHeader>

      <div className="flex flex-wrap items-center gap-2">
        <Step label="1 channel (grayscale)" />
        <ArrowRight size={16} className="text-slate-400" />
        <Step label={t.channel_handling} />
        <ArrowRight size={16} className="text-slate-400" />
        <Step label={`resize to ${t.input_size[0]}×${t.input_size[1]}`} />
        <ArrowRight size={16} className="text-slate-400" />
        <Step label="ImageNet mean/std normalize" />
      </div>

      <div className="flex gap-6 font-mono text-sm">
        <span className="text-paper">mean = [{t.normalization.mean.join(", ")}]</span>
        <span className="text-paper">std = [{t.normalization.std.join(", ")}]</span>
      </div>

      <p className="max-w-2xl font-body text-sm text-slate-400">
        Fashion-MNIST ships 28×28 single-channel images. ResNet-18 was pretrained on
        3-channel {t.input_size[0]}×{t.input_size[1]} ImageNet crops, so the grey channel is
        replicated 3× and resized up to the backbone's native resolution, then normalised
        with the exact statistics the pretrained weights expect.
      </p>
    </div>
  );
}
