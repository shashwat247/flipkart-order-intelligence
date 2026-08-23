import { CheckCircle2, XCircle } from "lucide-react";
import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { CodeBlock } from "../../components/CodeBlock";
import { useReport } from "../../lib/reports";

export default function ArtifactSamples() {
  const report = useReport("part2_artifact.json");
  if (report.status !== "ready") return <EmptyState file="part2_artifact.json" />;
  const a = report.data;

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Artifact & Samples" description="The saved classifier and the real exported test-split samples.">
        <ProvenanceStrip source={a.path} script="python3 -m part2.export_samples" />
      </ScreenHeader>

      <div className={`flex w-fit items-center gap-2 rounded-control border px-3 py-1.5 font-body text-sm ${
        a.loads_ok ? "border-verdant text-verdant" : "border-flag text-flag"
      }`}>
        {a.loads_ok ? <CheckCircle2 size={16} /> : <XCircle size={16} />}
        {a.architecture} ({a.head}) — load {a.loads_ok ? "succeeded" : "failed"}
      </div>

      <CodeBlock code={a.load_snippet} language="python" />

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        {a.sample_images.map((s) => (
          <div key={s.file} className="flex flex-col gap-2 rounded-panel border border-line bg-ink-800 p-2">
            <img src={`/samples/${s.file}`} alt={s.true_label} className="aspect-square w-full rounded-control border border-line bg-ink-900 object-contain" />
            <div className="font-mono text-[10px] text-slate-400">{s.file}</div>
            <div className="font-body text-xs text-slate-400">true: {s.true_label}</div>
            <div className={`font-body text-xs ${s.agrees_with_true_label ? "text-verdant" : "text-flag"}`}>
              pred: {s.predicted_class} ({(s.confidence * 100).toFixed(2)}%)
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
