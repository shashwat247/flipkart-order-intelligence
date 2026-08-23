import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { useReport } from "../../lib/reports";

export default function ConfusionPatterns() {
  const report = useReport("part2_eval.json");
  if (report.status !== "ready") return <EmptyState file="part2_eval.json" />;
  const e = report.data;

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Confusion Patterns" description="The most-confused category pairs, read off the matrix — not guessed.">
        <ProvenanceStrip source="reports/part2_evaluation.md" script="python3 -m part2.evaluate_product_classifier" />
      </ScreenHeader>

      <div className="flex flex-col gap-4">
        {e.pair_explanations.map((p) => (
          <div key={`${p.class_a}-${p.class_b}`} className="rounded-panel border border-line bg-ink-800 p-4">
            <div className="mb-2 flex items-center gap-2">
              <span className="font-body text-sm font-semibold text-paper">
                {p.class_a} ↔ {p.class_b}
              </span>
              <span className="rounded-control border border-flag/50 bg-flag/10 px-2 py-0.5 font-mono text-xs text-flag">
                {p.total_misclassifications} misclassifications
              </span>
            </div>
            <p className="mb-2 font-mono text-xs text-slate-400">{p.read_off}</p>
            <p className="font-body text-sm text-slate-400">{p.explanation}</p>
          </div>
        ))}
      </div>

      <p className="max-w-2xl font-body text-sm text-slate-400">
        The errors cluster inside two visually coherent families — upper-body garments and
        footwear — and almost never cross between them, which is the useful failure mode for
        a catalogue-department classifier: a mis-tagged shirt still lands in apparel.
      </p>
    </div>
  );
}
