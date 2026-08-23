import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { CodeBlock } from "../../components/CodeBlock";
import { useReport } from "../../lib/reports";

export default function DatasetGenerator() {
  const report = useReport("part1_data.json");
  if (report.status !== "ready") return <EmptyState file="part1_data.json" />;
  const { generator } = report.data;

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Dataset Generator" description="Task 1 — the seeded synthetic order dataset.">
        <ProvenanceStrip source="generate_orders.py" script="python3 generate_orders.py" />
      </ScreenHeader>

      <div className="flex gap-3">
        <div className="rounded-control border border-line bg-ink-800 px-3 py-2 font-mono text-sm text-paper">
          seed = {generator.seed}
        </div>
        <div className="rounded-control border border-line bg-ink-800 px-3 py-2 font-mono text-sm text-paper">
          N = {generator.n_rows}
        </div>
      </div>

      <p className="max-w-2xl font-body text-sm text-slate-400">
        The seed and N are locked — changing either (or the order in which the
        RNG is consumed) changes every downstream number in this project.
      </p>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="rounded-panel border border-line bg-ink-800 p-4">
          <h2 className="mb-3 font-body text-sm font-semibold text-paper">product_category probabilities</h2>
          <div className="flex flex-col gap-1.5">
            {generator.category_probs.map((c) => (
              <div key={c.category} className="flex items-center gap-2">
                <span className="w-28 shrink-0 font-body text-xs text-slate-400">{c.category}</span>
                <div className="h-2 flex-1 rounded-full bg-ink-900">
                  <div className="h-2 rounded-full bg-signal" style={{ width: `${c.probability * 100}%` }} />
                </div>
                <span className="w-12 shrink-0 text-right font-mono text-xs text-paper">{c.probability.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="rounded-panel border border-line bg-ink-800 p-4">
          <h2 className="mb-3 font-body text-sm font-semibold text-paper">payment_method probabilities</h2>
          <div className="flex flex-col gap-1.5">
            {generator.payment_probs.map((c) => (
              <div key={c.payment_method} className="flex items-center gap-2">
                <span className="w-28 shrink-0 font-body text-xs text-slate-400">{c.payment_method}</span>
                <div className="h-2 flex-1 rounded-full bg-ink-900">
                  <div className="h-2 rounded-full bg-tape" style={{ width: `${c.probability * 100}%` }} />
                </div>
                <span className="w-12 shrink-0 text-right font-mono text-xs text-paper">{c.probability.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-2">
        <h2 className="font-body text-sm font-semibold text-paper">Generating script (read-only)</h2>
        <CodeBlock code="python3 generate_orders.py" />
        <p className="font-body text-xs text-slate-400">
          Regenerating with the same seed reproduces `orders_dataset.csv` byte-for-byte.
        </p>
      </div>
    </div>
  );
}
