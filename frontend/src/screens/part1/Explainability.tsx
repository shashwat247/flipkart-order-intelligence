import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { RankedBars } from "../../components/RankedBars";
import { useReport } from "../../lib/reports";

export default function Explainability() {
  const report = useReport("part1_importance.json");
  if (report.status !== "ready") return <EmptyState file="part1_importance.json" />;
  const d = report.data;

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Explainability" description="Task 7 — impurity-based importance vs. permutation importance, on the same saved model.">
        <ProvenanceStrip source="reports/part1_importance_comparison.csv" script="python3 -m part1.feature_analysis" />
      </ScreenHeader>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div>
          <h2 className="mb-3 font-body text-sm font-semibold text-paper">Impurity-based (.feature_importances_)</h2>
          <RankedBars
            items={d.impurity.map((r) => ({ label: r.feature, value: r.value, rank: r.rank, flag: r.feature === d.biggest_drop }))}
          />
        </div>
        <div>
          <h2 className="mb-3 font-body text-sm font-semibold text-paper">Permutation (test-set ROC-AUC drop)</h2>
          <RankedBars
            diverging
            items={d.permutation.map((r) => ({ label: r.feature, value: r.value, rank: r.rank, flag: r.feature === d.biggest_drop }))}
          />
        </div>
      </div>

      <div className="max-w-2xl rounded-panel border border-flag bg-flag/10 p-4">
        <h2 className="mb-1 font-body text-sm font-semibold text-flag">Biggest demotion: {d.biggest_drop}</h2>
        <p className="font-body text-sm text-slate-400">{d.explanation}</p>
      </div>

      <p className="max-w-2xl font-body text-sm text-slate-400">
        Impurity importance overrates high-cardinality continuous columns because they offer
        more candidate split points at every node — some of which look good on training data
        by chance alone. Permutation importance sidesteps this by measuring the real
        held-out cost of destroying each column.
      </p>
    </div>
  );
}
