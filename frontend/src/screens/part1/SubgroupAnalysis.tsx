import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { RankedBars } from "../../components/RankedBars";
import { useReport } from "../../lib/reports";
import type { SubgroupRow } from "../../lib/types";

function toDelta(rows: SubgroupRow[], overallRecall: number, weakestSubgroup: string) {
  return rows.map((r) => ({
    label: r.subgroup,
    value: r.recall - overallRecall,
    flag: r.subgroup === weakestSubgroup,
  }));
}

export default function SubgroupAnalysis() {
  const report = useReport("part1_subgroups.json");
  if (report.status !== "ready") return <EmptyState file="part1_subgroups.json" />;
  const d = report.data;

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Subgroup Analysis" description="Task 8 — recall/precision by product_category and payment_method, at t*_rf.">
        <ProvenanceStrip
          source="reports/part1_subgroup_product_category.csv, part1_subgroup_payment_method.csv"
          script="python3 -m part1.subgroup_analysis"
        />
      </ScreenHeader>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div>
          <h2 className="mb-3 font-body text-sm font-semibold text-paper">Recall delta vs. overall — by product_category</h2>
          <RankedBars diverging items={toDelta(d.by_category, d.overall.recall, d.weakest.subgroup)} />
        </div>
        <div>
          <h2 className="mb-3 font-body text-sm font-semibold text-paper">Recall delta vs. overall — by payment_method</h2>
          <RankedBars diverging items={toDelta(d.by_payment, d.overall.recall, d.weakest.subgroup)} />
        </div>
      </div>

      <p className="font-mono text-xs text-slate-400">
        overall @ t*_rf={d.overall.threshold_rf.toFixed(2)}: precision {d.overall.precision.toFixed(4)}, recall{" "}
        {d.overall.recall.toFixed(4)}, F1 {d.overall.f1.toFixed(4)}
      </p>

      <div className="max-w-2xl rounded-panel border border-flag bg-flag/10 p-4">
        <h2 className="mb-1 font-body text-sm font-semibold text-flag">
          Weakest subgroup: {d.weakest.by} = {d.weakest.subgroup}
        </h2>
        <p className="mb-2 font-mono text-xs text-slate-400">
          recall {d.weakest.recall.toFixed(4)} vs. overall {d.overall.recall.toFixed(4)} — a gap of{" "}
          {(d.weakest.recall_gap * 100).toFixed(2)} pp {d.weakest.material ? "(material)" : "(below the material bar)"}
        </p>
        <p className="font-body text-sm text-slate-400">{d.proposed_fix}</p>
      </div>
    </div>
  );
}
