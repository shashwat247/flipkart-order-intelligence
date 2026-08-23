import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { MetricTile } from "../../components/MetricTile";
import { Heatmap } from "../../components/Heatmap";
import { useReport } from "../../lib/reports";

export default function RfTuning() {
  const report = useReport("part1_rf_grid.json");
  if (report.status !== "ready") return <EmptyState file="part1_rf_grid.json" />;
  const g = report.data;

  const estimatorsSet = Array.from(new Set(g.cells.map((c) => c.n_estimators))).sort((a, b) => a - b);
  const depthSet = Array.from(new Set(g.cells.map((c) => c.max_depth))).sort((a, b) => (a ?? 999) - (b ?? 999));
  const values = depthSet.map((depth) =>
    estimatorsSet.map((n) => g.cells.find((c) => c.n_estimators === n && c.max_depth === depth)?.cv_roc_auc ?? 0)
  );
  const bestRow = depthSet.findIndex((d) => d === g.best_params.max_depth);
  const bestCol = estimatorsSet.findIndex((n) => n === g.best_params.n_estimators);
  const overfitOk = g.auc_gap <= 0.05;

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Random Forest Tuning" description="Task 6 — GridSearchCV over n_estimators × max_depth, 5-fold, scored on ROC-AUC.">
        <ProvenanceStrip source="reports/part1_model_report.md, models/return_risk_metadata.json" script="python3 -m part1.train_return_risk" />
      </ScreenHeader>

      <div className="flex flex-wrap items-start gap-6">
        <Heatmap
          rowLabels={depthSet.map((d) => `depth=${d ?? "None"}`)}
          colLabels={estimatorsSet.map((n) => `n=${n}`)}
          values={values}
          bestCell={bestRow >= 0 && bestCol >= 0 ? [bestRow, bestCol] : undefined}
        />
        <div className="flex flex-col gap-3">
          <MetricTile label="Best CV ROC-AUC" value={g.best_cv_roc_auc.toFixed(4)} hero />
          <MetricTile label="Held-out test ROC-AUC" value={g.test_roc_auc.toFixed(4)} />
          <MetricTile
            label="|CV − test| gap (overfitting check, ≤0.05)"
            value={g.auc_gap.toFixed(4)}
            delta={overfitOk ? "within tolerance" : "exceeds tolerance"}
            deltaGood={overfitOk}
          />
          <p className="max-w-xs font-mono text-xs text-slate-400">
            best params: n_estimators={g.best_params.n_estimators}, max_depth={g.best_params.max_depth ?? "None"}
          </p>
        </div>
      </div>
    </div>
  );
}
