import { CheckCircle2, XCircle } from "lucide-react";
import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { useReport } from "../../lib/reports";

export default function SavedArtifact() {
  const report = useReport("part1_artifact.json");
  if (report.status !== "ready") return <EmptyState file="part1_artifact.json" />;
  const a = report.data;

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Saved Artifact" description="Task 9 — the persisted, tuned Random Forest pipeline and its threshold.">
        <ProvenanceStrip source={a.path} script="python3 -m part1.train_return_risk" />
      </ScreenHeader>

      <div className={`flex w-fit items-center gap-2 rounded-control border px-3 py-1.5 font-body text-sm ${
        a.loads_ok ? "border-verdant text-verdant" : "border-flag text-flag"
      }`}>
        {a.loads_ok ? <CheckCircle2 size={16} /> : <XCircle size={16} />}
        joblib.load {a.loads_ok ? "succeeded" : "failed"}
      </div>

      <div className="rounded-panel border border-line bg-ink-800 p-6">
        <span className="font-display text-sm text-slate-400">t*_rf</span>
        <div className="font-mono text-[56px] leading-none text-paper">{a.t_star_rf.toFixed(2)}</div>
        <div className="mt-4 flex gap-3 font-mono text-sm">
          <span className="rounded-control border border-verdant/50 bg-verdant/10 px-2 py-1 text-verdant">
            Low &lt; {a.buckets.low_max.toFixed(2)}
          </span>
          <span className="rounded-control border border-tape/50 bg-tape/10 px-2 py-1 text-tape">
            Medium up to {a.buckets.medium_max.toFixed(2)}
          </span>
          <span className="rounded-control border border-flag/50 bg-flag/10 px-2 py-1 text-flag">
            High ≥ {a.buckets.medium_max.toFixed(2)}
          </span>
        </div>
        <p className="mt-4 max-w-2xl font-body text-sm text-slate-400">{a.justification_sentence}</p>
      </div>

      <div className="flex gap-6 font-mono text-sm text-paper">
        <span>model: {a.model_type}</span>
        <span>test ROC-AUC: {a.test_roc_auc.toFixed(4)}</span>
        <span>n_estimators={a.best_params.n_estimators}, max_depth={a.best_params.max_depth ?? "None"}</span>
      </div>
    </div>
  );
}
