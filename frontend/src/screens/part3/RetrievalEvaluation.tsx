import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { MetricTile } from "../../components/MetricTile";
import { useReport } from "../../lib/reports";

export default function RetrievalEvaluation() {
  const report = useReport("part3_retrieval_eval.json");
  if (report.status !== "ready") return <EmptyState file="part3_retrieval_eval.json" />;
  const d = report.data;

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Retrieval Evaluation" description="Document-level Precision@3 and Recall@3 against a fixed answer key, written before retrieval ever ran.">
        <ProvenanceStrip source="part3/eval_queries.py" script="python3 -m part3.evaluate_retrieval" />
      </ScreenHeader>

      <div className="flex gap-4">
        <MetricTile label="Average Precision@3" value={d.mean_precision_at_3.toFixed(4)} hero />
        <MetricTile label="Average Recall@3" value={d.mean_recall_at_3.toFixed(4)} />
      </div>

      <p className="max-w-2xl font-body text-sm text-slate-400">
        Precision@3 is structurally capped at 1/3 for most of these queries (they have only
        one relevant document but the top 3 are retrieved) — Recall@3 is the metric that
        actually asks whether the right document made it in front of the response generator.
      </p>

      <div className="flex flex-col gap-3">
        {d.queries.map((q, i) => (
          <div key={q.query} className="rounded-panel border border-line bg-ink-800 p-4">
            <p className="mb-1 font-body text-sm text-paper">
              {i + 1}. {q.query}
            </p>
            <p className="mb-2 font-body text-xs text-slate-400">{q.rationale}</p>
            <div className="flex flex-wrap gap-4 font-mono text-xs">
              <span className="text-slate-400">
                relevant = {"{"}
                {q.relevant.join(", ")}
                {"}"}
              </span>
              <span className="text-slate-400">
                retrieved = {"{"}
                {q.retrieved.join(", ")}
                {"}"}
              </span>
            </div>
            <div className="mt-2 flex gap-4 font-mono text-xs text-paper">
              <span>Precision@3 = {q.n_hits}/3 = {q.precision_at_3.toFixed(4)}</span>
              <span>Recall@3 = {q.n_hits}/{q.relevant.length} = {q.recall_at_3.toFixed(4)}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
