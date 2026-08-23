import { CheckCircle2, XCircle } from "lucide-react";
import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { useReport } from "../../lib/reports";
import { useHashRoute } from "../../lib/router";

function StatusCard({ title, ready, detail, onClick }: { title: string; ready: boolean; detail: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex flex-col gap-2 rounded-panel border border-line bg-ink-800 p-4 text-left hover:border-signal/50"
    >
      <div className="flex items-center gap-2">
        {ready ? <CheckCircle2 size={16} className="text-verdant" /> : <XCircle size={16} className="text-flag" />}
        <span className="font-body text-sm font-semibold text-paper">{title}</span>
      </div>
      <p className="font-mono text-xs text-slate-400">{detail}</p>
    </button>
  );
}

function Edge({ label, target, onClick }: { label: string; target: string; onClick: (p: string) => void }) {
  return (
    <button
      type="button"
      onClick={() => onClick(target)}
      className="rounded-control border border-line bg-ink-900 px-2 py-1 font-mono text-xs text-slate-400 hover:border-signal hover:text-signal"
    >
      {label}
    </button>
  );
}

export default function Overview() {
  const [, navigate] = useHashRoute();
  const p1 = useReport("part1_artifact.json");
  const p2 = useReport("part2_artifact.json");
  const p3 = useReport("part3_kb.json");
  const data = useReport("part1_data.json");

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader
        title="Overview"
        description="Every artifact this console inspects, at a glance."
      >
        <ProvenanceStrip source="models/*.json, orders_dataset.csv, part3 knowledge base" script="scripts/export_reports.py" />
      </ScreenHeader>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <StatusCard
          title="Dataset (Part 1)"
          ready={data.status === "ready" && data.data.rows === 6000}
          detail={data.status === "ready" ? `${data.data.rows} rows, ${data.data.columns} columns` : "not loaded"}
          onClick={() => navigate("/p1/data-verification")}
        />
        <StatusCard
          title="Return-Risk Model (Part 1)"
          ready={p1.status === "ready" && p1.data.loads_ok}
          detail={p1.status === "ready" ? `${p1.data.model_type}, t*_rf=${p1.data.t_star_rf.toFixed(2)}` : "not loaded"}
          onClick={() => navigate("/p1/saved-artifact")}
        />
        <StatusCard
          title="Image Classifier (Part 2)"
          ready={p2.status === "ready" && p2.data.loads_ok}
          detail={p2.status === "ready" ? p2.data.architecture : "not loaded"}
          onClick={() => navigate("/p2/artifact-samples")}
        />
      </div>

      <div className="rounded-panel border border-line bg-ink-800 p-5">
        <h2 className="mb-3 font-body text-sm font-semibold text-paper">System dataflow</h2>
        {p3.status !== "ready" ? (
          <EmptyState file="part3_kb.json" />
        ) : (
          <div className="flex flex-col gap-4 font-mono text-xs text-slate-400">
            <div className="flex flex-wrap items-center gap-2">
              <Edge label="generate_orders.py" target="/p1/dataset-generator" onClick={navigate} />
              <span>→</span>
              <Edge label="orders_dataset.csv" target="/p1/data-verification" onClick={navigate} />
              <span>→</span>
              <Edge label="Random Forest" target="/p1/rf-tuning" onClick={navigate} />
              <span>→</span>
              <Edge label="return_risk_model.pkl" target="/p1/saved-artifact" onClick={navigate} />
              <span>→</span>
              <Edge label="agent tool" target="/p3/tools" onClick={navigate} />
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Edge label="Fashion-MNIST" target="/p2/dataset" onClick={navigate} />
              <span>→</span>
              <Edge label="backbone + head" target="/p2/training" onClick={navigate} />
              <span>→</span>
              <Edge label="product_classifier.pt" target="/p2/artifact-samples" onClick={navigate} />
              <span>→</span>
              <Edge label="agent tool" target="/p3/tools" onClick={navigate} />
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Edge label={`policy KB (${p3.data.n_documents} docs)`} target="/p3/knowledge-base" onClick={navigate} />
              <span>→</span>
              <Edge label={`${p3.data.n_chunks} chunks`} target="/p3/knowledge-base" onClick={navigate} />
              <span>→</span>
              <Edge label="FAISS index" target="/p3/retrieval-explorer" onClick={navigate} />
              <span>→</span>
              <Edge label="agent" target="/p3/agent-console" onClick={navigate} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
