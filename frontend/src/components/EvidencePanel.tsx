import { useState } from "react";
import { ChevronRight, FileText, ShieldAlert, ShieldCheck } from "lucide-react";
import type { Turn } from "../lib/agent";

const SOURCE_LABEL: Record<string, string> = {
  policy_kb: "Policy KB",
  return_risk_tool: "Return-risk model",
  image_classifier_tool: "Image classifier",
  product_catalog: "Product catalog",
  conversational: "Conversation",
};

/** The evidence behind one answer: which nodes ran, what was retrieved, and
 * whether the groundedness floor was cleared. Every value is the agent's own —
 * this panel is what lets a reader check the answer instead of trusting it. */
export function EvidencePanel({ turn }: { turn: Turn }) {
  const [open, setOpen] = useState(false);
  const g = turn.groundedness;

  return (
    <div className="mt-3 border-t border-line pt-2.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 font-mono text-xs text-slate-400">
        <span className="rounded-control bg-ink-700 px-2 py-0.5 text-paper">
          {SOURCE_LABEL[turn.response.source] ?? turn.response.source}
        </span>
        <span>confidence {(turn.response.confidence * 100).toFixed(2)}%</span>
        <span>intent {turn.intent}</span>

        {g && (
          <span className={`inline-flex items-center gap-1 ${g.grounded ? "text-verdant" : "text-flag"}`}>
            {g.grounded ? <ShieldCheck size={12} aria-hidden /> : <ShieldAlert size={12} aria-hidden />}
            grounded {String(g.grounded)} ({g.best_score.toFixed(4)} vs {g.threshold})
          </span>
        )}

        {turn.injection?.blocked && (
          <span className="inline-flex items-center gap-1 text-flag">
            <ShieldAlert size={12} aria-hidden /> injection blocked
          </span>
        )}
      </div>

      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="mt-2 inline-flex items-center gap-1 rounded-control font-body text-xs text-signal hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-signal"
      >
        <ChevronRight
          size={12}
          aria-hidden
          className={`transition-transform duration-node ${open ? "rotate-90" : ""}`}
        />
        {open ? "Hide" : "Show"} evidence
      </button>

      {open && (
        <div className="mt-2 flex flex-col gap-3">
          <div>
            <p className="mb-1 font-body text-xs font-semibold uppercase tracking-wide text-slate-400">
              Graph path
            </p>
            <div className="flex flex-wrap items-center gap-1.5">
              {turn.trace.map((node, i) => (
                <span key={`${node}-${i}`} className="flex items-center gap-1.5">
                  <code className="rounded-control bg-ink-700 px-1.5 py-0.5 font-mono text-xs text-paper">
                    {node}
                  </code>
                  {i < turn.trace.length - 1 && <span className="text-tape" aria-hidden>→</span>}
                </span>
              ))}
            </div>
          </div>

          {turn.doc_hits.length > 0 && (
            <div>
              <p className="mb-1 font-body text-xs font-semibold uppercase tracking-wide text-slate-400">
                Retrieved policy documents
              </p>
              <ul className="flex flex-col gap-1">
                {turn.doc_hits.map((d) => (
                  <li key={d.document_id} className="flex items-start gap-2 font-mono text-xs">
                    <FileText size={12} className="mt-0.5 shrink-0 text-slate-400" aria-hidden />
                    <span className="text-paper">{d.title}</span>
                    <span className="text-slate-400">({d.document_id})</span>
                    <span className="ml-auto shrink-0 text-signal">{d.score.toFixed(4)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {turn.intent_evidence?.matched_example && (
            <div>
              <p className="mb-1 font-body text-xs font-semibold uppercase tracking-wide text-slate-400">
                Nearest few-shot exemplar
              </p>
              <p className="font-mono text-xs text-slate-400">
                “{turn.intent_evidence.matched_example}” @{" "}
                <span className="text-paper">{turn.intent_evidence.similarity?.toFixed(4)}</span>
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
