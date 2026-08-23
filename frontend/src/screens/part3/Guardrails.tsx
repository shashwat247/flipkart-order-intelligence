import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { useReport } from "../../lib/reports";

export default function Guardrails() {
  const report = useReport("part3_guardrails.json");
  if (report.status !== "ready") return <EmptyState file="part3_guardrails.json" />;
  const g = report.data;

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Guardrails" description="Input side: prompt-injection patterns. Output side: the groundedness check.">
        <ProvenanceStrip source="part3/guardrails.py" />
      </ScreenHeader>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="flex flex-col gap-3">
          <h2 className="font-body text-sm font-semibold text-paper">Input side — {g.injection_patterns.length} patterns checked</h2>
          <div className="flex flex-col gap-1">
            {g.injection_patterns.map((p) => (
              <div key={p.name} className="rounded-control border border-line bg-ink-800 px-2 py-1 font-mono text-[11px] text-slate-400">
                <span className="text-paper">{p.name}</span>
              </div>
            ))}
          </div>
          <h3 className="mt-2 font-body text-xs font-semibold uppercase tracking-wide text-slate-400">Attempts logged</h3>
          {g.injection_examples.map((ex) => (
            <div key={ex.text} className={`rounded-panel border p-3 ${ex.blocked ? "border-flag bg-flag/10" : "border-verdant bg-verdant/10"}`}>
              <p className="font-body text-sm text-paper">{ex.text}</p>
              <p className={`mt-1 font-mono text-xs ${ex.blocked ? "text-flag" : "text-verdant"}`}>
                {ex.blocked ? "BLOCKED" : "not blocked (benign)"}
              </p>
              {ex.matches.map((m, i) => (
                <p key={i} className="font-mono text-[11px] text-slate-400">
                  {m.pattern} → "{m.matched_text}"
                </p>
              ))}
            </div>
          ))}
        </div>

        <div className="flex flex-col gap-3">
          <h2 className="font-body text-sm font-semibold text-paper">Output side — groundedness</h2>
          <div className="rounded-panel border border-line bg-ink-800 p-4">
            <p className="mb-2 font-mono text-sm text-paper">similarity floor = {g.similarity_threshold.toFixed(2)}</p>
            <div className="relative h-3 w-full rounded-full bg-ink-900">
              <div
                className={`h-3 rounded-full ${g.ungrounded_example.grounded ? "bg-verdant" : "bg-flag"}`}
                style={{ width: `${g.ungrounded_example.best_score * 100}%` }}
              />
              <div className="absolute top-0 h-3 w-px bg-paper" style={{ left: `${g.similarity_threshold * 100}%` }} />
            </div>
            <p className="mt-2 font-body text-xs text-slate-400">"{g.ungrounded_example.query}"</p>
            <p className="font-mono text-xs text-flag">
              best score {g.ungrounded_example.best_score.toFixed(4)} — {g.ungrounded_example.grounded ? "grounded" : "refused, not fabricated"}
            </p>
          </div>
          <h3 className="mt-2 font-body text-xs font-semibold uppercase tracking-wide text-slate-400">Out-of-domain calibration set</h3>
          <ul className="flex flex-col gap-1">
            {g.out_of_domain_queries.map((q) => (
              <li key={q} className="font-body text-xs text-slate-400">— {q}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
