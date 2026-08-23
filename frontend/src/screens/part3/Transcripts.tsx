import { useState } from "react";
import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { GraphRail } from "../../components/GraphRail";
import { JsonCard } from "../../components/JsonCard";
import { useReport } from "../../lib/reports";

const CATEGORY_TAGS: Record<string, string> = {
  "01_policy_electronics_return_window.txt": "policy-RAG",
  "02_policy_cod_refund_timeline.txt": "policy-RAG",
  "03_return_risk_high.txt": "return-risk tool",
  "04_return_risk_low.txt": "return-risk tool",
  "05_product_category_sneaker.txt": "product-category tool",
  "06_product_category_shirt.txt": "product-category tool",
  "07_multiturn_state_carried.txt": "multi-turn state",
  "08_fresh_conversation_state_reset.txt": "fresh-conversation state",
  "09_prompt_injection_blocked.txt": "injection deflected",
  "10_ungrounded_policy_refused.txt": "ungrounded refusal",
};

export default function Transcripts() {
  const report = useReport("part3_transcripts.json");
  const [filter, setFilter] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  if (report.status !== "ready") return <EmptyState file="part3_transcripts.json" />;
  const { transcripts } = report.data;
  const categories = Array.from(new Set(Object.values(CATEGORY_TAGS)));
  const visible = filter ? transcripts.filter((t) => CATEGORY_TAGS[t.filename] === filter) : transcripts;
  const active = transcripts.find((t) => t.filename === selected) ?? visible[0];

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Transcripts" description="All recorded conversations, produced by the real agent in MOCK_LLM mode.">
        <ProvenanceStrip source="transcripts/*.txt" script="python3 -m part3.run_transcripts" />
      </ScreenHeader>

      <div className="flex flex-wrap gap-1.5">
        <button
          type="button"
          onClick={() => setFilter(null)}
          className={`rounded-control border px-2 py-1 font-mono text-xs ${!filter ? "border-signal bg-signal/15 text-signal" : "border-line text-slate-400"}`}
        >
          all
        </button>
        {categories.map((c) => (
          <button
            key={c}
            type="button"
            onClick={() => setFilter(c)}
            className={`rounded-control border px-2 py-1 font-mono text-xs ${filter === c ? "border-signal bg-signal/15 text-signal" : "border-line text-slate-400"}`}
          >
            {c}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[280px_1fr]">
        <div className="flex flex-col gap-1">
          {visible.map((t) => (
            <button
              key={t.filename}
              type="button"
              onClick={() => setSelected(t.filename)}
              className={`rounded-control border px-2 py-1.5 text-left font-mono text-xs ${
                active?.filename === t.filename ? "border-signal bg-signal/10 text-paper" : "border-line text-slate-400 hover:text-paper"
              }`}
            >
              {t.filename}
            </button>
          ))}
        </div>

        {active && (
          <div className="flex flex-col gap-4">
            <p className="font-body text-sm text-slate-400">{active.header["Demonstrates"]}</p>
            {active.turns.map((turn, i) => (
              <div key={i} className="grid grid-cols-[1fr_auto] gap-4 rounded-panel border border-line bg-ink-800 p-4">
                <div className="flex flex-col gap-2">
                  <p className="font-body text-sm text-paper">
                    <span className="text-slate-400">USER: </span>
                    {turn.user}
                  </p>
                  {turn.response && <JsonCard data={turn.response} title="response" />}
                </div>
                <GraphRail path={turn.graph_path} blocked={turn.blocked} orientation="horizontal" />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
