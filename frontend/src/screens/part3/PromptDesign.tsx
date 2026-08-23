import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { DataTable } from "../../components/DataTable";
import { useReport } from "../../lib/reports";
import type { FewShotExampleLive } from "../../lib/types";

export default function PromptDesign() {
  const report = useReport("part3_prompt.json");
  if (report.status !== "ready") return <EmptyState file="part3_prompt.json" />;
  const p = report.data;

  const columns = [
    { key: "user", header: "example" },
    { key: "fine_intent", header: "fine intent", mono: true },
    { key: "live_lane", header: "live route", mono: true, render: (r: FewShotExampleLive) => (
      <span className="rounded-control border border-signal/50 bg-signal/10 px-1.5 py-0.5 text-signal">{r.live_lane}</span>
    ) },
  ];

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Prompt Design" description="The system prompt, engineered against the 4S principles + role prompting.">
        <ProvenanceStrip source="part3/prompts.py" />
      </ScreenHeader>

      <pre className="whitespace-pre-wrap rounded-panel border border-line bg-ink-800 p-4 font-mono text-xs leading-relaxed text-paper">
        {p.system_prompt}
      </pre>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {Object.entries(p.principle_annotations).map(([principle, text]) => (
          <div key={principle} className="rounded-panel border border-line bg-ink-800 p-3">
            <h2 className="mb-1 font-body text-sm font-semibold text-signal">{principle}</h2>
            <p className="font-body text-xs text-slate-400">{text}</p>
          </div>
        ))}
      </div>

      <div>
        <h2 className="mb-2 font-body text-sm font-semibold text-paper">
          Few-shot examples, paired with the route they actually produce right now
        </h2>
        <DataTable columns={columns} rows={p.few_shot_examples} rowKey={(r) => r.user} />
      </div>
    </div>
  );
}
