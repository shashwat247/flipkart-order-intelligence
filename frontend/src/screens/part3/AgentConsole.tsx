import { useState } from "react";
import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { GraphRail } from "../../components/GraphRail";
import { JsonCard } from "../../components/JsonCard";
import { useReport } from "../../lib/reports";
import { useHashRoute } from "../../lib/router";

const SOURCE_TARGET: Record<string, string> = {
  policy_kb: "/p3/knowledge-base",
  return_risk_tool: "/p3/tools",
  image_classifier_tool: "/p3/tools",
};

export default function AgentConsole() {
  const report = useReport("part3_transcripts.json");
  const [index, setIndex] = useState(0);
  const [rawIndex, setRawIndex] = useState<number | null>(null);
  const [, navigate] = useHashRoute();
  if (report.status !== "ready") return <EmptyState file="part3_transcripts.json" />;
  const { transcripts } = report.data;
  const transcript = transcripts[index];

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Agent Console" description="Replays a real recorded conversation through the LangGraph agent — MOCK_LLM, zero API keys.">
        <ProvenanceStrip source="transcripts/*.txt" script="python3 -m part3.run_transcripts" />
      </ScreenHeader>

      <select
        value={index}
        onChange={(e) => setIndex(Number(e.target.value))}
        className="w-fit rounded-control border border-line bg-ink-800 px-3 py-1.5 font-mono text-sm text-paper"
      >
        {transcripts.map((t, i) => (
          <option key={t.filename} value={i}>
            {t.filename}
          </option>
        ))}
      </select>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_auto_360px]">
        <div className="flex flex-col gap-3">
          {transcript.turns.map((turn, i) => (
            <div key={i} className="flex flex-col gap-2">
              <div className="w-fit max-w-lg rounded-panel border border-line bg-ink-800 px-3 py-2 font-body text-sm text-paper">
                {turn.user}
              </div>
              {turn.response && (
                <div className="ml-auto w-fit max-w-lg rounded-panel border border-signal/40 bg-signal/10 px-3 py-2">
                  <p className="font-body text-sm text-paper">{turn.response.answer}</p>
                  <div className="mt-2 flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => navigate(SOURCE_TARGET[turn.response!.source] ?? "/p3/graph-inspector")}
                      className="rounded-control border border-line bg-ink-900 px-1.5 py-0.5 font-mono text-[11px] text-signal hover:border-signal"
                    >
                      {turn.response.source}
                    </button>
                    <div className="h-1.5 w-20 rounded-full bg-ink-900">
                      <div className="h-1.5 rounded-full bg-verdant" style={{ width: `${turn.response.confidence * 100}%` }} />
                    </div>
                    <span className="font-mono text-[11px] text-slate-400">{(turn.response.confidence * 100).toFixed(1)}%</span>
                    <button
                      type="button"
                      onClick={() => setRawIndex(rawIndex === i ? null : i)}
                      className="ml-auto font-mono text-[11px] text-slate-400 hover:text-paper"
                    >
                      {rawIndex === i ? "hide raw" : "raw JSON"}
                    </button>
                  </div>
                  {rawIndex === i && <div className="mt-2"><JsonCard data={turn.response} /></div>}
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="hidden lg:block">
          <GraphRail path={transcript.turns[transcript.turns.length - 1]?.graph_path ?? []} blocked={transcript.turns.some((t) => t.blocked)} />
        </div>

        <div className="flex flex-col gap-2 rounded-panel border border-line bg-ink-800 p-3">
          <h2 className="font-body text-xs font-semibold uppercase tracking-wide text-slate-400">Inspector</h2>
          {Object.entries(transcript.header).map(([k, v]) => (
            <div key={k} className="font-mono text-[11px]">
              <span className="text-slate-400">{k}: </span>
              <span className="text-paper">{v}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
