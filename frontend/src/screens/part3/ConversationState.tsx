import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { useReport } from "../../lib/reports";
import type { Transcript } from "../../lib/types";

function Pane({ transcript, title }: { transcript?: Transcript; title: string }) {
  if (!transcript) return <EmptyState file="part3_transcripts.json" />;
  return (
    <div className="flex flex-col gap-3 rounded-panel border border-line bg-ink-800 p-4">
      <h2 className="font-body text-sm font-semibold text-paper">{title}</h2>
      {transcript.turns.map((turn, i) => (
        <div key={i} className="rounded-control border border-line bg-ink-900 p-3">
          <p className="mb-2 font-body text-sm text-paper">
            <span className="text-slate-400">turn {turn.turn_label} — USER: </span>
            {turn.user}
          </p>
          <div className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-0.5 font-mono text-xs">
            {Object.entries(turn.state_after).map(([k, v]) => (
              <>
                <span key={`${k}-k`} className="text-slate-400">{k}</span>
                <span
                  key={`${k}-v`}
                  className={v !== "None" && v !== "none" ? "text-verdant" : "text-slate-400"}
                >
                  {v}
                </span>
              </>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function ConversationState() {
  const report = useReport("part3_transcripts.json");
  if (report.status !== "ready") return <EmptyState file="part3_transcripts.json" />;
  const carried = report.data.transcripts.find((t) => t.filename.includes("multiturn_state_carried"));
  const fresh = report.data.transcripts.find((t) => t.filename.includes("fresh_conversation_state_reset"));

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Conversation State" description="The same state keys, carried across turns in one conversation vs. correctly empty in a fresh one.">
        <ProvenanceStrip source="transcripts/07_multiturn_state_carried.txt, 08_fresh_conversation_state_reset.txt" />
      </ScreenHeader>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Pane transcript={carried} title="Multi-turn — state carried" />
        <Pane transcript={fresh} title="Fresh conversation — same keys, correctly empty" />
      </div>

      <p className="max-w-2xl font-body text-sm text-slate-400">
        Turn 2 of the left conversation names no order id at all — the id resolves from
        state carried forward from turn 1. The right conversation asks the identical
        question with no prior turn, and the same state key comes back empty: the state
        lives on one `Conversation` object, not a global.
      </p>
    </div>
  );
}
