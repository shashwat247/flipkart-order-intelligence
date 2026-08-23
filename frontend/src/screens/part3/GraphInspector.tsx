import { useState } from "react";
import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { GraphRail } from "../../components/GraphRail";
import { JsonCard } from "../../components/JsonCard";
import { useReport } from "../../lib/reports";

const NODE_DESCRIPTIONS: Record<string, { role: string; io: string }> = {
  guard_node: {
    role: "INPUT GUARDRAIL — scans the raw user text for prompt-injection patterns before any retrieval or model call.",
    io: "in: query (str)  ->  out: injection {blocked, matches[]}",
  },
  intent_node: {
    role: "Classifies intent via nearest few-shot exemplar (cosine similarity) and resolves order-id/state carried from earlier turns.",
    io: "in: query, carried state  ->  out: intent, intent_evidence, order_id, order_features",
  },
  retrieval_node: {
    role: "THE CONDITIONAL BRANCH (policy lane). Top-k FAISS chunk search, rolled up to parent documents, plus the groundedness check.",
    io: "in: query  ->  out: chunk_hits, doc_hits, groundedness",
  },
  tool_node: {
    role: "THE CONDITIONAL BRANCH (return_risk / product_category lane). Calls the real saved Random Forest or ResNet-18.",
    io: "in: order_features or image_path  ->  out: tool_result",
  },
  response_node: {
    role: "Composes the final {answer, source, confidence} JSON from whatever the earlier nodes produced. MOCK_LLM: pure templating, cannot invent a policy.",
    io: "in: everything above  ->  out: response",
  },
};

export default function GraphInspector() {
  const report = useReport("part3_transcripts.json");
  const [selectedNode, setSelectedNode] = useState<string>("intent_node");
  if (report.status !== "ready") return <EmptyState file="part3_transcripts.json" />;
  const { transcripts } = report.data;

  // The last transcript turn whose graph_path actually passed through the
  // selected node — real evidence, not a placeholder.
  let lastTrace: { filename: string; turnLabel: string; response: unknown } | null = null;
  for (const t of transcripts) {
    for (const turn of t.turns) {
      if (turn.graph_path.includes(selectedNode) || (selectedNode === "direct" && turn.graph_path.length <= 3)) {
        lastTrace = { filename: t.filename, turnLabel: turn.turn_label, response: turn.response };
      }
    }
  }

  const desc = NODE_DESCRIPTIONS[selectedNode];

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Graph Inspector" description="The full LangGraph node graph. The conditional edge genuinely branches — a policy question never touches the tool node.">
        <ProvenanceStrip source="part3/graph.py" />
      </ScreenHeader>

      <div className="rounded-panel border border-line bg-ink-800 p-6">
        <GraphRail
          path={["guard_node", "intent_node", selectedNode === "retrieval_node" ? "retrieval_node" : selectedNode === "tool_node" ? "tool_node" : "intent_node", "response_node"]}
          orientation="horizontal"
          onNodeClick={setSelectedNode}
        />
      </div>

      {desc && (
        <div className="rounded-panel border border-line bg-ink-800 p-4">
          <h2 className="mb-1 font-mono text-sm font-semibold text-signal">{selectedNode}</h2>
          <p className="mb-2 font-body text-sm text-paper">{desc.role}</p>
          <p className="font-mono text-xs text-slate-400">{desc.io}</p>
        </div>
      )}

      {lastTrace ? (
        <div className="flex flex-col gap-2">
          <h2 className="font-body text-sm font-semibold text-paper">
            Last trace through this node: {lastTrace.filename} ({lastTrace.turnLabel})
          </h2>
          {lastTrace.response ? <JsonCard data={lastTrace.response} /> : (
            <p className="font-body text-sm text-slate-400">This turn produced no structured response (blocked before response_node ran isn't applicable here).</p>
          )}
        </div>
      ) : (
        <p className="font-body text-sm text-slate-400">No recorded transcript passed through this node.</p>
      )}
    </div>
  );
}
