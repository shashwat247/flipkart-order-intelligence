import { useState } from "react";
import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { useReport } from "../../lib/reports";
import type { KbDocument } from "../../lib/types";

// Mirrors streamlit_app/app.py::POLICY_CATEGORY_KEYWORDS exactly, so the same
// document is grouped the same way in both surfaces.
const CATEGORY_KEYWORDS: [string, string[]][] = [
  ["Footwear", ["footwear"]],
  ["Electronics", ["electronics"]],
  ["Home", ["home"]],
  ["Fashion", ["apparel"]],
  ["Exchange", ["exchange"]],
  ["Reverse Pickup", ["reverse_pickup", "reverse pickup"]],
  ["COD", ["cod"]],
  ["Refunds", ["refund"]],
  ["Delivery", ["delivery", "delayed"]],
  ["Returns", ["return", "cancellation", "damaged", "wrong_product", "non_returnable"]],
];

function categoryOf(doc: KbDocument): string {
  const haystack = `${doc.id} ${doc.title}`.toLowerCase();
  for (const [label, keywords] of CATEGORY_KEYWORDS) {
    if (keywords.some((k) => haystack.includes(k))) return label;
  }
  return "General";
}

export default function KnowledgeBase() {
  const report = useReport("part3_kb.json");
  const [expanded, setExpanded] = useState<string | null>(null);
  if (report.status !== "ready") return <EmptyState file="part3_kb.json" />;
  const kb = report.data;
  const grouped = new Map<string, KbDocument[]>();
  for (const doc of kb.documents) {
    const cat = categoryOf(doc);
    grouped.set(cat, [...(grouped.get(cat) ?? []), doc]);
  }

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Knowledge Base" description="The synthetic policy documents, sentence-chunked, each chunk pointing back to its parent.">
        <ProvenanceStrip source="part3/knowledge_base/*.md" script="python3 -m part3.build_index" />
      </ScreenHeader>

      <div className="flex flex-wrap gap-2 font-mono text-xs text-slate-400">
        <span className="rounded-control border border-line bg-ink-800 px-2 py-1">{kb.embedding_model}</span>
        <span className="rounded-control border border-line bg-ink-800 px-2 py-1">{kb.index_backend}</span>
        <span className="rounded-control border border-line bg-ink-800 px-2 py-1">{kb.n_chunks} chunks</span>
        <span className="rounded-control border border-verdant/50 bg-verdant/10 px-2 py-1 text-verdant">{kb.badge}</span>
      </div>

      {Array.from(grouped.entries()).map(([category, docs]) => (
        <div key={category} className="flex flex-col gap-2">
          <h2 className="font-body text-xs font-semibold uppercase tracking-wide text-slate-400">{category}</h2>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            {docs.map((doc) => (
              <div key={doc.id} className="rounded-panel border border-line bg-ink-800 p-3">
                <button
                  type="button"
                  onClick={() => setExpanded(expanded === doc.id ? null : doc.id)}
                  className="w-full text-left"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-body text-sm font-semibold text-paper">{doc.title}</span>
                    <span className="font-mono text-[11px] text-slate-400">{doc.n_chunks} chunks</span>
                  </div>
                  <span className="font-mono text-[11px] text-slate-400">{doc.id}</span>
                </button>
                {expanded === doc.id && (
                  <div className="mt-2 flex flex-col gap-1.5 border-t border-line pt-2">
                    {doc.chunks.map((c) => (
                      <div key={c.chunk_id} className="font-mono text-[11px] text-slate-400">
                        <span className="text-signal">{c.chunk_id}</span>
                        <span className="ml-2 font-body text-xs text-paper">{c.chunk_text}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
