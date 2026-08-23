import { useState } from "react";
import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { ChunkCard } from "../../components/ChunkCard";
import { useReport } from "../../lib/reports";

export default function RetrievalExplorer() {
  const report = useReport("part3_retrieval_eval.json");
  const [index, setIndex] = useState(0);
  if (report.status !== "ready") return <EmptyState file="part3_retrieval_eval.json" />;
  const d = report.data;
  const row = d.queries[index];

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader
        title="Retrieval Explorer"
        description="Query → top-k chunks ranked by cosine similarity against the groundedness floor. Restricted to the pre-scored evaluation queries — this is a static console with no backend to run a fresh embedding search against."
      >
        <ProvenanceStrip source="part3/eval_queries.py, FAISS index" script="python3 -m part3.evaluate_retrieval" />
      </ScreenHeader>

      <select
        value={index}
        onChange={(e) => setIndex(Number(e.target.value))}
        className="w-fit max-w-full rounded-control border border-line bg-ink-800 px-3 py-1.5 font-mono text-sm text-paper"
      >
        {d.queries.map((q, i) => (
          <option key={q.query} value={i}>
            {q.query}
          </option>
        ))}
      </select>

      <p className="max-w-2xl font-body text-xs text-slate-400">{row.rationale}</p>

      <div className="flex flex-col gap-2">
        {row.retrieved_chunks.map((c) => (
          <ChunkCard
            key={c.document_id}
            chunkText={c.chunk_text}
            score={c.score}
            threshold={d.similarity_threshold}
            documentTitle={c.document_title}
            documentId={c.document_id}
          />
        ))}
      </div>
    </div>
  );
}
