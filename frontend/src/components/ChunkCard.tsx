export function ChunkCard({
  chunkText,
  score,
  threshold,
  documentTitle,
  documentId,
  chunkId,
}: {
  chunkText: string;
  score: number;
  threshold: number;
  documentTitle: string;
  documentId: string;
  chunkId?: string;
}) {
  const qualifies = score >= threshold;
  const barPct = Math.min(100, Math.max(0, score * 100));
  const thresholdPct = Math.min(100, Math.max(0, threshold * 100));

  return (
    <div
      className={`flex flex-col gap-2 rounded-panel border p-3 ${
        qualifies ? "border-line bg-ink-800" : "border-line bg-ink-800 opacity-50"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="font-body text-xs text-slate-400">
          {documentTitle} <span className="font-mono text-slate-400">({documentId}{chunkId ? `::${chunkId.split("::")[1]}` : ""})</span>
        </span>
        <span className="font-mono text-xs text-paper">{score.toFixed(4)}</span>
      </div>
      <p className="font-body text-sm text-paper">{chunkText}</p>
      <div className="relative h-1.5 w-full rounded-full bg-ink-900">
        <div
          className={`h-1.5 rounded-full ${qualifies ? "bg-signal" : "bg-line"}`}
          style={{ width: `${barPct}%` }}
        />
        <div
          className="absolute top-0 h-1.5 w-px bg-flag"
          style={{ left: `${thresholdPct}%` }}
          title={`similarity floor ${threshold.toFixed(2)}`}
        />
      </div>
      {!qualifies && (
        <span className="font-mono text-xs text-flag">below the {threshold.toFixed(2)} similarity floor</span>
      )}
    </div>
  );
}
