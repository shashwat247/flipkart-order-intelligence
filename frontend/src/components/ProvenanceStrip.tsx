/** Every screen header carries this: the source file, the script that
 * generated it, and (optionally) a relative timestamp — in mono, since it's
 * data about the data, not prose. */
export function ProvenanceStrip({
  source,
  script,
  timestamp,
}: {
  source: string;
  script?: string;
  timestamp?: string;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-[12px] text-slate-400">
      <span className="rounded-control border border-line bg-ink-800 px-1.5 py-0.5">{source}</span>
      {script && (
        <>
          <span aria-hidden>·</span>
          <span>regenerate: {script}</span>
        </>
      )}
      {timestamp && (
        <>
          <span aria-hidden>·</span>
          <span>{timestamp}</span>
        </>
      )}
    </div>
  );
}
