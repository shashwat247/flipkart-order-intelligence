import { FileWarning } from "lucide-react";
import { EXPORT_COMMAND } from "../lib/reports";
import { CodeBlock } from "./CodeBlock";

/** Hard rule 1: no fabricated metrics. A missing/malformed report renders
 * this — naming the exact file and the command that produces it — never a
 * placeholder number. */
export function EmptyState({
  file,
  detail,
}: {
  file: string;
  detail?: string;
}) {
  return (
    <div className="flex flex-col items-start gap-3 rounded-panel border border-line bg-ink-800 p-6">
      <div className="flex items-center gap-2 text-flag">
        <FileWarning size={18} strokeWidth={2} aria-hidden />
        <span className="font-body text-sm font-medium">Report not available</span>
      </div>
      <p className="font-body text-sm text-slate-400">
        <span className="font-mono text-paper">{file}</span> is missing
        {detail ? <> or malformed ({detail})</> : null}. Generate it with:
      </p>
      <CodeBlock code={EXPORT_COMMAND} />
    </div>
  );
}
