import { GitBranch, GitCommit as GitCommitIcon, GitMerge } from "lucide-react";
import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { useReport } from "../../lib/reports";

export default function GitProvenance() {
  const meta = useReport("project_meta.json");

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Git Provenance" description="Branch/commit/merge history for this repository.">
        <ProvenanceStrip source="git log --graph --all" script="scripts/export_reports.py" />
      </ScreenHeader>

      {meta.status !== "ready" ? (
        <EmptyState file="project_meta.json" />
      ) : (
        <div className="flex flex-col gap-2 rounded-panel border border-line bg-ink-800 p-4">
          <div className="mb-2 flex items-center gap-2 font-mono text-xs text-slate-400">
            <GitBranch size={14} />
            <span>{meta.data.branch}</span>
            <GitCommitIcon size={14} />
            <span>{meta.data.short_commit}</span>
          </div>
          <ul className="flex flex-col gap-1">
            {meta.data.commits.map((c) => (
              <li key={c.hash} className="flex items-center gap-3 rounded-control px-2 py-1 hover:bg-ink-700">
                {c.parents.length > 1 ? (
                  <GitMerge size={14} className="shrink-0 text-tape" />
                ) : (
                  <span className="h-2 w-2 shrink-0 rounded-full bg-signal" />
                )}
                <span className="font-mono text-xs text-slate-400">{c.short_hash}</span>
                <span className="font-mono text-xs text-slate-400">{c.date}</span>
                <span className="truncate font-body text-sm text-paper">{c.subject}</span>
                <span className="ml-auto shrink-0 font-body text-xs text-slate-400">{c.author}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
