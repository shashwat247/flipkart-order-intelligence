import { Command, Menu } from "lucide-react";
import { useReport } from "../lib/reports";

export function TopBar({
  onOpenPalette,
  onToggleMobileNav,
}: {
  onOpenPalette: () => void;
  onToggleMobileNav: () => void;
}) {
  const meta = useReport("project_meta.json");
  const branch = meta.status === "ready" ? meta.data.branch : "…";
  const shortCommit = meta.status === "ready" ? meta.data.short_commit : "……";

  return (
    <header className="flex h-12 shrink-0 items-center gap-3 border-b border-line bg-ink-800 px-3 sm:gap-4 sm:px-4">
      <button
        type="button"
        onClick={onToggleMobileNav}
        aria-label="Toggle navigation"
        className="shrink-0 rounded-control border border-line p-1.5 text-slate-400 hover:text-paper lg:hidden"
      >
        <Menu size={16} />
      </button>
      <span className="shrink-0 font-display text-sm font-bold tracking-tight text-paper">Order Intelligence Console</span>
      <span className="hidden truncate font-mono text-xs text-slate-400 md:inline">
        flipkart-order-intelligence · {branch} @ {shortCommit}
      </span>
      <span className="hidden shrink-0 items-center gap-1.5 rounded-control border border-line bg-ink-900 px-2 py-0.5 font-mono text-[11px] text-tape sm:flex">
        <span className="h-1.5 w-1.5 rounded-full bg-tape" />
        MOCK_LLM · zero API keys · zero network
      </span>
      <button
        type="button"
        onClick={onOpenPalette}
        className="ml-auto flex shrink-0 items-center gap-1.5 rounded-control border border-line bg-ink-900 px-2 py-1 font-mono text-xs text-slate-400 hover:text-paper"
      >
        <Command size={13} />
        <span className="hidden sm:inline">K</span>
      </button>
    </header>
  );
}
