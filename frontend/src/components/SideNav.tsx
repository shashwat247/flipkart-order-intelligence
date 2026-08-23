import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { GROUPS, GROUP_WEIGHTS, SCREENS, type Group } from "../lib/screens";
import { useReportsStatus } from "../lib/reports";

function StatusDot({ status }: { status: "ready" | "partial" | "missing" }) {
  return (
    <span
      className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full ${
        status === "ready" ? "bg-verdant" : status === "partial" ? "bg-tape" : "border border-slate-400 bg-transparent"
      }`}
      aria-label={status}
    />
  );
}

function NavItem({ path, label, reportFiles, active, onNavigate }: {
  path: string; label: string; reportFiles: string[]; active: boolean; onNavigate: (p: string) => void;
}) {
  const status = useReportsStatus(reportFiles as never);
  return (
    <button
      type="button"
      onClick={() => onNavigate(path)}
      className={`flex w-full items-center gap-2 rounded-control px-2 py-1.5 text-left font-body text-sm transition-colors duration-node ease-node ${
        active ? "bg-signal/15 text-signal" : "text-slate-400 hover:bg-ink-700 hover:text-paper"
      }`}
    >
      <StatusDot status={status} />
      <span className="truncate">{label}</span>
    </button>
  );
}

export function SideNav({ route, onNavigate }: { route: string; onNavigate: (p: string) => void }) {
  const [collapsed, setCollapsed] = useState<Record<Group, boolean>>({
    Assistant: false, Project: false, "Part 1": false, "Part 2": false, "Part 3": false,
  });

  return (
    <nav className="flex h-full w-64 shrink-0 flex-col gap-4 overflow-y-auto border-r border-line bg-ink-800 p-3">
      {GROUPS.map((group) => {
        const screens = SCREENS.filter((s) => s.group === group);
        const isCollapsed = collapsed[group];
        return (
          <div key={group}>
            <button
              type="button"
              onClick={() => setCollapsed((c) => ({ ...c, [group]: !c[group] }))}
              className="flex w-full items-center justify-between px-2 py-1 font-body text-xs font-semibold uppercase tracking-wide text-slate-400 hover:text-paper"
            >
              <span className="flex items-center gap-1.5">
                {isCollapsed ? <ChevronRight size={12} /> : <ChevronDown size={12} />}
                {group}
              </span>
              {GROUP_WEIGHTS[group] && <span className="font-mono text-[10px] text-slate-400">{GROUP_WEIGHTS[group]}</span>}
            </button>
            {!isCollapsed && (
              <div className="mt-1 flex flex-col gap-0.5">
                {screens.map((s) => (
                  <NavItem
                    key={s.path}
                    path={s.path}
                    label={s.label}
                    reportFiles={s.reportFiles}
                    active={route === s.path}
                    onNavigate={onNavigate}
                  />
                ))}
              </div>
            )}
          </div>
        );
      })}
    </nav>
  );
}
