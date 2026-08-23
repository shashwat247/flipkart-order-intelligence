import { useEffect, useMemo, useRef, useState } from "react";
import { Search } from "lucide-react";
import { SCREENS } from "../lib/screens";

/** In-house fuzzy match over a static registry of screens (extended with
 * each screen's own keywords) — no external fuzzy-search dependency. */
function score(query: string, target: string): number {
  const q = query.toLowerCase();
  const t = target.toLowerCase();
  if (!q) return 0;
  if (t.includes(q)) return 100 - t.indexOf(q);
  let qi = 0;
  let matched = 0;
  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] === q[qi]) {
      qi++;
      matched++;
    }
  }
  return qi === q.length ? matched : -1;
}

export function CommandPalette({
  open,
  onClose,
  onNavigate,
}: {
  open: boolean;
  onClose: () => void;
  onNavigate: (path: string) => void;
}) {
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) {
      setQuery("");
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [open]);

  const results = useMemo(() => {
    if (!query) return SCREENS;
    return SCREENS.map((s) => {
      const haystack = [s.label, s.group, ...(s.keywords ?? [])].join(" ");
      return { screen: s, s: score(query, haystack) };
    })
      .filter((r) => r.s >= 0)
      .sort((a, b) => b.s - a.s)
      .map((r) => r.screen);
  }, [query]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-ink-900/70 pt-24"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg rounded-panel border border-line bg-ink-800 shadow-none"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 border-b border-line px-3 py-2">
          <Search size={16} className="text-slate-400" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Escape") onClose();
              if (e.key === "Enter" && results[0]) {
                onNavigate(results[0].path);
                onClose();
              }
            }}
            placeholder="Search screens, metrics, tools, documents…"
            className="w-full bg-transparent font-body text-sm text-paper outline-none placeholder:text-slate-400"
          />
          <kbd className="rounded-control border border-line px-1.5 py-0.5 font-mono text-[10px] text-slate-400">esc</kbd>
        </div>
        <div className="max-h-80 overflow-y-auto p-1.5">
          {results.length === 0 && (
            <p className="px-2 py-3 font-body text-sm text-slate-400">No matches.</p>
          )}
          {results.map((s) => (
            <button
              key={s.path}
              type="button"
              onClick={() => {
                onNavigate(s.path);
                onClose();
              }}
              className="flex w-full items-center justify-between rounded-control px-2 py-1.5 text-left hover:bg-ink-700"
            >
              <span className="font-body text-sm text-paper">{s.label}</span>
              <span className="font-mono text-[11px] text-slate-400">{s.group}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
