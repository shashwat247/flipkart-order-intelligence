import { useState } from "react";
import { CodeBlock } from "./CodeBlock";

export function JsonCard({ data, title }: { data: unknown; title?: string }) {
  const [raw, setRaw] = useState(false);
  const json = JSON.stringify(data, null, 2);

  return (
    <div className="flex flex-col gap-2 rounded-panel border border-line bg-ink-800 p-3">
      <div className="flex items-center justify-between">
        {title && <span className="font-body text-xs text-slate-400">{title}</span>}
        <div className="ml-auto flex overflow-hidden rounded-control border border-line font-body text-xs">
          <button
            type="button"
            onClick={() => setRaw(false)}
            className={`px-2 py-1 ${!raw ? "bg-signal text-ink-900" : "text-slate-400 hover:text-paper"}`}
          >
            Rendered
          </button>
          <button
            type="button"
            onClick={() => setRaw(true)}
            className={`px-2 py-1 ${raw ? "bg-signal text-ink-900" : "text-slate-400 hover:text-paper"}`}
          >
            Raw
          </button>
        </div>
      </div>
      {raw ? (
        <CodeBlock code={json} language="json" />
      ) : (
        <dl className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 font-mono text-sm">
          {Object.entries(data as Record<string, unknown>).map(([k, v]) => (
            <>
              <dt key={`${k}-k`} className="text-slate-400">{k}</dt>
              <dd key={`${k}-v`} className="text-paper">
                {typeof v === "object" ? JSON.stringify(v) : String(v)}
              </dd>
            </>
          ))}
        </dl>
      )}
    </div>
  );
}
