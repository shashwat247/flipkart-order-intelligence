import { useState } from "react";
import { Check, Copy } from "lucide-react";

export function CodeBlock({ code, language }: { code: string; language?: string }) {
  const [copied, setCopied] = useState(false);

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard API unavailable — the code is still visible and selectable
    }
  };

  return (
    <div className="group relative w-full max-w-full overflow-x-auto rounded-control border border-line bg-ink-900">
      <pre className="whitespace-pre px-3 py-2 font-mono text-[13px] text-paper">
        <code data-language={language}>{code}</code>
      </pre>
      <button
        type="button"
        onClick={onCopy}
        aria-label="Copy to clipboard"
        className="absolute right-2 top-2 rounded-control border border-line bg-ink-800 p-1 text-slate-400 opacity-0 transition-opacity duration-node ease-node hover:text-paper focus-visible:opacity-100 group-hover:opacity-100"
      >
        {copied ? <Check size={14} className="text-verdant" /> : <Copy size={14} />}
      </button>
    </div>
  );
}
