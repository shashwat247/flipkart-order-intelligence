import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, ArrowUp, Image as ImageIcon, RotateCcw, Sparkles } from "lucide-react";
import { ToolResultCard } from "../../components/ToolResultCard";
import { EvidencePanel } from "../../components/EvidencePanel";
import { AnswerText } from "../../components/AnswerText";
import {
  fetchSamples,
  fetchStatus,
  resetConversation,
  sendMessage,
  type ComponentStatus,
  type Turn,
} from "../../lib/agent";

interface UserMessage {
  role: "user";
  text: string;
  image: string | null;
}
interface AgentMessage {
  role: "agent";
  turn: Turn;
}
interface ErrorMessage {
  role: "error";
  text: string;
}
type Message = UserMessage | AgentMessage | ErrorMessage;

/** Suggestions, not a menu. They only prefill the composer — the text still
 * goes through the same /api/chat call any typed question does, so nothing is
 * answered from a lookup table. */
const SUGGESTIONS = [
  "Can I send these sneakers back?",
  "How long do I have to wait for a COD refund?",
  "I bought running shoes for ₹4,500 using COD — what's the return risk?",
  "What category is this product photo?",
];

export default function Assistant() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [image, setImage] = useState("");
  const [samples, setSamples] = useState<string[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [status, setStatus] = useState<{ components: ComponentStatus[]; ready: number; total: number; mode: string } | null>(null);
  const [offline, setOffline] = useState(false);

  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    fetchStatus().then(setStatus).catch(() => setOffline(true));
    fetchSamples().then(setSamples).catch(() => undefined);
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, pending]);

  const send = useCallback(
    async (text: string, attached: string) => {
      const trimmed = text.trim();
      if (!trimmed || pending) return;

      setMessages((m) => [...m, { role: "user", text: trimmed, image: attached || null }]);
      setDraft("");
      setImage("");
      setPending(true);

      try {
        const turn = await sendMessage(trimmed, conversationId, attached || null);
        setConversationId(turn.conversation_id);
        setOffline(false);
        setMessages((m) => [...m, { role: "agent", turn }]);
      } catch (err) {
        setMessages((m) => [
          ...m,
          {
            role: "error",
            text:
              err instanceof Error && err.message.startsWith("Failed to fetch")
                ? "Can't reach the agent. Start the backend with: python3 -m backend.api"
                : `The agent returned an error — ${err instanceof Error ? err.message : String(err)}`,
          },
        ]);
      } finally {
        setPending(false);
        inputRef.current?.focus();
      }
    },
    [conversationId, pending],
  );

  async function startNewConversation() {
    try {
      const { conversation_id } = await resetConversation();
      setConversationId(conversation_id);
    } catch {
      setConversationId(null);
    }
    setMessages([]);
    setDraft("");
    setImage("");
    inputRef.current?.focus();
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends; Shift+Enter is a newline. Matches every chat product a
    // support agent would already know how to use.
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send(draft, image);
    }
  }

  const lastOrderId = [...messages].reverse().find((m): m is AgentMessage => m.role === "agent")?.turn.order_id ?? null;
  const turnCount = messages.filter((m) => m.role === "agent").length;

  return (
    <div className="mx-auto flex h-full max-w-4xl flex-col">
      {/* ------------------------------------------------------------ header */}
      <header className="mb-4 flex flex-wrap items-start justify-between gap-3 border-b border-line pb-4">
        <div className="min-w-0">
          <h1 className="font-display text-2xl font-bold tracking-tight text-paper">Support Assistant</h1>
          <p className="font-body text-sm text-slate-400">
            Ask anything in your own words. Every answer is produced by the real LangGraph agent.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {status && (
            <span
              className="inline-flex items-center gap-1.5 rounded-control border border-line px-2.5 py-1 font-mono text-xs"
              title={status.components.map((c) => `${c.name}: ${c.detail}`).join("\n")}
            >
              <span
                className={`h-1.5 w-1.5 rounded-full ${status.ready === status.total ? "bg-verdant" : "bg-tape"}`}
                aria-hidden
              />
              <span className="text-slate-400">
                {status.ready}/{status.total} ready · {status.mode}
              </span>
            </span>
          )}
          <button
            type="button"
            onClick={startNewConversation}
            className="inline-flex items-center gap-1.5 rounded-control border border-line px-2.5 py-1 font-body text-xs text-paper transition-colors duration-node hover:bg-ink-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-signal"
          >
            <RotateCcw size={12} aria-hidden />
            New conversation
          </button>
        </div>
      </header>

      {/* --------------------------------------------------- conversation state */}
      {(turnCount > 0 || lastOrderId) && (
        <p className="mb-3 font-mono text-xs text-slate-400" aria-live="polite">
          conversation {conversationId ?? "—"} · turns {turnCount}
          {lastOrderId !== null && <> · remembering order {lastOrderId}</>}
        </p>
      )}

      {/* ---------------------------------------------------------- messages */}
      <div className="min-h-0 flex-1 overflow-y-auto pr-1">
        {offline && messages.length === 0 && (
          <div className="mb-4 flex items-start gap-2 rounded-panel border border-flag/40 bg-flag/10 p-4">
            <AlertCircle size={16} className="mt-0.5 shrink-0 text-flag" aria-hidden />
            <div>
              <p className="font-body text-sm text-paper">The agent backend isn’t running.</p>
              <code className="mt-1 block font-mono text-xs text-slate-400">python3 -m backend.api</code>
            </div>
          </div>
        )}

        {messages.length === 0 && !offline && (
          <div className="flex flex-col items-center justify-center py-10 text-center">
            <Sparkles size={22} className="mb-3 text-signal" aria-hidden />
            <p className="mb-1 font-display text-lg font-bold text-paper">Ask about an order</p>
            <p className="mb-5 max-w-md font-body text-sm text-slate-400">
              Return and refund policies, how likely an order is to come back, or what category a
              product photo shows. Type anything — these are only examples.
            </p>
            <ul className="flex flex-wrap justify-center gap-2">
              {SUGGESTIONS.map((s) => (
                <li key={s}>
                  <button
                    type="button"
                    onClick={() => {
                      setDraft(s);
                      inputRef.current?.focus();
                    }}
                    className="rounded-control border border-line px-3 py-1.5 text-left font-body text-xs text-slate-400 transition-colors duration-node hover:border-signal hover:text-paper focus:outline-none focus-visible:ring-2 focus-visible:ring-signal"
                  >
                    {s}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        <ol className="flex flex-col gap-4">
          {messages.map((m, i) => (
            <li key={i} className="animate-rise">
              {m.role === "user" && (
                <div className="flex justify-end">
                  <div className="max-w-[85%] rounded-panel bg-signal px-3.5 py-2.5 sm:max-w-[75%]">
                    <p className="whitespace-pre-wrap font-body text-sm text-white">{m.text}</p>
                    {m.image && (
                      <p className="mt-1.5 inline-flex items-center gap-1 font-mono text-xs text-white/80">
                        <ImageIcon size={11} aria-hidden />
                        {m.image}
                      </p>
                    )}
                  </div>
                </div>
              )}

              {m.role === "agent" && (
                <div className="max-w-[92%] rounded-panel border border-line bg-ink-800 px-4 py-3">
                  <AnswerText text={m.turn.response.answer} />
                  {m.turn.tool_result && <ToolResultCard tool={m.turn.tool_result} />}
                  <EvidencePanel turn={m.turn} />
                </div>
              )}

              {m.role === "error" && (
                <div className="flex items-start gap-2 rounded-panel border border-flag/40 bg-flag/10 px-4 py-3">
                  <AlertCircle size={15} className="mt-0.5 shrink-0 text-flag" aria-hidden />
                  <p className="font-body text-sm text-paper">{m.text}</p>
                </div>
              )}
            </li>
          ))}

          {pending && (
            <li className="animate-rise" aria-live="polite">
              <div className="inline-flex items-center gap-2 rounded-panel border border-line bg-ink-800 px-4 py-3">
                <span className="flex gap-1" aria-hidden>
                  <span className="h-1.5 w-1.5 animate-dot rounded-full bg-slate-400" />
                  <span className="h-1.5 w-1.5 animate-dot rounded-full bg-slate-400 [animation-delay:150ms]" />
                  <span className="h-1.5 w-1.5 animate-dot rounded-full bg-slate-400 [animation-delay:300ms]" />
                </span>
                <span className="font-mono text-xs text-slate-400">agent is working…</span>
              </div>
            </li>
          )}
        </ol>
        <div ref={endRef} />
      </div>

      {/* --------------------------------------------------------- composer */}
      <form
        className="mt-4 flex flex-col gap-2 border-t border-line pt-3"
        onSubmit={(e) => {
          e.preventDefault();
          void send(draft, image);
        }}
      >
        <div className="flex items-end gap-2">
          <label htmlFor="composer" className="sr-only">
            Ask the support assistant
          </label>
          <textarea
            id="composer"
            ref={inputRef}
            rows={1}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask about returns, refunds, an order's risk, or a product photo…"
            className="max-h-40 min-h-[42px] flex-1 resize-y rounded-control border border-line bg-ink-800 px-3 py-2.5 font-body text-sm text-paper placeholder:text-slate-400 focus:border-signal focus:outline-none focus-visible:ring-2 focus-visible:ring-signal"
          />
          <button
            type="submit"
            disabled={!draft.trim() || pending}
            aria-label="Send message"
            className="flex h-[42px] w-[42px] shrink-0 items-center justify-center rounded-control bg-signal text-white transition-opacity duration-node hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-signal"
          >
            <ArrowUp size={18} aria-hidden />
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <label htmlFor="sample" className="font-mono text-xs text-slate-400">
            Attach photo:
          </label>
          <select
            id="sample"
            value={image}
            onChange={(e) => setImage(e.target.value)}
            className="rounded-control border border-line bg-ink-800 px-2 py-1 font-mono text-xs text-paper focus:border-signal focus:outline-none focus-visible:ring-2 focus-visible:ring-signal"
          >
            <option value="">(none)</option>
            {samples.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          <span className="ml-auto font-mono text-xs text-slate-400">Enter to send · Shift+Enter for a new line</span>
        </div>
      </form>
    </div>
  );
}
