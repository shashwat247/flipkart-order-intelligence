/**
 * The signature element. Renders the LangGraph agent as a wired circuit:
 * guard -> intent -> [conditional fork: retrieve | tool | direct] -> generate.
 * The traversed path lights in --tape node by node; untraversed branches stay
 * dimmed at --line. A blocked turn clamps the rail in --flag at the fork,
 * since that's where the guardrail's routing decision actually happens.
 */
export function GraphRail({
  path,
  blocked,
  orientation = "vertical",
  onNodeClick,
}: {
  /** the real trace, e.g. ["guard_node","intent_node","retrieval_node","response_node"] */
  path: string[];
  blocked?: boolean;
  orientation?: "vertical" | "horizontal";
  onNodeClick?: (node: string) => void;
}) {
  const has = (n: string) => path.includes(n);
  const branch: "retrieve" | "tool" | "direct" | null = has("retrieval_node")
    ? "retrieve"
    : has("tool_node")
      ? "tool"
      : has("intent_node")
        ? "direct"
        : null;

  const nodeColor = (active: boolean, isBlocked?: boolean) =>
    isBlocked ? "border-flag bg-flag/15 text-flag" : active ? "border-tape bg-tape/15 text-tape" : "border-line text-slate-400";
  const lineColor = (active: boolean, isBlocked?: boolean) =>
    isBlocked ? "bg-flag" : active ? "bg-tape" : "bg-line";

  const Node = ({
    id,
    label,
    active,
    isBlocked,
  }: {
    id: string;
    label: string;
    active: boolean;
    isBlocked?: boolean;
  }) => (
    <button
      type="button"
      disabled={!onNodeClick}
      onClick={() => onNodeClick?.(id)}
      className={`flex items-center justify-center rounded-control border px-2 py-1.5 font-mono text-[11px] uppercase tracking-wide transition-colors duration-node ease-node ${nodeColor(
        active,
        isBlocked
      )} ${onNodeClick ? "cursor-pointer hover:brightness-125" : "cursor-default"}`}
    >
      {label}
    </button>
  );

  const isVertical = orientation === "vertical";

  return (
    <div className={`flex ${isVertical ? "h-full flex-col items-center gap-0 py-2" : "w-full flex-row items-center gap-0 px-2"}`}>
      <Node id="guard_node" label="guard" active={has("guard_node")} />
      <div className={isVertical ? `h-6 w-px ${lineColor(has("intent_node"))}` : `h-px w-6 ${lineColor(has("intent_node"))}`} />
      <Node id="intent_node" label="intent" active={has("intent_node")} />

      {/* the conditional fork — three branch slots, at most one lit */}
      <div className={isVertical ? `h-6 w-px ${lineColor(branch !== null, blocked)}` : `h-px w-6 ${lineColor(branch !== null, blocked)}`} />
      <div className={`flex ${isVertical ? "flex-row gap-2" : "flex-col gap-2"}`}>
        <Node id="retrieval_node" label="retrieve" active={branch === "retrieve"} />
        <Node id="tool_node" label="tool" active={branch === "tool"} />
        <Node id="direct" label={blocked ? "blocked" : "direct"} active={branch === "direct"} isBlocked={blocked && branch === "direct"} />
      </div>
      <div className={isVertical ? `h-6 w-px ${lineColor(has("response_node"))}` : `h-px w-6 ${lineColor(has("response_node"))}`} />
      <Node id="response_node" label="generate" active={has("response_node")} />
    </div>
  );
}
