import { Suspense, lazy, useEffect, useState } from "react";
import { useHashRoute } from "./lib/router";
import { SCREENS } from "./lib/screens";
import { SideNav } from "./components/SideNav";
import { TopBar } from "./components/TopBar";
import { CommandPalette } from "./components/CommandPalette";

const SCREEN_COMPONENTS: Record<string, ReturnType<typeof lazy>> = {
  "/assistant": lazy(() => import("./screens/assistant/Assistant")),

  "/overview": lazy(() => import("./screens/project/Overview")),
  "/run-instructions": lazy(() => import("./screens/project/RunInstructions")),
  "/git-provenance": lazy(() => import("./screens/project/GitProvenance")),

  "/p1/dataset-generator": lazy(() => import("./screens/part1/DatasetGenerator")),
  "/p1/data-verification": lazy(() => import("./screens/part1/DataVerification")),
  "/p1/preprocessing": lazy(() => import("./screens/part1/Preprocessing")),
  "/p1/baseline": lazy(() => import("./screens/part1/Baseline")),
  "/p1/threshold-sweep": lazy(() => import("./screens/part1/ThresholdSweep")),
  "/p1/rf-tuning": lazy(() => import("./screens/part1/RfTuning")),
  "/p1/explainability": lazy(() => import("./screens/part1/Explainability")),
  "/p1/subgroup-analysis": lazy(() => import("./screens/part1/SubgroupAnalysis")),
  "/p1/saved-artifact": lazy(() => import("./screens/part1/SavedArtifact")),

  "/p2/dataset": lazy(() => import("./screens/part2/Dataset")),
  "/p2/preprocessing": lazy(() => import("./screens/part2/Preprocessing")),
  "/p2/training": lazy(() => import("./screens/part2/Training")),
  "/p2/finetuning": lazy(() => import("./screens/part2/Finetuning")),
  "/p2/evaluation": lazy(() => import("./screens/part2/Evaluation")),
  "/p2/confusion-matrix": lazy(() => import("./screens/part2/ConfusionMatrix")),
  "/p2/confusion-patterns": lazy(() => import("./screens/part2/ConfusionPatterns")),
  "/p2/artifact-samples": lazy(() => import("./screens/part2/ArtifactSamples")),

  "/p3/agent-console": lazy(() => import("./screens/part3/AgentConsole")),
  "/p3/graph-inspector": lazy(() => import("./screens/part3/GraphInspector")),
  "/p3/conversation-state": lazy(() => import("./screens/part3/ConversationState")),
  "/p3/knowledge-base": lazy(() => import("./screens/part3/KnowledgeBase")),
  "/p3/retrieval-explorer": lazy(() => import("./screens/part3/RetrievalExplorer")),
  "/p3/tools": lazy(() => import("./screens/part3/Tools")),
  "/p3/prompt-design": lazy(() => import("./screens/part3/PromptDesign")),
  "/p3/guardrails": lazy(() => import("./screens/part3/Guardrails")),
  "/p3/transcripts": lazy(() => import("./screens/part3/Transcripts")),
  "/p3/retrieval-evaluation": lazy(() => import("./screens/part3/RetrievalEvaluation")),
};

export default function App() {
  const [route, navigate] = useHashRoute();
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  const navigateAndCloseMobileNav = (path: string) => {
    navigate(path);
    setMobileNavOpen(false);
  };

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const Screen = SCREEN_COMPONENTS[route];
  const known = SCREENS.some((s) => s.path === route);

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-ink-900 text-paper">
      <TopBar onOpenPalette={() => setPaletteOpen(true)} onToggleMobileNav={() => setMobileNavOpen((v) => !v)} />
      <div className="relative flex min-h-0 flex-1">
        {/* Desktop: a permanent 256px rail. Below lg: an off-canvas drawer,
            so the console stays usable down to 375px instead of squeezing
            everything into a sliver of remaining width. */}
        <div className="hidden lg:block">
          <SideNav route={route} onNavigate={navigate} />
        </div>
        {mobileNavOpen && (
          <div className="fixed inset-0 z-40 flex lg:hidden">
            <div className="absolute inset-0 bg-ink-900/70" onClick={() => setMobileNavOpen(false)} />
            <div className="relative z-10">
              <SideNav route={route} onNavigate={navigateAndCloseMobileNav} />
            </div>
          </div>
        )}
        <main
          className={
            route === "/assistant"
              ? "flex min-w-0 flex-1 flex-col overflow-hidden p-4 sm:p-6"
              : "min-w-0 flex-1 overflow-y-auto p-4 sm:p-6"
          }
        >
          <Suspense fallback={<div className="font-mono text-sm text-slate-400">Loading…</div>}>
            {known && Screen ? <Screen /> : <NotFound onNavigate={navigate} />}
          </Suspense>
        </main>
      </div>
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} onNavigate={navigate} />
    </div>
  );
}

function NotFound({ onNavigate }: { onNavigate: (p: string) => void }) {
  return (
    <div className="flex flex-col gap-2">
      <p className="font-body text-sm text-slate-400">Unknown screen.</p>
      <button
        type="button"
        onClick={() => onNavigate("/overview")}
        className="w-fit rounded-control border border-line px-3 py-1.5 font-body text-sm text-signal hover:bg-ink-700"
      >
        Back to Overview
      </button>
    </div>
  );
}
