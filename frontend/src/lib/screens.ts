import type { ReportRegistry } from "./types";

export type Group = "Assistant" | "Project" | "Part 1" | "Part 2" | "Part 3";

export interface ScreenMeta {
  path: string;
  group: Group;
  label: string;
  /** which report files this screen depends on — drives the nav status dot */
  reportFiles: (keyof ReportRegistry)[];
  keywords?: string[];
}

export const GROUP_WEIGHTS: Record<Group, string> = {
  Assistant: "",
  Project: "",
  "Part 1": "35",
  "Part 2": "25",
  "Part 3": "40",
};

export const SCREENS: ScreenMeta[] = [
  { path: "/assistant", group: "Assistant", label: "Support Assistant", reportFiles: [], keywords: ["chat", "ask", "agent", "support"] },

  { path: "/overview", group: "Project", label: "Overview", reportFiles: ["part1_artifact.json", "part2_artifact.json", "part3_kb.json"] },
  { path: "/run-instructions", group: "Project", label: "Run Instructions", reportFiles: [] },
  { path: "/git-provenance", group: "Project", label: "Git Provenance", reportFiles: ["project_meta.json"] },

  { path: "/p1/dataset-generator", group: "Part 1", label: "Dataset Generator", reportFiles: ["part1_data.json"] },
  { path: "/p1/data-verification", group: "Part 1", label: "Data Verification", reportFiles: ["part1_data.json"], keywords: ["MAR", "MCAR", "MNAR", "missingness"] },
  { path: "/p1/preprocessing", group: "Part 1", label: "Preprocessing", reportFiles: ["part1_data.json"] },
  { path: "/p1/baseline", group: "Part 1", label: "Baseline", reportFiles: ["part1_baseline.json"], keywords: ["DummyClassifier"] },
  { path: "/p1/threshold-sweep", group: "Part 1", label: "Threshold Sweep", reportFiles: ["part1_threshold_sweep.json"], keywords: ["logistic regression", "F1"] },
  { path: "/p1/rf-tuning", group: "Part 1", label: "Random Forest Tuning", reportFiles: ["part1_rf_grid.json"], keywords: ["GridSearchCV", "ROC-AUC"] },
  { path: "/p1/explainability", group: "Part 1", label: "Explainability", reportFiles: ["part1_importance.json"], keywords: ["impurity", "permutation importance"] },
  { path: "/p1/subgroup-analysis", group: "Part 1", label: "Subgroup Analysis", reportFiles: ["part1_subgroups.json"], keywords: ["recall gap", "root cause"] },
  { path: "/p1/saved-artifact", group: "Part 1", label: "Saved Artifact", reportFiles: ["part1_artifact.json"], keywords: ["t*_rf", "threshold_rf"] },

  { path: "/p2/dataset", group: "Part 2", label: "Dataset", reportFiles: ["part2_dataset.json"], keywords: ["Fashion-MNIST"] },
  { path: "/p2/preprocessing", group: "Part 2", label: "Preprocessing", reportFiles: ["part2_training.json"], keywords: ["ImageNet", "normalize"] },
  { path: "/p2/training", group: "Part 2", label: "Model & Training", reportFiles: ["part2_training.json"], keywords: ["ResNet-18", "backbone"] },
  { path: "/p2/finetuning", group: "Part 2", label: "Fine-tuning", reportFiles: ["part2_training.json"] },
  { path: "/p2/evaluation", group: "Part 2", label: "Evaluation", reportFiles: ["part2_eval.json"], keywords: ["accuracy", "per-class"] },
  { path: "/p2/confusion-matrix", group: "Part 2", label: "Confusion Matrix", reportFiles: ["part2_eval.json"] },
  { path: "/p2/confusion-patterns", group: "Part 2", label: "Confusion Patterns", reportFiles: ["part2_eval.json"], keywords: ["shirt", "coat", "confused"] },
  { path: "/p2/artifact-samples", group: "Part 2", label: "Artifact & Samples", reportFiles: ["part2_artifact.json"], keywords: ["sample images", "classifier"] },

  { path: "/p3/agent-console", group: "Part 3", label: "Agent Console", reportFiles: ["part3_transcripts.json"], keywords: ["chat"] },
  { path: "/p3/graph-inspector", group: "Part 3", label: "Graph Inspector", reportFiles: ["part3_transcripts.json"], keywords: ["LangGraph", "conditional edge"] },
  { path: "/p3/conversation-state", group: "Part 3", label: "Conversation State", reportFiles: ["part3_transcripts.json"], keywords: ["multi-turn", "state"] },
  { path: "/p3/knowledge-base", group: "Part 3", label: "Knowledge Base", reportFiles: ["part3_kb.json"], keywords: ["policy", "chunks", "FAISS"] },
  { path: "/p3/retrieval-explorer", group: "Part 3", label: "Retrieval Explorer", reportFiles: ["part3_retrieval_eval.json", "part3_kb.json"], keywords: ["similarity"] },
  { path: "/p3/tools", group: "Part 3", label: "Tools", reportFiles: ["part3_tools.json", "part2_artifact.json"], keywords: ["check_return_risk", "classify_product_image"] },
  { path: "/p3/prompt-design", group: "Part 3", label: "Prompt Design", reportFiles: ["part3_prompt.json"], keywords: ["4S", "few-shot", "system prompt"] },
  { path: "/p3/guardrails", group: "Part 3", label: "Guardrails", reportFiles: ["part3_guardrails.json"], keywords: ["injection", "groundedness"] },
  { path: "/p3/transcripts", group: "Part 3", label: "Transcripts", reportFiles: ["part3_transcripts.json"] },
  { path: "/p3/retrieval-evaluation", group: "Part 3", label: "Retrieval Evaluation", reportFiles: ["part3_retrieval_eval.json"], keywords: ["Precision@3", "Recall@3"] },
];

export const GROUPS: Group[] = ["Assistant", "Project", "Part 1", "Part 2", "Part 3"];
