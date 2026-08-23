// Every type below mirrors exactly what scripts/export_reports.py writes.
// No field here may be invented — if export_reports.py doesn't produce it,
// it doesn't belong in these types.

export interface CategoryRate {
  label: string;
  orders: number;
  returns: number;
  return_rate: number;
}

export interface Part1Data {
  rows: number;
  columns: number;
  column_names: string[];
  return_rate: number;
  rating_missing_pct: number;
  generator: {
    seed: number;
    n_rows: number;
    category_probs: { category: string; probability: number }[];
    payment_probs: { payment_method: string; probability: number }[];
  };
  by_category: CategoryRate[];
  by_payment: CategoryRate[];
  missingness: {
    verdict: "MCAR" | "MAR" | "MNAR";
    cod_missing_rate: number;
    non_cod_missing_rate: number;
    justification: string;
  };
}

export interface Part1Baseline {
  accuracy: number;
  f1_positive: number;
  note: string;
}

export interface ThresholdPoint {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  accuracy: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface Part1ThresholdSweep {
  points: ThresholdPoint[];
  best_threshold: number;
  threshold_rf: number;
  tradeoff_paragraph: string;
}

export interface RfGridCell {
  n_estimators: number;
  max_depth: number | null;
  cv_roc_auc: number;
  cv_roc_auc_std: number;
}

export interface Part1RfGrid {
  cells: RfGridCell[];
  best_params: { n_estimators: number; max_depth: number | null };
  best_cv_roc_auc: number;
  test_roc_auc: number;
  auc_gap: number;
}

export interface ImportanceRow {
  feature: string;
  value: number;
  rank: number;
}

export interface Part1Importance {
  impurity: ImportanceRow[];
  permutation: ImportanceRow[];
  biggest_drop: string;
  explanation: string;
}

export interface SubgroupRow {
  subgroup: string;
  n_test: number;
  actual_returns: number;
  flagged: number;
  tp: number;
  fp: number;
  fn: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface Part1Subgroups {
  by_category: SubgroupRow[];
  by_payment: SubgroupRow[];
  overall: { precision: number; recall: number; f1: number; threshold_rf: number };
  weakest: { by: string; subgroup: string; recall: number; recall_gap: number; material: boolean };
  proposed_fix: string;
}

export interface Part1Artifact {
  path: string;
  loads_ok: boolean;
  t_star_rf: number;
  buckets: { low_max: number; medium_max: number };
  model_type: string;
  best_params: { n_estimators: number; max_depth: number | null };
  test_roc_auc: number;
  justification_sentence: string;
}

export interface Part2ClassChip {
  class_name: string;
  index: number;
  train_pool_count: number;
  head_train_count: number;
  val_count: number;
  test_count: number;
}

export interface Part2Dataset {
  class_names: string[];
  classes: Part2ClassChip[];
  split_sizes: { train: number; val: number; test: number };
  source: string;
  test_untouched_note: string;
}

export interface HeadHistoryEpoch {
  epoch: number;
  train_loss: number;
  val_accuracy: number;
  seconds: number;
}

export interface Part2Training {
  device: string;
  backbone: string;
  strategy: string;
  feature_caching: string;
  optimizer: string;
  head_learning_rate: number;
  head_batch_size: number;
  head_epochs: number;
  head_history: HeadHistoryEpoch[];
  finetune_triggered: boolean;
  finetune_trigger_threshold: number;
  val_accuracy_before_finetuning: number;
  val_accuracy_after_finetuning: number;
  finetune_history: HeadHistoryEpoch[];
  total_parameters: number;
  input_size: [number, number];
  normalization: { mean: number[]; std: number[] };
  channel_handling: string;
}

export interface PerClassMetric {
  class: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ConfusionPair {
  true_class: string;
  predicted_class: string;
  count: number;
}

export interface PairExplanation {
  class_a: string;
  class_b: string;
  total_misclassifications: number;
  read_off: string;
  explanation: string;
}

export interface Part2Eval {
  test_accuracy: number;
  reference_accuracy: number;
  class_names: string[];
  confusion_matrix: number[][];
  per_class: PerClassMetric[];
  confusion_pairs: ConfusionPair[];
  pair_explanations: PairExplanation[];
}

export interface SampleImage {
  file: string;
  true_label: string;
  true_index: number;
  source: string;
  test_split_index: number;
  predicted_class: string;
  confidence: number;
  top3: { label: string; probability: number }[];
  agrees_with_true_label: boolean;
}

export interface Part2Artifact {
  path: string;
  loads_ok: boolean;
  architecture: string;
  head: string;
  load_snippet: string;
  sample_images: SampleImage[];
}

export interface KbChunk {
  chunk_id: string;
  document_id: string;
  document_title: string;
  chunk_text: string;
}

export interface KbDocument {
  id: string;
  title: string;
  text: string;
  chunks: KbChunk[];
  n_chunks: number;
}

export interface Part3Kb {
  documents: KbDocument[];
  n_documents: number;
  n_chunks: number;
  embedding_model: string;
  index_backend: string;
  badge: string;
}

export interface RetrievedChunk {
  document_id: string;
  document_title: string;
  chunk_text: string;
  score: number;
}

export interface RetrievalEvalQuery {
  query: string;
  relevant: string[];
  retrieved: string[];
  hits: string[];
  n_hits: number;
  precision_at_3: number;
  recall_at_3: number;
  rationale: string;
  scores: number[];
  retrieved_chunks: RetrievedChunk[];
}

export interface Part3RetrievalEval {
  queries: RetrievalEvalQuery[];
  mean_precision_at_3: number;
  mean_recall_at_3: number;
  k: number;
  top_k_chunks: number;
  similarity_threshold: number;
}

export interface TranscriptTurn {
  turn_label: string;
  user: string | null;
  graph_path: string[];
  blocked: boolean;
  response: { answer: string; source: string; confidence: number } | null;
  state_after: Record<string, string>;
}

export interface Transcript {
  filename: string;
  header: Record<string, string>;
  turns: TranscriptTurn[];
}

export interface Part3Transcripts {
  transcripts: Transcript[];
}

export interface FewShotExampleLive {
  user: string;
  fine_intent: string;
  live_lane: string;
}

export interface Part3Prompt {
  system_prompt: string;
  principle_annotations: Record<string, string>;
  few_shot_examples: FewShotExampleLive[];
}

export interface InjectionPattern {
  name: string;
  pattern: string;
}

export interface InjectionExample {
  text: string;
  blocked: boolean;
  matches: { pattern: string; matched_text: string }[];
  n_patterns_checked: number;
}

export interface Part3Guardrails {
  injection_patterns: InjectionPattern[];
  injection_examples: InjectionExample[];
  similarity_threshold: number;
  ungrounded_example: { query: string; best_score: number; grounded: boolean };
  out_of_domain_queries: string[];
}

export interface ReturnRiskExample {
  order_id: number;
  features: Record<string, string | number>;
  return_probability: number;
  actual_returned: boolean;
}

export interface Part3Tools {
  return_risk_examples: ReturnRiskExample[];
  threshold_rf: number;
  function_signature: string;
  artifact_path: string;
  image_function_signature: string;
  image_artifact_path: string;
}

export interface GitCommit {
  hash: string;
  short_hash: string;
  author: string;
  date: string;
  subject: string;
  parents: string[];
}

export interface ProjectMeta {
  repo: string;
  commit: string;
  short_commit: string;
  branch: string;
  commits: GitCommit[];
}

// Registry mapping every contract filename to its type — reports.ts's loader
// is generic over this so a screen gets full type inference from the filename.
export interface ReportRegistry {
  "part1_data.json": Part1Data;
  "part1_baseline.json": Part1Baseline;
  "part1_threshold_sweep.json": Part1ThresholdSweep;
  "part1_rf_grid.json": Part1RfGrid;
  "part1_importance.json": Part1Importance;
  "part1_subgroups.json": Part1Subgroups;
  "part1_artifact.json": Part1Artifact;
  "part2_dataset.json": Part2Dataset;
  "part2_training.json": Part2Training;
  "part2_eval.json": Part2Eval;
  "part2_artifact.json": Part2Artifact;
  "part3_kb.json": Part3Kb;
  "part3_retrieval_eval.json": Part3RetrievalEval;
  "part3_transcripts.json": Part3Transcripts;
  "part3_prompt.json": Part3Prompt;
  "part3_guardrails.json": Part3Guardrails;
  "part3_tools.json": Part3Tools;
  "project_meta.json": ProjectMeta;
}
