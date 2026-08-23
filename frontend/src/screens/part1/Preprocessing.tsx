import { ArrowRight } from "lucide-react";
import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";

const NUMERIC_FEATURES = [
  "price_inr", "discount_pct", "customer_tenure_days", "num_previous_orders",
  "num_previous_returns", "delivery_distance_km", "delivery_days",
  "is_weekend_order", "rating_given",
];
const CATEGORICAL_FEATURES = ["product_category", "payment_method"];

function Node({ title, columns }: { title: string; columns: string[] }) {
  return (
    <div className="flex flex-col gap-2 rounded-panel border border-line bg-ink-800 p-3">
      <span className="font-body text-sm font-semibold text-paper">{title}</span>
      <div className="flex flex-wrap gap-1">
        {columns.map((c) => (
          <span key={c} className="rounded-control border border-line bg-ink-900 px-1.5 py-0.5 font-mono text-[11px] text-slate-400">
            {c}
          </span>
        ))}
      </div>
    </div>
  );
}

export default function Preprocessing() {
  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Preprocessing" description="Task 3 — the leakage-free ColumnTransformer + Pipeline.">
        <ProvenanceStrip source="part1/common.py::build_preprocessor" />
      </ScreenHeader>

      <div className="inline-flex w-fit items-center gap-1.5 rounded-control border border-verdant bg-verdant/10 px-3 py-1.5 font-mono text-xs text-verdant">
        fit on train split only
      </div>

      <div className="flex flex-col gap-3">
        <span className="font-body text-xs uppercase tracking-wide text-slate-400">numeric ({NUMERIC_FEATURES.length} columns)</span>
        <div className="flex flex-wrap items-center gap-2">
          <Node title="SimpleImputer(strategy='median')" columns={NUMERIC_FEATURES} />
          <ArrowRight size={16} className="shrink-0 text-slate-400" />
          <Node title="StandardScaler()" columns={NUMERIC_FEATURES} />
        </div>
      </div>

      <div className="flex flex-col gap-3">
        <span className="font-body text-xs uppercase tracking-wide text-slate-400">categorical ({CATEGORICAL_FEATURES.length} columns)</span>
        <div className="flex flex-wrap items-center gap-2">
          <Node title="SimpleImputer(strategy='most_frequent')" columns={CATEGORICAL_FEATURES} />
          <ArrowRight size={16} className="shrink-0 text-slate-400" />
          <Node title="OneHotEncoder(handle_unknown='ignore')" columns={CATEGORICAL_FEATURES} />
        </div>
      </div>

      <p className="max-w-2xl font-body text-sm text-slate-400">
        `.fit()` is only ever called on the training split — the imputer's medians/modes and
        the scaler's mean/std are learned inside the pipeline, so the test split is only ever
        `.transform()`-ed and no test statistic can leak backwards into training.
        `handle_unknown="ignore"` means an unseen category at inference time yields an
        all-zero one-hot block instead of raising, which matters because Part 3's agent
        feeds arbitrary order dictionaries into this same pipeline.
      </p>
    </div>
  );
}
