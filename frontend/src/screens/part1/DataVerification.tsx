import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { MetricTile } from "../../components/MetricTile";
import { DataTable } from "../../components/DataTable";
import { useReport } from "../../lib/reports";
import type { CategoryRate } from "../../lib/types";

export default function DataVerification() {
  const report = useReport("part1_data.json");
  if (report.status !== "ready") return <EmptyState file="part1_data.json" />;
  const d = report.data;

  const columns = [
    { key: "label", header: "value", mono: false },
    { key: "orders", header: "orders", align: "right" as const, mono: true, sortValue: (r: CategoryRate) => r.orders },
    { key: "returns", header: "returns", align: "right" as const, mono: true, sortValue: (r: CategoryRate) => r.returns },
    {
      key: "return_rate", header: "return rate", align: "right" as const, mono: true,
      sortValue: (r: CategoryRate) => r.return_rate,
      render: (r: CategoryRate) => `${(r.return_rate * 100).toFixed(2)}%`,
    },
  ];

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Data Verification" description="Task 2 — shape, rates, and the missingness mechanism.">
        <ProvenanceStrip source="orders_dataset.csv" script="python3 -m part1.verify_dataset" />
      </ScreenHeader>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricTile label="Total rows" value={String(d.rows)} hero />
        <MetricTile label="Return rate" value={`${(d.return_rate * 100).toFixed(2)}%`} />
        <MetricTile label="rating_given missing" value={`${d.rating_missing_pct.toFixed(2)}%`} />
        <MetricTile label="Columns" value={String(d.columns)} />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div>
          <h2 className="mb-2 font-body text-sm font-semibold text-paper">Return rate by product_category</h2>
          <DataTable columns={columns} rows={d.by_category} rowKey={(r) => r.label} />
        </div>
        <div>
          <h2 className="mb-2 font-body text-sm font-semibold text-paper">Return rate by payment_method</h2>
          <DataTable columns={columns} rows={d.by_payment} rowKey={(r) => r.label} />
        </div>
      </div>

      <div className="rounded-panel border border-line bg-ink-800 p-4">
        <div className="mb-2 flex items-center gap-2">
          <span className="rounded-control border border-signal bg-signal/15 px-2 py-0.5 font-mono text-xs font-semibold text-signal">
            {d.missingness.verdict}
          </span>
          <span className="font-body text-xs text-slate-400">— missingness mechanism</span>
        </div>
        <div className="mb-3 flex gap-6 font-mono text-sm">
          <span className="text-paper">COD missing rate: {(d.missingness.cod_missing_rate * 100).toFixed(2)}%</span>
          <span className="text-paper">non-COD missing rate: {(d.missingness.non_cod_missing_rate * 100).toFixed(2)}%</span>
        </div>
        <p className="font-body text-sm text-slate-400">{d.missingness.justification}</p>
      </div>
    </div>
  );
}
