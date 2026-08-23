import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ScreenHeader } from "../../components/ScreenHeader";
import { ProvenanceStrip } from "../../components/ProvenanceStrip";
import { EmptyState } from "../../components/EmptyState";
import { useReport } from "../../lib/reports";

export default function Training() {
  const report = useReport("part2_training.json");
  if (report.status !== "ready") return <EmptyState file="part2_training.json" />;
  const t = report.data;

  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Model & Training" description={`${t.backbone} — ${t.strategy}.`}>
        <ProvenanceStrip source="reports/part2_training_log.json" script="python3 -m part2.train_product_classifier" />
      </ScreenHeader>

      <div className="flex items-center gap-1 overflow-x-auto rounded-panel border border-line bg-ink-800 p-3">
        {["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"].map((l) => (
          <span key={l} className="shrink-0 rounded-control border border-line bg-ink-900 px-2 py-1 font-mono text-xs text-slate-400">
            {l} (frozen)
          </span>
        ))}
        <span className="shrink-0 rounded-control border border-tape bg-tape/15 px-2 py-1 font-mono text-xs text-tape">
          fc: Linear(512, 10) — trainable
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          ["optimizer", t.optimizer],
          ["learning rate", String(t.head_learning_rate)],
          ["batch size", String(t.head_batch_size)],
          ["epochs", String(t.head_epochs)],
        ].map(([label, value]) => (
          <div key={label} className="rounded-control border border-line bg-ink-800 px-3 py-2">
            <div className="font-body text-[11px] text-slate-400">{label}</div>
            <div className="font-mono text-sm text-paper">{value}</div>
          </div>
        ))}
      </div>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={t.head_history} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#26324A" strokeDasharray="2 4" />
            <XAxis dataKey="epoch" stroke="#8A99B5" tick={{ fill: "#8A99B5", fontFamily: "IBM Plex Mono", fontSize: 11 }} />
            <YAxis stroke="#8A99B5" tick={{ fill: "#8A99B5", fontFamily: "IBM Plex Mono", fontSize: 11 }} domain={[0, 1]} />
            <Tooltip contentStyle={{ background: "#101827", border: "1px solid #26324A", borderRadius: 8 }} labelStyle={{ color: "#E9EEF7" }} />
            <Line type="monotone" dataKey="val_accuracy" name="val accuracy" stroke="#2E7CF6" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="train_loss" name="train loss" stroke="#FFB43D" dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <p className="max-w-2xl font-body text-sm text-slate-400">{t.feature_caching}.</p>
    </div>
  );
}
