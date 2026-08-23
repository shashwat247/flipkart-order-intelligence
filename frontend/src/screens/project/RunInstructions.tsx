import { ScreenHeader } from "../../components/ScreenHeader";
import { CodeBlock } from "../../components/CodeBlock";

function Card({ title, commands }: { title: string; commands: string }) {
  return (
    <div className="flex flex-col gap-2 rounded-panel border border-line bg-ink-800 p-4">
      <h2 className="font-body text-sm font-semibold text-paper">{title}</h2>
      <CodeBlock code={commands} />
    </div>
  );
}

export default function RunInstructions() {
  return (
    <div className="flex flex-col gap-6">
      <ScreenHeader title="Run Instructions" description="The README's regenerate/train/run commands, copyable." />

      <Card
        title="Mock-mode run (default — no API key)"
        commands={"python3 -m part3.agent --demo\npython3 -m part3.agent --ask \"How many days do I have to return a mobile phone?\""}
      />
      <Card
        title="Part 1 — Return-risk pipeline"
        commands={[
          "python3 generate_orders.py",
          "python3 -m part1.verify_dataset",
          "python3 -m part1.train_return_risk",
          "python3 -m part1.feature_analysis",
          "python3 -m part1.subgroup_analysis",
          "python3 -m part1.evaluate_return_risk",
        ].join("\n")}
      />
      <Card
        title="Part 2 — Product image categoriser"
        commands={[
          "python3 -m part2.cache_features",
          "python3 -m part2.train_product_classifier",
          "python3 -m part2.evaluate_product_classifier",
          "python3 -m part2.export_samples",
        ].join("\n")}
      />
      <Card
        title="Part 3 — Support agent"
        commands={[
          "python3 -m part3.build_index",
          "python3 -m part3.calibrate_threshold",
          "python3 -m part3.evaluate_retrieval",
          "python3 -m part3.run_transcripts",
        ].join("\n")}
      />
      <Card title="Verify everything" commands={"pytest\npython3 validate_project.py"} />
      <Card
        title="Backend API (required by the Support Assistant)"
        commands={"python3 -m backend.api                       # http://127.0.0.1:8000"}
      />
      <Card
        title="This console"
        commands={"python3 scripts/export_reports.py\ncd frontend\nnpm install\nnpm run dev              # http://localhost:5173"}
      />
      <Card title="Streamlit app (secondary UI)" commands={"streamlit run streamlit_app/app.py           # http://localhost:8501"} />
    </div>
  );
}
