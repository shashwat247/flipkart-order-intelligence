export function ScreenHeader({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children?: React.ReactNode; // a ProvenanceStrip, typically
}) {
  return (
    <div className="mb-5 flex flex-col gap-1.5 border-b border-line pb-4">
      <h1 className="font-display text-2xl font-bold tracking-tight text-paper">{title}</h1>
      <p className="font-body text-sm text-slate-400">{description}</p>
      {children}
    </div>
  );
}
