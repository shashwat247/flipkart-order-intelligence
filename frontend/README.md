# Order Intelligence Console

A static React + Vite + TypeScript + Tailwind operator console for the
`flipkart-order-intelligence` repo. Full documentation, design-system notes
and the exact commands live in the root
[`../README.md`](../README.md#order-intelligence-console).

Quick start:

```bash
python3 ../scripts/export_reports.py   # from repo root: writes public/reports/*.json
npm install
npm run dev                             # http://localhost:5173
```

No backend, no API keys, no browser storage — every screen reads a
pre-exported JSON report and renders its own empty state (naming the exact
missing file and the command that produces it) if one isn't there.
