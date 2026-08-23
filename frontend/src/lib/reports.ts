import { useEffect, useState } from "react";
import type { ReportRegistry } from "./types";

export type ReportState<T> =
  | { status: "loading" }
  | { status: "ready"; data: T }
  | { status: "missing"; file: string }
  | { status: "malformed"; file: string; detail: string };

const cache = new Map<string, Promise<unknown>>();

function load(file: string): Promise<unknown> {
  let promise = cache.get(file);
  if (!promise) {
    promise = fetch(`/reports/${file}`).then(async (res) => {
      if (!res.ok) {
        throw new MissingReportError(file);
      }
      try {
        return await res.json();
      } catch (err) {
        throw new MalformedReportError(file, err instanceof Error ? err.message : String(err));
      }
    });
    cache.set(file, promise);
  }
  return promise;
}

class MissingReportError extends Error {
  file: string;
  constructor(file: string) {
    super(`missing report: ${file}`);
    this.file = file;
  }
}
class MalformedReportError extends Error {
  file: string;
  detail: string;
  constructor(file: string, detail: string) {
    super(`malformed report: ${file}`);
    this.file = file;
    this.detail = detail;
  }
}

/**
 * Load one JSON report contract. Never throws into the caller — every failure
 * mode (missing file, malformed JSON) resolves to a typed state so a screen
 * can render EmptyState uniformly instead of crashing or showing a fabricated
 * number. This is the ONLY way a screen may read `/reports/*`.
 */
export function useReport<K extends keyof ReportRegistry>(
  file: K
): ReportState<ReportRegistry[K]> {
  const [state, setState] = useState<ReportState<ReportRegistry[K]>>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading" });
    load(file)
      .then((data) => {
        if (!cancelled) setState({ status: "ready", data: data as ReportRegistry[K] });
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof MissingReportError) {
          setState({ status: "missing", file: err.file });
        } else if (err instanceof MalformedReportError) {
          setState({ status: "malformed", file: err.file, detail: err.detail });
        } else {
          setState({ status: "malformed", file, detail: String(err) });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [file]);

  return state;
}

/** The shell command that regenerates every report — shown in every EmptyState. */
export const EXPORT_COMMAND = "python3 scripts/export_reports.py";

export type NavStatus = "ready" | "partial" | "missing";

/** Aggregate status for a nav item that depends on several report files —
 * green only when every one of them parsed, hollow when none did. */
export function useReportsStatus(files: (keyof ReportRegistry)[]): NavStatus {
  const [status, setStatus] = useState<NavStatus>("missing");

  useEffect(() => {
    if (files.length === 0) {
      setStatus("ready");
      return;
    }
    let cancelled = false;
    Promise.allSettled(files.map((f) => load(f))).then((results) => {
      if (cancelled) return;
      const ok = results.filter((r) => r.status === "fulfilled").length;
      setStatus(ok === files.length ? "ready" : ok === 0 ? "missing" : "partial");
    });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [files.join(",")]);

  return status;
}
