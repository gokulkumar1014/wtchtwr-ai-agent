import { useMemo, useState } from "react";
import type { ThinkingStep } from "@/store/useChat";

interface ThinkingTraceProps {
  steps: ThinkingStep[];
}

const formatMetaValue = (value: unknown): string => {
  if (Array.isArray(value)) {
    return value.map(formatMetaValue).join(", ");
  }
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "object") {
    return Object.values(value as Record<string, unknown>)
      .map(formatMetaValue)
      .join(", ");
  }
  return String(value);
};

const summarizeMeta = (meta?: Record<string, unknown>): string | null => {
  if (!meta) return null;
  const parts = Object.entries(meta)
    .map(([key, value]) => {
      const formatted = formatMetaValue(value);
      return formatted ? `${key}: ${formatted}` : null;
    })
    .filter(Boolean) as string[];
  return parts.length ? parts.join(" • ") : null;
};

const formatDuration = (ms?: number | null): string => {
  if (!ms || ms <= 0) {
    return "—";
  }
  if (ms < 1000) {
    return `${Math.round(ms)}ms`;
  }
  return `${(ms / 1000).toFixed(1)}s`;
};

export const ThinkingTrace: React.FC<ThinkingTraceProps> = ({ steps }) => {
  const [expanded, setExpanded] = useState(false);
  const totalMs = useMemo(() => {
    const candidates = steps
      .map((step) => (typeof step.elapsed_ms === "number" ? step.elapsed_ms : null))
      .filter((value): value is number => value !== null);
    return candidates.length ? Math.max(...candidates) : null;
  }, [steps]);

  return (
    <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50/60 p-3">
      <button
        type="button"
        className="flex w-full items-center justify-between text-xs font-semibold text-slate-600"
        onClick={() => setExpanded((prev) => !prev)}
      >
        <span className="flex items-center gap-2">
          <span className="text-base leading-none">{expanded ? "▾" : "▸"}</span>
          Thought for {formatDuration(totalMs)}
        </span>
        <span className="text-slate-400">{steps.length} steps</span>
      </button>
      {expanded && (
        <ol className="mt-3 space-y-2 border-l border-slate-200 pl-4 text-xs text-slate-600">
          {steps.map((step, index) => {
            const meta = summarizeMeta(step.meta as Record<string, unknown> | undefined);
            return (
              <li key={`${step.phase}-${index}`} className="space-y-1">
                <div className="font-semibold text-slate-700">{step.title || step.phase}</div>
                {step.detail && <div className="text-slate-500">{step.detail}</div>}
                <div className="text-slate-400 flex items-center gap-2 text-[11px]">
                  <span>{step.phase}</span>
                  <span>•</span>
                  <span>{formatDuration(step.elapsed_ms)}</span>
                  {meta && (
                    <>
                      <span>•</span>
                      <span>{meta}</span>
                    </>
                  )}
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
};
