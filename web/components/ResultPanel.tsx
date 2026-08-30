"use client";

import type { Interval, PredictResponse } from "@/lib/types";

export default function ResultPanel({
  result,
  onReset,
}: {
  result: PredictResponse;
  onReset: () => void;
}) {
  const flagged = result.above_cutoff || result.prediction === 1;
  const outOfRange = result.eligibility.status === "out_of_range";

  const accent = outOfRange
    ? "var(--color-muted)"
    : flagged
      ? "var(--color-flag)"
      : "var(--color-accent)";
  const wash = outOfRange
    ? "var(--color-card)"
    : flagged
      ? "var(--color-flag-soft)"
      : "var(--color-accent-soft)";

  return (
    <section
      id="result"
      className="scroll-mt-36 rounded-lg border p-6 print:border-black"
      style={{ borderColor: accent, background: wash }}
      role="status"
      aria-live="polite"
      tabIndex={-1}
    >
      <span className="eyebrow">Result</span>

      <h2 className="mt-1 text-2xl leading-tight" style={{ color: accent }}>
        {outOfRange
          ? "This screen doesn't apply at your child's age"
          : flagged
            ? "Worth booking a developmental assessment"
            : "No elevated traits on this screen"}
      </h2>

      {result.eligibility.message && (
        <p className="mt-3 rounded-md border border-line bg-card p-3 text-sm leading-relaxed text-ink-soft">
          {result.eligibility.message}
        </p>
      )}

      <div className="mt-5 grid gap-3 sm:grid-cols-3">
        <Stat
          label="Q-CHAT-10 score"
          value={`${result.qchat_score}/10`}
          note={`cut-off ${result.cutoff}`}
        />
        <Stat
          label="Model estimate"
          value={
            result.probability === null
              ? "—"
              : `${Math.round(result.probability * 100)}%`
          }
          note={`threshold ${result.threshold.toFixed(2)}`}
        />
        <Stat
          label="Test sensitivity"
          value={fmtInterval(result.performance.sensitivity)}
          note="on held-out data"
        />
      </div>

      <p className="mt-5 text-sm leading-relaxed text-ink-soft">
        {result.interpretation}
      </p>

      <div className="mt-5 flex flex-wrap items-center gap-3">
        <span className="eyebrow">Item scores</span>
        <div className="flex flex-wrap gap-1">
          {Object.entries(result.item_scores).map(([id, score]) => (
            <span
              key={id}
              className="rounded-sm border border-line bg-card px-1.5 py-0.5 font-mono text-[0.6875rem]"
              style={{ color: score ? accent : "var(--color-muted)" }}
            >
              {id}:{score}
            </span>
          ))}
        </div>
      </div>

      <dl className="mt-5 grid gap-1 border-t border-line pt-4 font-mono text-[0.6875rem] text-muted">
        <Meta label="Reference" value={result.screening_id.slice(0, 8)} />
        <Meta label="Completed" value={result.completed_at.replace("T", " ")} />
        <Meta
          label="Model"
          value={`${result.model_name} v${result.model_version} (${result.calibration} calibrated)`}
        />
      </dl>

      <div className="mt-6 flex flex-wrap gap-3 print:hidden">
        <button
          type="button"
          onClick={() => window.print()}
          className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white"
        >
          Save or print for your appointment
        </button>
        <button
          type="button"
          onClick={onReset}
          className="rounded-md border border-ink-soft px-4 py-2 text-sm text-ink hover:bg-card"
        >
          Start a new screening
        </button>
      </div>
    </section>
  );
}

function fmtInterval(interval: Interval | null): string {
  if (!interval) return "—";
  return `${Math.round(interval.estimate * 100)}%`;
}

function Stat({
  label,
  value,
  note,
}: {
  label: string;
  value: string;
  note: string;
}) {
  return (
    <div className="rounded-md border border-line bg-card px-4 py-3">
      <div className="eyebrow">{label}</div>
      <div className="mt-1 font-mono text-xl text-ink">{value}</div>
      <div className="mt-0.5 font-mono text-[0.625rem] text-muted">{note}</div>
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex gap-2">
      <dt className="min-w-20 uppercase tracking-wider">{label}</dt>
      <dd className="m-0">{value}</dd>
    </div>
  );
}
