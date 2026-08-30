"use client";

import { useEffect, useState } from "react";
import ModelVerdict from "@/components/ModelVerdict";
import { getMetrics } from "@/lib/api";
import type { Interval, MetricsPayload } from "@/lib/types";

const HEADLINE = [
  "Recall (Sensitivity)",
  "Specificity",
  "Precision",
  "F1",
  "ROC-AUC",
  "MCC",
];

export default function MetricsPage() {
  const [data, setData] = useState<MetricsPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMetrics().then(setData).catch((e: Error) => setError(e.message));
  }, []);

  if (error) {
    return (
      <main className="mx-auto max-w-3xl px-5 py-16">
        <h2 className="text-2xl">No metrics yet</h2>
        <p className="mt-3 text-sm text-ink-soft">{error}</p>
        <pre className="mt-4 rounded-md border border-line bg-card p-4 font-mono text-xs">
          python src/train.py
        </pre>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="mx-auto max-w-3xl px-5 py-16" aria-busy="true">
        <p className="eyebrow">Loading metrics</p>
      </main>
    );
  }

  const columns = HEADLINE.filter((c) => data.columns.includes(c));
  const prod = data.production;
  const ci = prod?.bootstrap_ci ?? {};
  const generalPop = data.prevalence?.find((p) => p.prevalence === 0.017);

  return (
    <main className="mx-auto max-w-3xl px-5 py-8">
      <span className="eyebrow">Held-out test set</span>
      <h2 className="mt-1 text-2xl">Evaluation</h2>
      <p className="mt-3 max-w-xl text-sm leading-relaxed text-ink-soft">
        Sensitivity leads because this is a screen. A missed case delays
        intervention during the years it helps most; a false alarm costs one
        assessment that rules it out.
      </p>

      {prod?.selection && <ModelVerdict selection={prod.selection} />}

      {Object.keys(ci).length > 0 && (
        <>
          <h3 className="mt-9 text-lg">
            Headline metrics with 95% confidence intervals
          </h3>
          <p className="mt-2 text-sm text-ink-soft">
            Percentile bootstrap over {prod?.n_test ?? "the"} test rows. A point
            estimate from a test set this size is not a result on its own.
          </p>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            {Object.entries(ci)
              .filter(([k]) => k !== "F1")
              .map(([name, interval]) => (
                <IntervalCard
                  key={name}
                  name={name}
                  interval={interval as Interval}
                />
              ))}
          </div>
        </>
      )}

      {prod?.operating_point && (
        <section className="mt-9 rounded-lg border border-line bg-card p-5">
          <span className="eyebrow">Operating point</span>
          <h3 className="mt-1 text-lg">
            Threshold {prod.operating_point.threshold.toFixed(3)}, not 0.5
          </h3>
          <p className="mt-2 text-sm leading-relaxed text-ink-soft">
            {prod.operating_point.note} At this threshold the model catches{" "}
            {(prod.operating_point.sensitivity * 100).toFixed(1)}% of cases and
            correctly clears{" "}
            {(prod.operating_point.specificity * 100).toFixed(1)}% of
            non-cases. The default 0.5 is an artifact of the library, not a
            clinical decision.
          </p>
        </section>
      )}

      {data.prevalence && (
        <>
          <h3 className="mt-9 text-lg">PPV depends on who you screen</h3>
          <p className="mt-2 max-w-xl text-sm leading-relaxed text-ink-soft">
            Sensitivity and specificity belong to the test. Positive predictive
            value belongs to the population. This dataset is roughly 69%
            positive because families self-selected into it — deploy the same
            model in a general clinic and the picture changes completely.
          </p>
          <div className="mt-4 overflow-x-auto rounded-lg border border-line bg-card">
            <table className="w-full border-collapse text-sm">
              <caption className="sr-only">
                Positive and negative predictive value at different prevalences
              </caption>
              <thead>
                <tr className="border-b border-line">
                  <th className="eyebrow px-4 py-3 text-left">Prevalence</th>
                  <th className="eyebrow px-3 py-3 text-right">PPV</th>
                  <th className="eyebrow px-3 py-3 text-right">NPV</th>
                  <th className="eyebrow px-3 py-3 text-right">
                    False alarms / 1000
                  </th>
                  <th className="eyebrow px-3 py-3 text-right">Missed / 1000</th>
                </tr>
              </thead>
              <tbody>
                {data.prevalence.map((row) => (
                  <tr
                    key={row.prevalence}
                    className="border-b border-line last:border-0"
                  >
                    <td className="px-4 py-2.5 font-mono text-xs text-ink">
                      {(row.prevalence * 100).toFixed(1)}%
                      {row.prevalence === 0.017 && (
                        <span className="ml-2 text-muted">general pop.</span>
                      )}
                      {row.prevalence === 0.69 && (
                        <span className="ml-2 text-muted">this dataset</span>
                      )}
                    </td>
                    <Num value={row.PPV} pct />
                    <Num value={row.NPV} pct />
                    <Num value={row.false_alarms_per_1000} />
                    <Num value={row.missed_per_1000} />
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {generalPop && (
            <p className="mt-3 text-sm leading-relaxed text-ink-soft">
              At general-population prevalence the PPV is{" "}
              <strong>{(generalPop.PPV * 100).toFixed(1)}%</strong> — of every
              100 children flagged, roughly {Math.round(generalPop.PPV * 100)}{" "}
              would turn out to be autistic. That is normal for a screening
              instrument and is exactly why the result routes to an assessment
              rather than stating a conclusion.
            </p>
          )}
        </>
      )}

      <h3 className="mt-9 text-lg">All models</h3>
      <div className="mt-4 overflow-x-auto rounded-lg border border-line bg-card">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="border-b border-line">
              <th className="eyebrow px-4 py-3 text-left">Model</th>
              {columns.map((c) => (
                <th key={c} className="eyebrow px-3 py-3 text-right">
                  {c.replace(" (Sensitivity)", "")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row) => (
              <tr
                key={String(row.model)}
                className="border-b border-line last:border-0"
              >
                <td className="px-4 py-2.5 text-ink">{String(row.model)}</td>
                {columns.map((c) => (
                  <Num key={c} value={Number(row[c])} />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {prod?.mcnemar && Object.keys(prod.mcnemar).length > 0 && (
        <section className="mt-6 rounded-lg border border-line bg-card p-5">
          <span className="eyebrow">McNemar tests</span>
          <h3 className="mt-1 text-base">
            Are those differences bigger than chance?
          </h3>
          <ul className="mt-3 grid gap-1.5 text-sm text-ink-soft">
            {Object.entries(prod.mcnemar).map(([name, test]) => (
              <li key={name} className="flex flex-wrap gap-2">
                <span>vs {name}</span>
                <span className="font-mono text-xs text-muted">
                  p={test.p_value.toFixed(4)} · {test.note}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {Number(data.rows[0]?.["F1"]) > 0.99 && (
        <section className="mt-6 rounded-lg border border-flag bg-flag-soft p-5">
          <span className="eyebrow">Interpret with care</span>
          <p className="mt-2 text-sm leading-relaxed text-ink-soft">
            Near-perfect scores here are a property of the dataset, not evidence
            the model detects autism traits. The label is <code>Yes</code>{" "}
            exactly when the Q-CHAT-10 score exceeds 3, and that score is the sum
            of A1–A10 — so the target is an arithmetic function of the inputs.
            Re-run with <code className="font-mono">--no-behaviour</code> for the
            honest baseline on background variables alone.
          </p>
        </section>
      )}

      <p className="mt-8 text-xs leading-relaxed text-muted">
        Calibration curves, precision-recall curves, the threshold sweep and
        subgroup breakdowns are written to{" "}
        <code className="font-mono">reports/</code>, alongside{" "}
        <code className="font-mono">MODEL_CARD.md</code>.
      </p>
    </main>
  );
}

function IntervalCard({ name, interval }: { name: string; interval: Interval }) {
  return (
    <div className="rounded-md border border-line bg-card px-4 py-3">
      <div className="eyebrow">{name}</div>
      <div className="mt-1 font-mono text-xl text-ink">
        {interval.estimate.toFixed(3)}
      </div>
      <div className="mt-0.5 font-mono text-[0.625rem] text-muted">
        95% CI {interval.lo.toFixed(3)} – {interval.hi.toFixed(3)}
      </div>
    </div>
  );
}

function Num({ value, pct = false }: { value: number; pct?: boolean }) {
  const display = Number.isFinite(value)
    ? pct
      ? `${(value * 100).toFixed(1)}%`
      : value.toFixed(3)
    : "—";
  return (
    <td
      className="px-3 py-2.5 text-right font-mono text-xs"
      style={{
        color: value >= 0.95 ? "var(--color-accent)" : "var(--color-ink-soft)",
      }}
    >
      {display}
    </td>
  );
}
