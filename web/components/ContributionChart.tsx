"use client";

import type { Contribution } from "@/lib/types";

/**
 * Diverging bars showing which answers drove the prediction.
 *
 * A parent shown "94%" should be able to see what produced it. Bars are scaled
 * to the largest absolute contribution rather than to an absolute unit, because
 * the raw SHAP magnitudes are in the model's internal units and mean nothing on
 * their own — the useful information is the ranking and the direction.
 */
export default function ContributionChart({
  contributions,
  accent,
}: {
  contributions: Contribution[];
  accent: string;
}) {
  if (contributions.length === 0) return null;

  const max = Math.max(...contributions.map((c) => Math.abs(c.contribution)));

  return (
    <section className="mt-6 border-t border-line pt-5">
      <span className="eyebrow">What drove this result</span>
      <p className="mt-1.5 text-sm leading-relaxed text-ink-soft">
        Answers pushing toward a referral are on the right, answers pushing away
        on the left. Length shows relative influence, not certainty.
      </p>

      <ul className="mt-4 grid gap-2">
        {contributions.map((c, i) => {
          const width = (Math.abs(c.contribution) / max) * 50;
          const up = c.contribution > 0;
          return (
            <li
              key={c.feature}
              className="rise grid grid-cols-[1fr_auto] gap-3"
              style={{ animationDelay: `${i * 55}ms` }}
            >
              <div>
                <div className="mb-1 flex items-baseline justify-between gap-2">
                  <span className="text-sm text-ink">{c.label}</span>
                  {!c.is_item && (
                    <span className="font-mono text-[0.625rem] uppercase tracking-wider text-muted">
                      background
                    </span>
                  )}
                </div>

                <div
                  className="relative h-3 rounded-sm bg-paper"
                  role="img"
                  aria-label={`${c.label} ${
                    up ? "increases" : "decreases"
                  } the likelihood, relative influence ${Math.round(
                    (Math.abs(c.contribution) / max) * 100,
                  )} percent`}
                >
                  <div className="absolute left-1/2 top-0 h-full w-px bg-line" />
                  <div
                    className="absolute top-0 h-full rounded-sm transition-[width] duration-700 ease-out"
                    style={{
                      width: `${width}%`,
                      left: up ? "50%" : `${50 - width}%`,
                      background: up ? accent : "var(--color-muted)",
                    }}
                  />
                </div>
              </div>

              <span className="self-end pb-0.5 font-mono text-[0.625rem] text-muted">
                {up ? "↑" : "↓"}
              </span>
            </li>
          );
        })}
      </ul>

      <p className="mt-4 text-xs leading-relaxed text-muted">
        These are SHAP values: each bar is that answer&rsquo;s contribution to
        this specific prediction, holding the others fixed. They explain what the
        model did, which is not the same as explaining your child.
      </p>
    </section>
  );
}
