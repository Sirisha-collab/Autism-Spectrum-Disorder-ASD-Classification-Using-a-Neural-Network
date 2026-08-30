"use client";

import type { Selection } from "@/lib/types";

/**
 * States which model is deployed and why, since "sorted by F1, take row one"
 * is the wrong rule for a screening tool and the ranking alone doesn't say so.
 */
export default function ModelVerdict({ selection }: { selection: Selection }) {
  const { winner, f1_leader, confident, reason, ranking, excluded } = selection;
  const disagrees = f1_leader && f1_leader !== winner;

  return (
    <section className="mt-9">
      <span className="eyebrow">Model choice</span>
      <h3 className="mt-1 text-2xl">{winner}</h3>

      <div
        className="mt-4 rounded-lg border p-5"
        style={{
          borderColor: confident
            ? "var(--color-accent)"
            : "var(--color-flag)",
          background: confident
            ? "var(--color-accent-soft)"
            : "var(--color-flag-soft)",
        }}
      >
        <p className="text-sm leading-relaxed text-ink-soft">{reason}</p>

        {!confident && (
          <p className="mt-3 border-t border-line pt-3 text-sm leading-relaxed text-ink-soft">
            The margin over the runner-up sits inside the noise, so treat this
            as &ldquo;no model is clearly better&rdquo; rather than a win.
            Re-running with a different random seed could reorder the top few.
          </p>
        )}
      </div>

      {disagrees && (
        <div className="mt-4 rounded-lg border border-line bg-card p-5">
          <span className="eyebrow">Why not the top row of the table</span>
          <p className="mt-2 text-sm leading-relaxed text-ink-soft">
            Sorting by F1 puts <strong>{f1_leader}</strong> first. F1 treats a
            missed case and a false alarm as equally costly, which is wrong
            here: a missed case delays intervention during the years it helps
            most, while a false alarm costs one assessment that rules it out.
            Ranking on sensitivity and separation instead selects{" "}
            <strong>{winner}</strong>, and that is the model the API serves.
          </p>
        </div>
      )}

      <h4 className="mt-7 text-base">How the choice was made</h4>
      <ol className="mt-3 grid gap-2">
        {selection.criteria.map((c, i) => (
          <li
            key={c.name}
            className="flex gap-3 rounded-md border border-line bg-card px-4 py-3"
          >
            <span className="font-mono text-xs text-muted">{i + 1}</span>
            <div>
              <div className="text-sm text-ink">{c.name}</div>
              <p className="mt-0.5 text-xs leading-relaxed text-ink-soft">
                {c.why}
              </p>
            </div>
          </li>
        ))}
      </ol>

      <h4 className="mt-7 text-base">Ranked on those criteria</h4>
      <div className="mt-3 overflow-x-auto rounded-lg border border-line bg-card">
        <table className="w-full border-collapse text-sm">
          <caption className="sr-only">
            Models ranked by Youden&rsquo;s J among those clearing the
            sensitivity floor
          </caption>
          <thead>
            <tr className="border-b border-line">
              <th className="eyebrow px-4 py-3 text-left">Model</th>
              <th className="eyebrow px-3 py-3 text-right">Sens.</th>
              <th className="eyebrow px-3 py-3 text-right">Spec.</th>
              <th className="eyebrow px-3 py-3 text-right">Youden J</th>
              <th className="eyebrow px-3 py-3 text-right">CV σ</th>
            </tr>
          </thead>
          <tbody>
            {ranking.map((row, i) => {
              const isWinner = row.model === winner;
              return (
                <tr
                  key={row.model}
                  className="border-b border-line last:border-0"
                  style={{
                    background: isWinner ? "var(--color-accent-soft)" : undefined,
                  }}
                >
                  <td className="px-4 py-2.5 text-ink">
                    <span className="mr-2 font-mono text-xs text-muted">
                      {i + 1}
                    </span>
                    {row.model}
                    {isWinner && (
                      <span className="ml-2 font-mono text-[0.625rem] uppercase tracking-wider text-accent">
                        deployed
                      </span>
                    )}
                  </td>
                  <Cell value={row.sensitivity} />
                  <Cell value={row.specificity} />
                  <Cell value={row.youden_j} strong />
                  <Cell value={row.cv_std} />
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {excluded && excluded.length > 0 && (
        <div className="mt-3 rounded-lg border border-line bg-card p-4">
          <span className="eyebrow">
            Excluded — below the{" "}
            {(selection.sensitivity_floor * 100).toFixed(0)}% sensitivity floor
          </span>
          <ul className="mt-2 grid gap-1 text-sm text-ink-soft">
            {excluded.map((row) => (
              <li key={row.model} className="flex flex-wrap gap-2">
                <span>{row.model}</span>
                <span className="font-mono text-xs text-muted">
                  sensitivity {(row.sensitivity * 100).toFixed(1)}% — misses{" "}
                  {((1 - row.sensitivity) * 100).toFixed(1)}% of cases
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

function Cell({ value, strong = false }: { value: number; strong?: boolean }) {
  return (
    <td
      className="px-3 py-2.5 text-right font-mono text-xs"
      style={{
        color: strong ? "var(--color-ink)" : "var(--color-ink-soft)",
      }}
    >
      {Number.isFinite(value) ? value.toFixed(3) : "—"}
    </td>
  );
}
