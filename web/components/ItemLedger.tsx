"use client";

import type { QItem } from "@/lib/types";

type Props = {
  items: QItem[];
  answers: Record<string, number>;
  cutoff: number;
  activeId: string | null;
  onJump: (id: string) => void;
};

/**
 * Ten cells, one per item, filling in as answers arrive, with the running total
 * beside them. The score is literally the sum of these cells, so the strip is
 * the scoring rule made visible rather than hidden behind a submit button.
 */
export default function ItemLedger({
  items,
  answers,
  cutoff,
  activeId,
  onJump,
}: Props) {
  const scoreOf = (item: QItem) => {
    const chosen = answers[item.id];
    if (chosen === undefined) return null;
    return item.scoringOptions.includes(chosen) ? 1 : 0;
  };

  const total = items.reduce((sum, item) => sum + (scoreOf(item) ?? 0), 0);
  const answered = items.filter((i) => answers[i.id] !== undefined).length;
  const complete = answered === items.length;

  return (
    <div className="sticky top-0 z-10 border-b border-line bg-paper/95 backdrop-blur">
      <div className="mx-auto max-w-3xl px-5 py-3">
        <div className="flex items-end justify-between gap-4">
          <div
            className="grid flex-1 grid-cols-10 gap-1.5"
            role="group"
            aria-label="Item scores"
          >
            {items.map((item) => {
              const score = scoreOf(item);
              const state =
                score === 1
                  ? "scored"
                  : score === 0
                    ? "zero"
                    : item.id === activeId
                      ? "active"
                      : "empty";
              return (
                <button
                  key={item.id}
                  type="button"
                  data-state={state}
                  className="ledger-cell"
                  onClick={() => onJump(item.id)}
                  aria-label={
                    score === null
                      ? `Item ${item.id}, not yet answered. Jump to it.`
                      : `Item ${item.id}, scored ${score}. Jump to it.`
                  }
                >
                  <span aria-hidden="true">
                    {score === null ? item.id.slice(1) : score}
                  </span>
                </button>
              );
            })}
          </div>

          <div className="shrink-0 text-right">
            <div className="eyebrow">Score</div>
            <div
              className="font-mono text-2xl leading-none"
              style={{
                color:
                  complete && total > cutoff ? "var(--color-flag)" : undefined,
              }}
            >
              {total}
              <span className="text-sm text-muted">/10</span>
            </div>
          </div>
        </div>

        <div
          className="mt-2 h-0.5 w-full overflow-hidden rounded-full bg-line"
          role="progressbar"
          aria-valuenow={answered}
          aria-valuemin={0}
          aria-valuemax={items.length}
          aria-label="Questionnaire progress"
        >
          <div
            className="h-full bg-accent transition-[width] duration-300"
            style={{ width: `${(answered / items.length) * 100}%` }}
          />
        </div>

        <p className="mt-1.5 font-mono text-[0.6875rem] text-muted">
          {answered} of {items.length} answered &middot; referral cut-off is{" "}
          {cutoff}
        </p>
      </div>
    </div>
  );
}
