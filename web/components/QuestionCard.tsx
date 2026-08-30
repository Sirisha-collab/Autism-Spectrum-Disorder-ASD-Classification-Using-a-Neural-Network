"use client";

import { useEffect, useRef } from "react";
import type { QItem } from "@/lib/types";

type Props = {
  item: QItem;
  index: number;
  total: number;
  chosen: number | undefined;
  onChoose: (optionIndex: number) => void;
};

/**
 * A radiogroup, not a row of buttons. Arrow keys move between options and the
 * whole group is one tab stop, which is what a screen reader user expects.
 */
export default function QuestionCard({
  item,
  index,
  total,
  chosen,
  onChoose,
}: Props) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  useEffect(() => {
    refs.current = refs.current.slice(0, item.options.length);
  }, [item.options.length]);

  function handleKey(e: React.KeyboardEvent, i: number) {
    const last = item.options.length - 1;
    let next: number | null = null;

    if (e.key === "ArrowDown" || e.key === "ArrowRight") next = i === last ? 0 : i + 1;
    if (e.key === "ArrowUp" || e.key === "ArrowLeft") next = i === 0 ? last : i - 1;
    if (e.key === "Home") next = 0;
    if (e.key === "End") next = last;

    if (next !== null) {
      e.preventDefault();
      onChoose(next);
      refs.current[next]?.focus();
    }
  }

  return (
    <section
      id={item.id}
      className="scroll-mt-36 rounded-lg border border-line bg-card p-5"
      aria-labelledby={`${item.id}-label`}
    >
      <div className="mb-3 flex items-baseline gap-3">
        <span className="eyebrow">{item.id}</span>
        <span className="font-mono text-[0.6875rem] text-muted">
          {index + 1} of {total}
        </span>
      </div>

      <h2 id={`${item.id}-label`} className="mb-4 text-base leading-snug text-ink">
        {item.question}
      </h2>

      <div
        role="radiogroup"
        aria-labelledby={`${item.id}-label`}
        className="grid gap-2"
      >
        {item.options.map((option, i) => {
          const selected = chosen === i;
          return (
            <button
              key={option}
              ref={(el) => {
                refs.current[i] = el;
              }}
              type="button"
              role="radio"
              aria-checked={selected}
              aria-pressed={selected}
              tabIndex={selected || (chosen === undefined && i === 0) ? 0 : -1}
              className="option"
              onClick={() => onChoose(i)}
              onKeyDown={(e) => handleKey(e, i)}
            >
              <span className="option-key" aria-hidden="true">
                {i + 1}
              </span>
              <span>{option}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
