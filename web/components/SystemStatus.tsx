"use client";

import { useEffect, useState } from "react";

type Health = {
  status: string;
  models_available: number;
  best_model_ready: boolean;
};

/**
 * Live backend status in the shell.
 *
 * A product that depends on a service says whether that service is up, rather
 * than letting the first failed action be the user's discovery. Polls slowly —
 * this is a status light, not telemetry.
 */
export default function SystemStatus() {
  const [health, setHealth] = useState<Health | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    let alive = true;

    const check = async () => {
      try {
        const res = await fetch("/api/health", { cache: "no-store" });
        if (!res.ok) throw new Error();
        const body = (await res.json()) as Health;
        if (alive) {
          setHealth(body);
          setDown(false);
        }
      } catch {
        if (alive) setDown(true);
      }
    };

    check();
    const timer = setInterval(check, 30_000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  const state = down
    ? { color: "var(--color-flag)", label: "Service offline" }
    : !health
      ? { color: "var(--color-muted)", label: "Checking" }
      : !health.best_model_ready
        ? { color: "var(--color-flag)", label: "No model trained" }
        : {
            color: "var(--color-accent)",
            label: `${health.models_available} models loaded`,
          };

  return (
    <div
      className="flex items-center gap-2"
      title={state.label}
      role="status"
      aria-live="polite"
    >
      <span
        aria-hidden="true"
        className="inline-block h-1.5 w-1.5 shrink-0 rounded-full"
        style={{ background: state.color }}
      />
      <span className="font-mono text-[0.625rem] uppercase tracking-wider text-muted">
        {state.label}
      </span>
    </div>
  );
}
