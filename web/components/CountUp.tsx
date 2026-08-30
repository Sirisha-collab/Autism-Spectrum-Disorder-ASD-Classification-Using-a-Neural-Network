"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Counts from zero to the final score.
 *
 * The ledger has been accumulating this number across ten questions, so the
 * reveal completes an arc the user has already been watching rather than
 * decorating the result. Skipped entirely under reduced-motion.
 */
export default function CountUp({
  to,
  duration = 700,
}: {
  to: number;
  duration?: number;
}) {
  const reduced =
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const [value, setValue] = useState(reduced ? to : 0);
  const frame = useRef<number>(0);

  useEffect(() => {
    if (reduced) {
      setValue(to);
      return;
    }
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(Math.round(eased * to));
      if (t < 1) frame.current = requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame.current);
  }, [to, duration, reduced]);

  return <>{value}</>;
}
