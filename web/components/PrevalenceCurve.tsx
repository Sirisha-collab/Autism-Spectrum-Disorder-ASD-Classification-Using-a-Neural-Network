"use client";

type Row = {
  prevalence: number;
  PPV: number;
  NPV: number;
  false_alarms_per_1000: number;
  missed_per_1000: number;
};

/**
 * PPV plotted against the prevalence of the population being screened.
 *
 * This is the project's central finding and it was previously only a table.
 * The curve makes the point immediately: the model is fixed, the population
 * isn't, and the same test behaves completely differently depending on who
 * walks through the door.
 */
export default function PrevalenceCurve({
  sensitivity,
  specificity,
  rows,
}: {
  sensitivity: number;
  specificity: number;
  rows: Row[];
}) {
  const W = 640;
  const H = 260;
  const PAD = { top: 16, right: 16, bottom: 34, left: 44 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const ppvAt = (p: number) => {
    const tp = sensitivity * p;
    const fp = (1 - specificity) * (1 - p);
    return tp + fp === 0 ? 0 : tp / (tp + fp);
  };

  const x = (p: number) => PAD.left + p * plotW;
  const y = (v: number) => PAD.top + (1 - v) * plotH;

  const points = Array.from({ length: 121 }, (_, i) => i / 120).map(
    (p) => `${x(p)},${y(ppvAt(p))}`,
  );
  const path = `M ${points.join(" L ")}`;
  const area = `${path} L ${x(1)},${y(0)} L ${x(0)},${y(0)} Z`;

  const markers = [
    { p: 0.017, label: "General population", tone: "var(--color-flag)" },
    { p: 0.69, label: "This dataset", tone: "var(--color-muted)" },
  ];

  const general = rows.find((r) => r.prevalence === 0.017);

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`Positive predictive value falls from ${(ppvAt(0.69) * 100).toFixed(0)} percent at 69 percent prevalence to ${(ppvAt(0.017) * 100).toFixed(0)} percent at 1.7 percent prevalence.`}
      >
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={y(v)}
              y2={y(v)}
              stroke="var(--color-line-soft)"
              strokeWidth="1"
            />
            <text
              x={PAD.left - 8}
              y={y(v) + 3.5}
              textAnchor="end"
              fill="var(--color-muted)"
              style={{ font: "500 9px var(--font-mono)" }}
            >
              {v * 100}%
            </text>
          </g>
        ))}

        <path d={area} fill="var(--color-accent-soft)" opacity="0.55" />
        <path
          d={path}
          fill="none"
          stroke="var(--color-accent)"
          strokeWidth="2"
          strokeLinejoin="round"
        />

        {markers.map((m) => (
          <g key={m.p}>
            <line
              x1={x(m.p)}
              x2={x(m.p)}
              y1={PAD.top}
              y2={y(0)}
              stroke={m.tone}
              strokeWidth="1"
              strokeDasharray="3 3"
            />
            <circle cx={x(m.p)} cy={y(ppvAt(m.p))} r="3.5" fill={m.tone} />
            <text
              x={x(m.p) + (m.p > 0.5 ? -8 : 8)}
              y={PAD.top + 11}
              textAnchor={m.p > 0.5 ? "end" : "start"}
              fill={m.tone}
              style={{ font: "500 10px var(--font-mono)" }}
            >
              {m.label}
            </text>
            <text
              x={x(m.p) + (m.p > 0.5 ? -8 : 8)}
              y={PAD.top + 24}
              textAnchor={m.p > 0.5 ? "end" : "start"}
              fill="var(--color-muted)"
              style={{ font: "400 10px var(--font-mono)" }}
            >
              PPV {(ppvAt(m.p) * 100).toFixed(1)}%
            </text>
          </g>
        ))}

        {[0, 0.25, 0.5, 0.75, 1].map((p) => (
          <text
            key={p}
            x={x(p)}
            y={H - 12}
            textAnchor="middle"
            fill="var(--color-muted)"
            style={{ font: "400 9px var(--font-mono)" }}
          >
            {p * 100}%
          </text>
        ))}

        <text
          x={PAD.left + plotW / 2}
          y={H - 1}
          textAnchor="middle"
          fill="var(--color-muted)"
          style={{ font: "400 9px var(--font-mono)" }}
        >
          prevalence in the screened population
        </text>
      </svg>

      <figcaption className="mt-3 text-sm leading-relaxed text-ink-soft">
        Sensitivity and specificity belong to the test and stay fixed along this
        curve. Positive predictive value belongs to the population.
        {general && (
          <>
            {" "}
            At general-population prevalence the PPV is{" "}
            <strong>{(general.PPV * 100).toFixed(1)}%</strong>, producing about{" "}
            {general.false_alarms_per_1000.toFixed(0)} false alarms per 1,000
            children screened — normal for a screening instrument, and exactly
            why the result routes to an assessment instead of stating a
            conclusion.
          </>
        )}
      </figcaption>
    </figure>
  );
}
