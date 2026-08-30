"use client";

import { useState } from "react";

/**
 * Nobody answers a question about their child before they know what this is,
 * what it can't tell them, and where the answers go.
 */
export default function ConsentGate({ onAccept }: { onAccept: () => void }) {
  const [agreed, setAgreed] = useState(false);

  return (
    <main className="mx-auto max-w-2xl px-5 py-12">
      <span className="eyebrow">Before you start</span>
      <h2 className="mt-1 text-3xl leading-tight">
        What this screening can and can&rsquo;t tell you
      </h2>

      <div className="mt-7 grid gap-4">
        <Point title="It is not a diagnosis">
          This is the Q-CHAT-10, a ten-question screen used to decide whether a
          child should be referred for a full developmental assessment. Autism is
          diagnosed by a clinician over multiple sessions. Nothing here replaces
          that.
        </Point>

        <Point title="Most children it flags are not autistic">
          Screening tools are built to catch as many cases as possible, which
          means they also flag many children who turn out not to be autistic. A
          flagged result is a reason to book an assessment, not a verdict.
        </Point>

        <Point title="A clear result doesn't settle the question">
          A ten-item screen can miss things. If you have concerns about your
          child&rsquo;s development, raise them with your paediatrician whatever
          this says.
        </Point>

        <Point title="It is designed for toddlers">
          The questions were validated on children 18 to 24 months old. Outside
          roughly 12 to 36 months the score isn&rsquo;t interpretable, and
          you&rsquo;ll be told if that applies.
        </Point>

        <Point title="Your answers stay on this device">
          Answers are sent to the scoring service and are not stored against your
          name. A draft is kept in this browser tab so you can finish later, and
          it clears when you close the tab.
        </Point>
      </div>

      <label className="mt-8 flex cursor-pointer items-start gap-3 rounded-lg border border-line bg-card p-4">
        <input
          type="checkbox"
          checked={agreed}
          onChange={(e) => setAgreed(e.target.checked)}
          className="mt-0.5 h-4 w-4 accent-[var(--color-accent)]"
        />
        <span className="text-sm leading-relaxed text-ink-soft">
          I understand this is a screening aid, not a diagnosis, and that I
          should discuss any concerns with a qualified clinician.
        </span>
      </label>

      <button
        type="button"
        disabled={!agreed}
        onClick={onAccept}
        className="mt-5 w-full rounded-md bg-accent px-5 py-3 text-sm font-medium text-white transition disabled:cursor-not-allowed disabled:bg-line disabled:text-muted"
      >
        Start the questionnaire
      </button>
    </main>
  );
}

function Point({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-line bg-card p-4">
      <h3 className="text-base text-ink">{title}</h3>
      <p className="mt-1.5 text-sm leading-relaxed text-ink-soft">{children}</p>
    </div>
  );
}
