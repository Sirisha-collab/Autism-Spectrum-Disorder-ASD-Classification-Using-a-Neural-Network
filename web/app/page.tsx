"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ConsentGate from "@/components/ConsentGate";
import ItemLedger from "@/components/ItemLedger";
import QuestionCard from "@/components/QuestionCard";
import { QuestionnaireSkeleton } from "@/components/Skeleton";
import ResultPanel from "@/components/ResultPanel";
import { getModels, getQuestions, predict } from "@/lib/api";
import { clearDraft, loadDraft, saveDraft } from "@/lib/storage";
import type {
  ModelsPayload,
  PredictResponse,
  QuestionsPayload,
} from "@/lib/types";

type Stage = "consent" | "questions" | "result";

export default function ScreeningPage() {
  const [stage, setStage] = useState<Stage>("consent");
  const [data, setData] = useState<QuestionsPayload | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelsPayload | null>(null);
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [demographics, setDemographics] = useState<
    Record<string, string | number>
  >({});
  const [chosenModel, setChosenModel] = useState("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resumed, setResumed] = useState(false);
  const liveRef = useRef<HTMLParagraphElement>(null);

  useEffect(() => {
    getQuestions()
      .then((payload) => {
        setData(payload);
        const draft = loadDraft();
        if (draft && Object.keys(draft.answers).length > 0) {
          setAnswers(draft.answers);
          setDemographics(draft.demographics);
          setResumed(true);
          setStage("questions");
        } else {
          const defaults: Record<string, string | number> = {};
          for (const d of payload.demographics) {
            defaults[d.id] = d.kind === "int" ? 22 : (d.options?.[0] ?? "");
          }
          setDemographics(defaults);
        }
      })
      .catch((e: Error) => setError(e.message));

    getModels().then(setModelInfo).catch(() => setModelInfo(null));
  }, []);

  useEffect(() => {
    if (stage === "questions" && Object.keys(answers).length > 0) {
      saveDraft({ answers, demographics });
    }
  }, [answers, demographics, stage]);

  const firstUnanswered = useMemo(
    () => data?.items.find((i) => answers[i.id] === undefined)?.id ?? null,
    [data, answers],
  );

  const complete = data?.items.every((i) => answers[i.id] !== undefined) ?? false;

  const choose = useCallback(
    (itemId: string, optionIndex: number) => {
      setResult(null);
      setAnswers((prev) => ({ ...prev, [itemId]: optionIndex }));

      if (!data) return;
      const idx = data.items.findIndex((i) => i.id === itemId);
      const next = data.items[idx + 1];
      if (next && answers[next.id] === undefined) {
        requestAnimationFrame(() =>
          document.getElementById(next.id)?.scrollIntoView({ block: "center" }),
        );
      }
    },
    [data, answers],
  );

  async function runScreening() {
    setBusy(true);
    setError(null);
    try {
      const res = await predict({
        answers,
        demographics,
        model: chosenModel || null,
        consent: true,
      });
      setResult(res);
      setStage("result");
      clearDraft();
      requestAnimationFrame(() => {
        const el = document.getElementById("result");
        el?.scrollIntoView({ block: "center" });
        el?.focus();
      });
    } catch (e) {
      setError((e as Error).message);
      requestAnimationFrame(() => liveRef.current?.focus());
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    setAnswers({});
    setResult(null);
    setError(null);
    setResumed(false);
    clearDraft();
    setStage("consent");
    window.scrollTo({ top: 0 });
  }

  if (error && !data) {
    return (
      <main className="mx-auto max-w-2xl px-5 py-16">
        <h2 className="text-2xl">The screening service isn&rsquo;t running</h2>
        <p className="mt-3 text-sm leading-relaxed text-ink-soft">{error}</p>
        <pre className="mt-4 overflow-x-auto rounded-lg border border-line-soft bg-card p-4 font-mono text-xs">
          uvicorn api.main:app --reload --port 8000
        </pre>
        <button
          type="button"
          onClick={() => window.location.reload()}
          className="btn-secondary mt-4"
        >
          Try again
        </button>
      </main>
    );
  }

  if (!data) return <QuestionnaireSkeleton />;

  if (stage === "consent") {
    return <ConsentGate onAccept={() => setStage("questions")} />;
  }

  return (
    <>
      <ItemLedger
        items={data.items}
        answers={answers}
        cutoff={data.cutoff}
        activeId={firstUnanswered}
        onJump={(id) =>
          document.getElementById(id)?.scrollIntoView({ block: "center" })
        }
      />

      <main className="mx-auto max-w-3xl px-5 py-8">
        {resumed && (
          <p className="mb-6 rounded-md border border-accent bg-accent-soft p-3 text-sm text-ink-soft">
            Picked up where you left off. Your answers were saved in this tab.
          </p>
        )}

        <p className="max-w-xl text-sm leading-relaxed text-ink-soft">
          Answer for how your child usually behaves, not their best or worst day.
          There are no right answers, and a question you find hard to answer is
          itself worth mentioning to your paediatrician.
        </p>

        <div className="mt-8 grid gap-4">
          {data.items.map((item, i) => (
            <QuestionCard
              key={item.id}
              item={item}
              index={i}
              total={data.items.length}
              chosen={answers[item.id]}
              onChoose={(optionIndex) => choose(item.id, optionIndex)}
            />
          ))}
        </div>

        <section className="card mt-10 p-5">
          <span className="eyebrow">Background</span>
          <h2 className="mt-1 mb-4 text-base text-ink">About your child</h2>

          <div className="grid gap-4 sm:grid-cols-2">
            {data.demographics.map((d) => (
              <div key={d.id}>
                <label
                  htmlFor={`demo-${d.id}`}
                  className="mb-1.5 block text-sm text-ink-soft"
                >
                  {d.question}
                </label>
                {d.kind === "int" ? (
                  <>
                    <input
                      id={`demo-${d.id}`}
                      type="number"
                      min={data.ageRange.min}
                      max={data.ageRange.max}
                      value={demographics[d.id] as number}
                      aria-describedby={`demo-${d.id}-hint`}
                      onChange={(e) =>
                        setDemographics((p) => ({
                          ...p,
                          [d.id]: Number(e.target.value),
                        }))
                      }
                      className="w-full rounded-md border border-line bg-paper px-3 py-2 font-mono text-sm"
                    />
                    <p
                      id={`demo-${d.id}-hint`}
                      className="mt-1 font-mono text-[0.625rem] text-muted"
                    >
                      validated for {data.ageRange.validated[0]}–
                      {data.ageRange.validated[1]} months
                    </p>
                  </>
                ) : (
                  <select
                    id={`demo-${d.id}`}
                    value={demographics[d.id] as string}
                    onChange={(e) =>
                      setDemographics((p) => ({ ...p, [d.id]: e.target.value }))
                    }
                    className="w-full rounded-md border border-line bg-paper px-3 py-2 text-sm"
                  >
                    {d.options?.map((o) => (
                      <option key={o} value={o}>
                        {o}
                      </option>
                    ))}
                  </select>
                )}
              </div>
            ))}
          </div>

          {modelInfo && modelInfo.models.length > 0 && (
            <div className="mt-5 border-t border-line pt-5">
              <label htmlFor="model-select" className="eyebrow">
                Model
              </label>
              <select
                id="model-select"
                value={chosenModel}
                onChange={(e) => setChosenModel(e.target.value)}
                className="mt-1.5 w-full rounded-md border border-line bg-paper px-3 py-2 text-sm sm:max-w-xs"
              >
                <option value="">
                  Calibrated best model
                  {modelInfo.best ? ` (${modelInfo.best})` : ""}
                </option>
                {modelInfo.models.map((m) => (
                  <option key={m.stem} value={m.stem}>
                    {m.name}
                  </option>
                ))}
              </select>
              <p className="mt-1.5 text-xs text-muted">
                The default is calibrated and uses a threshold tuned for
                sensitivity. The others run at 0.5 and are here for comparison.
              </p>
            </div>
          )}
        </section>

        <p
          ref={liveRef}
          tabIndex={-1}
          role="alert"
          className={
            error
              ? "mt-6 rounded-md border border-flag bg-flag-soft p-4 text-sm text-ink"
              : "sr-only"
          }
        >
          {error ?? ""}
        </p>

        <button
          type="button"
          disabled={!complete || busy}
          onClick={runScreening}
          className={`
    mt-6 w-full rounded-lg px-5 py-3.5
    text-sm font-semibold text-white
    shadow-sm transition-all duration-200
    focus:outline-none focus-visible:ring-2 focus-visible:ring-green-500
    focus-visible:ring-offset-2
    ${complete && !busy
              ? "bg-green-600 hover:-translate-y-0.5 hover:bg-green-700 hover:shadow-lg"
              : "cursor-not-allowed bg-gray-300 text-gray-500"
            }
  `}
        >
          {busy ? (
            <span className="flex items-center justify-center gap-2">
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
              Submitting…
            </span>
          ) : complete ? (
            <span className="flex items-center justify-center gap-2">
              Submit & Generate Result
              <span aria-hidden="true">→</span>
            </span>
          ) : (
            `${data.items.length - Object.keys(answers).length} questions left`
          )}
        </button>

        {result && (
          <div className="mt-6">
            <ResultPanel result={result} onReset={reset} />
          </div>
        )}
      </main>
    </>
  );
}
