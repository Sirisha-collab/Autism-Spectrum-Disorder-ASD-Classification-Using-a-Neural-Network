/**
 * Skeletons shaped like the content that replaces them, so the layout doesn't
 * jump when data lands.
 */
export function QuestionnaireSkeleton() {
  return (
    <div aria-busy="true" aria-label="Loading the questionnaire">
      <div className="border-b border-line-soft bg-paper">
        <div className="mx-auto max-w-3xl px-5 py-3">
          <div className="grid grid-cols-10 gap-1.5">
            {Array.from({ length: 10 }).map((_, i) => (
              <div key={i} className="skeleton aspect-square" />
            ))}
          </div>
          <div className="skeleton mt-3 h-2 w-40" />
        </div>
      </div>

      <div className="mx-auto max-w-3xl px-5 py-8">
        <div className="skeleton h-4 w-full max-w-md" />
        <div className="skeleton mt-2 h-4 w-2/3 max-w-sm" />
        <div className="mt-8 grid gap-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="card p-5">
              <div className="skeleton h-3 w-16" />
              <div className="skeleton mt-3 h-5 w-3/4" />
              <div className="mt-4 grid gap-2">
                {Array.from({ length: 5 }).map((_, j) => (
                  <div key={j} className="skeleton h-11 w-full" />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export function MetricsSkeleton() {
  return (
    <div
      className="mx-auto max-w-3xl px-5 py-8"
      aria-busy="true"
      aria-label="Loading metrics"
    >
      <div className="skeleton h-3 w-24" />
      <div className="skeleton mt-2 h-8 w-52" />
      <div className="skeleton mt-4 h-4 w-full max-w-lg" />
      <div className="mt-8 grid gap-3 sm:grid-cols-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="card p-4">
            <div className="skeleton h-3 w-20" />
            <div className="skeleton mt-2 h-6 w-16" />
            <div className="skeleton mt-2 h-3 w-28" />
          </div>
        ))}
      </div>
      <div className="skeleton mt-8 h-64 w-full rounded-lg" />
    </div>
  );
}
