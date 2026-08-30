/**
 * Draft persistence. A parent interrupted mid-questionnaire shouldn't lose ten
 * answers — but the draft is deliberately short-lived and local only.
 */
const KEY = "qchat10:draft:v1";
const MAX_AGE_MS = 1000 * 60 * 60 * 24; // 24 hours

export type Draft = {
  answers: Record<string, number>;
  demographics: Record<string, string | number>;
  savedAt: number;
};

export function saveDraft(draft: Omit<Draft, "savedAt">): void {
  try {
    sessionStorage.setItem(KEY, JSON.stringify({ ...draft, savedAt: Date.now() }));
  } catch {
    /* storage unavailable — the questionnaire still works, it just won't resume */
  }
}

export function loadDraft(): Draft | null {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (!raw) return null;
    const draft = JSON.parse(raw) as Draft;
    if (Date.now() - draft.savedAt > MAX_AGE_MS) {
      clearDraft();
      return null;
    }
    return draft;
  } catch {
    return null;
  }
}

export function clearDraft(): void {
  try {
    sessionStorage.removeItem(KEY);
  } catch {
    /* nothing to clean up */
  }
}
