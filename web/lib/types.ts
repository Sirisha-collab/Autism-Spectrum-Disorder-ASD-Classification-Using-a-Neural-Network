export type QItem = {
  id: string;
  question: string;
  options: string[];
  scoringOptions: number[];
};

export type DemographicQuestion = {
  id: string;
  question: string;
  kind: "int" | "choice";
  options: string[] | null;
};

export type QuestionsPayload = {
  items: QItem[];
  demographics: DemographicQuestion[];
  cutoff: number;
  ageRange: { min: number; max: number; validated: [number, number] };
};

export type Interval = { estimate: number; lo: number; hi: number };

export type Eligibility = {
  status: "validated" | "outside_validation_window" | "out_of_range";
  message: string;
};

export type PredictResponse = {
  screening_id: string;
  completed_at: string;
  prediction: number;
  label: string;
  probability: number | null;
  threshold: number;
  qchat_score: number;
  cutoff: number;
  above_cutoff: boolean;
  item_scores: Record<string, number>;
  model_name: string;
  model_version: string;
  calibration: string;
  eligibility: Eligibility;
  performance: {
    sensitivity: Interval | null;
    specificity: Interval | null;
    npv: Interval | null;
  };
  interpretation: string;
};

export type ModelsPayload = {
  models: { stem: string; name: string }[];
  best: string | null;
};

export type SelectionRow = {
  model: string;
  sensitivity: number;
  specificity: number;
  youden_j: number;
  cv_std: number;
  roc_auc: number;
  passes_floor: boolean;
  complexity: number;
};

export type Selection = {
  winner: string;
  f1_leader?: string;
  confident: boolean;
  headline: string;
  reason: string;
  runner_up: string | null;
  sensitivity_floor: number;
  criteria: { name: string; why: string }[];
  ranking: SelectionRow[];
  excluded?: SelectionRow[];
  mcnemar: Record<string, { p_value: number; note: string }>;
};

export type MetricsPayload = {
  columns: string[];
  rows: Record<string, string | number>[];
  production?: {
    best_model: string;
    n_train?: number;
    n_test?: number;
    operating_point: { threshold: number; sensitivity: number; specificity: number; note: string };
    bootstrap_ci: Record<string, Interval>;
    calibration?: { before: Record<string, number> | null; after: Record<string, number> };
    subgroups?: Record<string, { rows: Record<string, unknown>[]; sensitivity_gap: number | null }>;
    mcnemar?: Record<string, { p_value: number; note: string }>;
    selection?: Selection;
  };
  prevalence?: {
    prevalence: number;
    PPV: number;
    NPV: number;
    false_alarms_per_1000: number;
    missed_per_1000: number;
  }[];
};
