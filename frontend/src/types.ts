export const CLASS_NAMES = [
  "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
] as const;

export type ClassName = (typeof CLASS_NAMES)[number];

export interface Health {
  status: string;
  model_ready: boolean;
  uptime_sec: number;
  db_samples: number;
}

export interface PredictResult {
  predicted_class: number;
  predicted_label: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface RandomSample {
  label: number;
  label_name: string;
  pixels: number[];
}

export interface Insights {
  total_train_samples: number;
  num_classes: number;
  image_size: string;
  classes: Record<string, { name: string; count: number }>;
  pixel_statistics: { mean: number; std: number; min: number; max: number };
  db_uploaded_samples: number;
  model_ready: boolean;
  uptime_sec: number;
}

export interface EvaluationBlock {
  accuracy: number;
  test_loss?: number;
  f1_score: number;
  precision: number;
  recall: number;
  per_class_f1?: Record<string, number>;
  per_class_precision?: Record<string, number>;
  per_class_recall?: Record<string, number>;
}

export interface RetrainBlock {
  accuracy: number;
  f1_score: number;
  precision: number;
  recall: number;
  epochs_ran: number;
  samples: number;
}

export interface TrainingConfig {
  model?: string;
  input_shape?: number[];
  head?: string;
  phase1_epochs_ran?: number;
  phase2_epochs_ran?: number;
  batch_size?: number;
  val_split?: number;
  optimizer_phase1?: string;
  optimizer_phase2?: string;
  augmentation?: string[];
  callbacks?: string[];
}

export interface Metrics {
  training_config?: TrainingConfig;
  evaluation?: EvaluationBlock;
  initial_training?: EvaluationBlock;
  retrain?: RetrainBlock;
  history?: {
    accuracy?: number[];
    val_accuracy?: number[];
    loss?: number[];
    val_loss?: number[];
  };
}

export interface EpochLog {
  epoch: number;
  loss: number;
  accuracy: number;
  val_loss: number | null;
  val_accuracy: number | null;
}

export interface RetrainStatus {
  running: boolean;
  elapsed_sec: number | null;
  phase: string;
  current_epoch: number;
  total_epochs: number;
  epoch_logs: EpochLog[];
  n_samples: number;
  steps: { msg: string; elapsed: number }[];
  current_step: string;
  last_result:
    | ({ success?: boolean; error?: string } & Partial<RetrainBlock>)
    | null;
}

export interface RetrainHistoryRow {
  id?: number;
  trained_at?: string;
  samples_used?: number;
  accuracy?: number;
  f1_score?: number;
  precision?: number;
  recall?: number;
  epochs_ran?: number;
  notes?: string;
}

export interface UploadResponse {
  message: string;
  samples_added: number;
  total_in_db: number;
}
