// Frontend/src/types/experiment.ts

export interface FeatureTypes {
  numerical: string[];
  categorical: string[];
}

export interface ModelConfig {
  model_type: string;
  display_name: string;
  preset: "fast" | "balanced" | "accurate";
  hyperparameters: Record<string, any>;
  custom_hyperparameters?: Record<string, any> | null;
}

export interface ExperimentConfig {
  taskType?: "classification" | "regression";
  targetColumn?: string;
  selectedFeatures?: string[];
  featureTypes?: FeatureTypes;
  excludedColumns?: string[];
  trainTestSplit?: number;
  randomSeed?: number;
  models?: ModelConfig[];
  enableOptimization?: boolean;
  
  // Deprecated fields (backward compatibility)
  features?: string[];
  selectedModels?: string[];
}

export interface HyperparameterTuning {
  enabled: boolean;
  method: string;
  optimizer: string;
  n_trials: number;
  cv_folds: number;
  cv_strategy: string;
  best_trial: number;
  cv_score: number;
  test_score: number;
  improvement_vs_default: string;
  optimization_time_seconds: number;
  convergence_trial: number;
  trial_scores: number[];
  best_params: Record<string, any>;
  dataset_adjustments_applied: string[];
}

export interface ExperimentResponse {
  id: string;
  name: string;
  datasetId: string;
  datasetName?: string;
  status: "in_progress" | "completed";
  config: ExperimentConfig;
  results?: Record<string, any>;
  createdAt: string;
  updatedAt?: string;
}

export interface ColumnInfo {
  name: string;
  dtype: string;
  missing_count: number;
  missing_percent: number;
  unique_count: number;
  is_id_column: boolean;
  sample_values?: any[];
}

export interface EDAResponse {
  dataset_info: {
    id: string;
    name: string;
    row_count: number;
    column_count: number;
    file_size_bytes: number;
    file_name: string;
    memory_usage_bytes: number;
  };
  columns: ColumnInfo[];
  numeric_columns: string[];
  categorical_columns: string[];
  id_columns: string[];
  statistics: Record<string, any>;
  correlations: {
    columns: string[];
    matrix: number[][];
    pairs: Array<{
      col1: string;
      col2: string;
      correlation: number;
    }>;
  } | null;
  missing_data_summary: {
    total_missing: number;
    total_cells: number;
    missing_percent: number;
    columns_with_missing: Array<{
      column: string;
      missing_count: number;
      missing_percent: number;
    }>;
  };
  preview_data: {
    columns: string[];
    rows: Array<Record<string, any>>;
    total_rows: number;
    page_size: number;
  };
}
