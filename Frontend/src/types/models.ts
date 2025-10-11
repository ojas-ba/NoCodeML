// Frontend/src/types/models.ts

export interface MLModel {
  model_type: string;
  display_name: string;
  description: string;
}

export interface ModelsResponse {
  classification: MLModel[];
  regression: MLModel[];
}

export interface ModelsContextType {
  models: ModelsResponse | null;
  loading: boolean;
  error: string | null;
  getModelsByTask: (taskType: 'classification' | 'regression') => MLModel[];
  refreshModels: () => Promise<void>;
}
