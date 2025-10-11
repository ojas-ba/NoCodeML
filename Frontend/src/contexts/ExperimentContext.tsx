import { createContext, useContext, useState, ReactNode, useEffect } from "react";
import { experimentAPI } from "@/services/apiService";
import { toast } from "sonner";

export interface Experiment {
  id: string;
  name: string;
  datasetId: string;
  datasetName?: string;
  status: "in_progress" | "completed";
  createdAt: string;
  updatedAt: string;
  config: {
    taskType?: "classification" | "regression";
    targetColumn?: string;
    selectedFeatures?: string[]; // NEW: replaces features
    features?: string[]; // DEPRECATED: for backward compatibility
    featureTypes?: {
      numerical: string[];
      categorical: string[];
    };
    excludedColumns?: string[];
    trainTestSplit?: number;
    randomSeed?: number;
    selectedModels?: string[]; // DEPRECATED: use models instead
    models?: Array<{
      model_type: string;
      display_name: string;
      preset: string;
      hyperparameters: Record<string, any>;
      custom_hyperparameters?: Record<string, any> | null;
    }>;
  };
  results?: any;
}

interface ExperimentContextType {
  currentExperiment: Experiment | null;
  experiments: Experiment[];
  loading: boolean;
  setCurrentExperiment: (experiment: Experiment | null) => void;
  loadExperiment: (id: string) => Promise<void>;
  createExperiment: (name: string, datasetId: string) => Promise<Experiment>;
  updateExperiment: (id: string, updates: Partial<Experiment>) => Promise<void>;
  deleteExperiment: (id: string) => Promise<void>;
  fetchExperiments: () => Promise<void>;
  saveExperimentConfig: (config: Partial<Experiment["config"]>) => Promise<void>;
  saveExperimentResults: (results: any) => Promise<void>;
}

const ExperimentContext = createContext<ExperimentContextType | undefined>(undefined);

export const ExperimentProvider = ({ children }: { children: ReactNode }) => {
  const [currentExperiment, setCurrentExperiment] = useState<Experiment | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchExperiments = async () => {
    setLoading(true);
    try {
      const response = await experimentAPI.list();
      // Backend returns paginated response with experiments array
      setExperiments(response.experiments || response);
    } catch (error: any) {
      toast.error(error.message || "Failed to fetch experiments");
    } finally {
      setLoading(false);
    }
  };

  const loadExperiment = async (id: string) => {
    setLoading(true);
    try {
      const experiment = await experimentAPI.get(id);
      setCurrentExperiment(experiment);
    } catch (error: any) {
      toast.error(error.message || "Failed to load experiment");
    } finally {
      setLoading(false);
    }
  };

  const createExperiment = async (name: string, datasetId: string) => {
    setLoading(true);
    try {
      const experiment = await experimentAPI.create({ name, datasetId });
      setExperiments([experiment, ...experiments]);
      setCurrentExperiment(experiment);
      toast.success("Experiment created successfully!");
      return experiment;
    } catch (error: any) {
      toast.error(error.message || "Failed to create experiment");
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const updateExperiment = async (id: string, updates: Partial<Experiment>) => {
    try {
      const updated = await experimentAPI.update(id, updates);
      setExperiments(experiments.map(exp => exp.id === id ? updated : exp));
      if (currentExperiment?.id === id) {
        setCurrentExperiment(updated);
      }
      toast.success("Experiment updated successfully!");
    } catch (error: any) {
      toast.error(error.message || "Failed to update experiment");
      throw error;
    }
  };

  const deleteExperiment = async (id: string) => {
    try {
      await experimentAPI.delete(id);
      setExperiments(experiments.filter(exp => exp.id !== id));
      if (currentExperiment?.id === id) {
        setCurrentExperiment(null);
      }
      toast.success("Experiment deleted successfully!");
    } catch (error: any) {
      toast.error(error.message || "Failed to delete experiment");
      throw error;
    }
  };

  const saveExperimentConfig = async (config: Partial<Experiment["config"]>) => {
    if (!currentExperiment) return;
    
    const updatedConfig = { ...currentExperiment.config, ...config };
    await updateExperiment(currentExperiment.id, { config: updatedConfig });
  };

  const saveExperimentResults = async (results: any) => {
    if (!currentExperiment) return;
    
    await updateExperiment(currentExperiment.id, { 
      results, 
      status: "completed" 
    });
  };

  return (
    <ExperimentContext.Provider
      value={{
        currentExperiment,
        experiments,
        loading,
        setCurrentExperiment,
        loadExperiment,
        createExperiment,
        updateExperiment,
        deleteExperiment,
        fetchExperiments,
        saveExperimentConfig,
        saveExperimentResults,
      }}
    >
      {children}
    </ExperimentContext.Provider>
  );
};

export const useExperiment = () => {
  const context = useContext(ExperimentContext);
  if (context === undefined) {
    throw new Error("useExperiment must be used within an ExperimentProvider");
  }
  return context;
};
