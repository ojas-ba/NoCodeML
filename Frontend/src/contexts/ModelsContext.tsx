import { createContext, useContext, useState, ReactNode, useEffect } from "react";
import { modelsAPI } from "@/services/apiService";
import { toast } from "sonner";
import { MLModel, ModelsResponse, ModelsContextType } from "@/types/models";

const ModelsContext = createContext<ModelsContextType | undefined>(undefined);

export const ModelsProvider = ({ children }: { children: ReactNode }) => {
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchModels = async () => {
    if (models) return; // Don't fetch if already loaded
    
    setLoading(true);
    setError(null);
    try {
      const response = await modelsAPI.getAll();
      setModels(response);
    } catch (error: any) {
      const errorMessage = error?.message || "Failed to fetch available models";
      setError(errorMessage);
      toast.error(errorMessage);
      console.error("Error fetching models:", error);
    } finally {
      setLoading(false);
    }
  };

  const refreshModels = async () => {
    setModels(null); // Clear cache to force refresh
    await fetchModels();
  };

  const getModelsByTask = (taskType: 'classification' | 'regression'): MLModel[] => {
    if (!models) return [];
    return models[taskType] || [];
  };

  // Auto-fetch models on mount
  useEffect(() => {
    fetchModels();
  }, []);

  const value: ModelsContextType = {
    models,
    loading,
    error,
    getModelsByTask,
    refreshModels
  };

  return (
    <ModelsContext.Provider value={value}>
      {children}
    </ModelsContext.Provider>
  );
};

export const useModels = (): ModelsContextType => {
  const context = useContext(ModelsContext);
  if (context === undefined) {
    throw new Error("useModels must be used within a ModelsProvider");
  }
  return context;
};