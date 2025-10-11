import { useState, useCallback, useEffect } from 'react';
import { edaAPI } from '@/services/apiService';
import { EDAResponse, ColumnInfo } from '@/types/experiment';

interface PlotResponse {
  data: any[];
  layout: Record<string, any>;
  is_sampled: boolean;
  total_rows: number;
  displayed_rows: number;
  plot_type: string;
}

interface PlotConfig {
  plot_type: string;
  x_column: string;
  y_column?: string;
  group_by?: string;
  options?: Record<string, any>;
}

interface EDAHook {
  edaData: EDAResponse | null;
  plotData: PlotResponse | null;
  isLoading: boolean;
  plotLoading: boolean;
  error: string | null;
  plotError: string | null;
  loadEDASummary: (forceRefresh?: boolean) => Promise<void>;
  generatePlot: (config: PlotConfig) => Promise<void>;
}

// Create a simple in-memory cache
const edaCache: Map<string, EDAResponse> = new Map();

export const useEDA = (datasetId: string | undefined) => {
  const [edaData, setEdaData] = useState<EDAResponse | null>(null);
  const [plotData, setPlotData] = useState<PlotResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [plotLoading, setPlotLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [plotError, setPlotError] = useState<string | null>(null);

  const loadEDASummary = useCallback(async (forceRefresh = false) => {
    if (!datasetId) {
      setError('No dataset ID provided');
      return;
    }

    // Check cache first
    if (!forceRefresh && edaCache.has(datasetId)) {
      setEdaData(edaCache.get(datasetId)!);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await edaAPI.getSummary(datasetId);
      setEdaData(response);
      edaCache.set(datasetId, response); // Cache the result
    } catch (err: any) {
      console.error('Failed to load EDA summary:', err);
      setError(err.message || 'Failed to load EDA summary');
    } finally {
      setIsLoading(false);
    }
  }, [datasetId]);

  // Auto-load on mount if datasetId is provided
  useEffect(() => {
    if (datasetId) {
      loadEDASummary();
    }
  }, [datasetId, loadEDASummary]);

  const generatePlot = useCallback(async (config: PlotConfig) => {
    if (!datasetId) {
      setPlotError('No dataset ID provided');
      return;
    }

    setPlotLoading(true);
    setPlotError(null);

    try {
      const response = await edaAPI.generatePlot(datasetId, config);
      setPlotData(response);
    } catch (err: any) {
      console.error('Failed to generate plot:', err);
      setPlotError(err.message || 'Failed to generate plot');
    } finally {
      setPlotLoading(false);
    }
  }, [datasetId]);

  return {
    edaData,
    plotData,
    isLoading,
    plotLoading,
    error,
    plotError,
    loadEDASummary,
    generatePlot
  };
};
