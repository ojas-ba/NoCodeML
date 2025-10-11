import React, { createContext, useContext, useState, useEffect } from 'react';
import { useToast } from '@/hooks/use-toast';
import apiService from '@/services/apiService';

// Run-based architecture types
interface TrainingRun {
  id: string;
  run_number: number;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  started_at?: string;
  completed_at?: string;
  duration_seconds?: number;
  progress?: {
    percent: number;
    message: string;
  };
  results_summary?: {
    total_models: number;
    successful: number;
    failed: number;
    best_model?: {
      model_type: string;
      display_name: string;
      metric: string;
      value: number;
    };
  };
  error_message?: string;
  created_at: string;
}

interface TrainingContextType {
  currentRun: TrainingRun | null;
  isTraining: boolean;
  error: string | null;
  
  startTraining: (experimentId: string) => Promise<void>;
  stopPolling: () => void;
  clearTraining: () => void;
}

const TrainingContext = createContext<TrainingContextType | undefined>(undefined);

export const useTraining = () => {
  const context = useContext(TrainingContext);
  if (!context) {
    throw new Error('useTraining must be used within TrainingProvider');
  }
  return context;
};

export const TrainingProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currentRun, setCurrentRun] = useState<TrainingRun | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);
  
  const { toast } = useToast();

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
        setPollingInterval(null);
      }
    };
  }, []); // Empty deps - only run on unmount

  const startTraining = async (experimentId: string) => {
    // Prevent multiple simultaneous training runs
    if (isTraining) {
      toast({
        title: "Training in Progress",
        description: "Please wait for current training to complete",
        variant: "destructive"
      });
      return;
    }

    try {
      setError(null);
      setIsTraining(true);
      
      // Start training run
      const response = await apiService.training.startRun(experimentId);
      
      setCurrentRun({
        id: response.run_id,
        run_number: response.run_number,
        status: 'pending',
        created_at: response.created_at
      });
      
      toast({
        title: "Training Started",
        description: `Run #${response.run_number} has been queued`,
      });
      
      // Start polling with retry limit
      let retryCount = 0;
      const maxRetries = 100; // 5 minutes max (100 * 3 seconds)
      
      const interval = setInterval(async () => {
        // Check retry limit
        if (retryCount++ > maxRetries) {
          clearInterval(interval);
          setPollingInterval(null);
          setIsTraining(false);
          setError('Polling timeout - please refresh to check status');
          toast({
            title: "Polling Timeout",
            description: "Training may still be running. Please refresh to check status.",
            variant: "destructive"
          });
          return;
        }

        try {
          const status = await apiService.training.getRunStatus(response.run_id);
          
          setCurrentRun(prev => prev ? { 
            ...prev, 
            status: status.status,
            progress: status.progress,
            error_message: status.error_message,
            results_summary: status.results_summary 
          } : null);
          
          // Terminal states - stop polling
          if (status.status === 'completed' || status.status === 'failed') {
            clearInterval(interval);
            setPollingInterval(null);
            setIsTraining(false);
            
            if (status.status === 'completed') {
              toast({
                title: "Training Complete",
                description: `Run #${response.run_number} finished successfully`,
              });
            } else {
              setError(status.error_message || 'Training failed');
              toast({
                title: "Training Failed",
                description: status.error_message || 'Unknown error',
                variant: "destructive",
              });
            }
          }
        } catch (err) {
          console.error('Polling error:', err);
          // Don't stop polling on transient errors
        }
      }, 3000);
      
      setPollingInterval(interval);
      
    } catch (err: any) {
      setIsTraining(false);
      setError(err.response?.data?.detail || 'Failed to start training');
      toast({
        title: "Training Failed",
        description: err.response?.data?.detail || 'Failed to start training',
        variant: "destructive",
      });
    }
  };

  const stopPolling = () => {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      setPollingInterval(null);
    }
    setIsTraining(false);
  };

  const clearTraining = () => {
    stopPolling();
    setCurrentRun(null);
    setError(null);
  };

  const value: TrainingContextType = {
    currentRun,
    isTraining,
    error,
    startTraining,
    stopPolling,
    clearTraining,
  };

  return (
    <TrainingContext.Provider value={value}>
      {children}
    </TrainingContext.Provider>
  );
};

export default TrainingContext;