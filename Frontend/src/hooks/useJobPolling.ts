import { useState, useEffect, useCallback } from "react";
import { jobAPI } from "@/services/apiService";
import { toast } from "sonner";

export interface JobStatus {
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  resultUrl: string | null;
  error: string | null;
}

interface UseJobPollingOptions {
  jobId: string | null;
  onComplete?: (resultUrl: string) => void;
  onError?: (error: string) => void;
  pollingInterval?: number;
  enabled?: boolean;
}

export const useJobPolling = ({
  jobId,
  onComplete,
  onError,
  pollingInterval = 2000,
  enabled = true
}: UseJobPollingOptions) => {
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [isPolling, setIsPolling] = useState(false);

  const checkJobStatus = useCallback(async () => {
    if (!jobId || !enabled) return;

    try {
      const status = await jobAPI.getStatus(jobId);
      setJobStatus(status);

      if (status.status === 'completed') {
        setIsPolling(false);
        if (status.resultUrl && onComplete) {
          onComplete(status.resultUrl);
        }
      } else if (status.status === 'failed') {
        setIsPolling(false);
        const errorMessage = status.error || 'Job failed';
        toast.error(errorMessage);
        if (onError) {
          onError(errorMessage);
        }
      }
    } catch (error: any) {
      setIsPolling(false);
      toast.error(error.message || 'Failed to check job status');
      if (onError) {
        onError(error.message);
      }
    }
  }, [jobId, enabled, onComplete, onError]);

  useEffect(() => {
    if (!jobId || !enabled) {
      setIsPolling(false);
      return;
    }

    setIsPolling(true);
    checkJobStatus();

    const interval = setInterval(() => {
      checkJobStatus();
    }, pollingInterval);

    return () => clearInterval(interval);
  }, [jobId, enabled, pollingInterval, checkJobStatus]);

  return {
    jobStatus,
    isPolling,
    refetch: checkJobStatus
  };
};