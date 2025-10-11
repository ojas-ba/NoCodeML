import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Loader2, Play, CheckCircle2, XCircle, Clock } from 'lucide-react';
import { useTraining } from '@/contexts/TrainingContext';

interface TrainingStepProps {
  experimentId: string;
  experimentConfig: any;
  onComplete: () => void;
}

const TrainingStep: React.FC<TrainingStepProps> = ({
  experimentId,
  experimentConfig,
  onComplete
}) => {
  const { currentRun, isTraining, error, startTraining } = useTraining();
  
  // Validate experiment config exists
  if (!experimentConfig) {
    return (
      <div className="flex items-center justify-center py-12">
        <p className="text-muted-foreground">Loading configuration...</p>
      </div>
    );
  }

  const modelsCount = experimentConfig?.models?.length || 0;
  
  const handleStart = async () => {
    await startTraining(experimentId);
  };
  
  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <Badge className="bg-green-500"><CheckCircle2 className="w-4 h-4 mr-1" />Completed</Badge>;
      case 'running':
        return <Badge className="bg-blue-500"><Loader2 className="w-4 h-4 mr-1 animate-spin" />Training</Badge>;
      case 'failed':
        return <Badge variant="destructive"><XCircle className="w-4 h-4 mr-1" />Failed</Badge>;
      default:
        return <Badge variant="secondary"><Clock className="w-4 h-4 mr-1" />Pending</Badge>;
    }
  };
  
  const getProgress = () => {
    if (!currentRun) return 0;
    if (currentRun.status === 'completed') return 100;
    if (currentRun.status === 'failed') return 0;
    if (currentRun.status === 'running') {
      // Use progress from API (calculate_progress returns {percent, message})
      if (currentRun.progress?.percent !== undefined) {
        return currentRun.progress.percent;
      }
      return 50; // Fallback to 50% if no progress data
    }
    return 0; // pending
  };

  return (
    <div className="space-y-6">
      {/* Training Configuration Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Training Configuration</CardTitle>
          <CardDescription>
            Ready to train {modelsCount} model{modelsCount !== 1 ? 's' : ''} on your selected features
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Task Type:</span>
              <span className="font-medium">{experimentConfig?.taskType || 'N/A'}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Target Column:</span>
              <span className="font-medium">{experimentConfig?.targetColumn || 'N/A'}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Selected Features:</span>
              <span className="font-medium">{experimentConfig?.selectedFeatures?.length || 0}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Models:</span>
              <span className="font-medium">{modelsCount}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Training Status */}
      {currentRun && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Run #{currentRun.run_number}</CardTitle>
              {getStatusBadge(currentRun.status)}
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <Progress value={getProgress()} className="h-2" />
            
            {currentRun.status === 'running' && (
              <p className="text-sm text-muted-foreground">
                {currentRun.progress?.message || 'Training in progress... This may take several minutes.'}
              </p>
            )}
            
            {currentRun.status === 'completed' && currentRun.results_summary && (
              <div className="space-y-2">
                <p className="text-sm font-medium text-green-600">
                  ✓ Training completed successfully!
                </p>
                <div className="text-sm space-y-1">
                  <div>Models trained: {currentRun.results_summary.successful}/{currentRun.results_summary.total_models}</div>
                  {currentRun.results_summary.best_model && (
                    <div>Best model: {currentRun.results_summary.best_model.display_name} ({currentRun.results_summary.best_model.value.toFixed(3)})</div>
                  )}
                </div>
              </div>
            )}
            
            {error && (
              <div className="text-sm text-destructive">
                Error: {error}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Actions */}
      <div className="flex gap-3">
        <Button
          onClick={handleStart}
          disabled={isTraining || modelsCount === 0}
          className="flex-1"
        >
          {isTraining ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Training...
            </>
          ) : (
            <>
              <Play className="w-4 h-4 mr-2" />
              Start Training
            </>
          )}
        </Button>
        
        {currentRun?.status === 'completed' && (
          <Button onClick={onComplete} variant="secondary">
            View Results
          </Button>
        )}
      </div>
    </div>
  );
};

export default TrainingStep;
