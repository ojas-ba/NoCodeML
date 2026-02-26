import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Eye, Download, Loader2, ArrowLeft, AlertTriangle, CheckCircle2, AlertCircle, Sparkles, Info } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import apiService from '@/services/apiService';

interface ResultsStepProps {
  experimentId: string;
  onBack: () => void;
}

interface RunListItem {
  id: string;
  run_number: number;
  status: string;
  started_at?: string;
  completed_at?: string;
  duration_seconds?: number;
  results_summary?: any;
  created_at: string;
}

interface ConfusionMatrix {
  matrix: number[][];
  labels: string[];
}

interface FeatureImportance {
  features: string[];
  importance: number[];
}

interface Metrics {
  // New structure (nested train/test)
  train?: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
    r2_score?: number;
    mae?: number;
    rmse?: number;
    mse?: number;
  };
  test?: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
    r2_score?: number;
    mae?: number;
    rmse?: number;
    mse?: number;
  };
  // Old structure (flat) - for backward compatibility
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  r2_score?: number;
  mae?: number;
  rmse?: number;
  mse?: number;
}

interface RunDetails {
  id: string;
  run_number: number;
  status: string;
  config_snapshot: any;
  results: {
    task_type?: string;
    dataset_info?: {
      total_samples: number;
      train_samples: number;
      test_samples: number;
      n_features: number;
      feature_names: string[];
    };
    models: Array<{
      model_type: string;
      display_name: string;
      metrics: Metrics;
      feature_importance?: FeatureImportance;
      confusion_matrix?: ConfusionMatrix;
      error?: string;
    }>;
    best_model?: {
      model_type: string;
      display_name: string;
      metric: string;
      value: number;
    };
    summary: {
      total_models: number;
      successful: number;
      failed: number;
    };
  };
  artifacts?: any;
  created_at: string;
}

const ResultsStep: React.FC<ResultsStepProps> = ({ experimentId, onBack }) => {
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [selectedRun, setSelectedRun] = useState<RunDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    if (!experimentId) {
      setFetchError('No experiment ID provided');
      setLoading(false);
      return;
    }
    fetchRuns(page);
  }, [experimentId, page]);

  const fetchRuns = async (pageNum: number) => {
    if (!experimentId) {
      setFetchError('No experiment ID provided');
      setLoading(false);
      return;
    }
    
    try {
      setLoading(true);
      setFetchError(null);
      const response = await apiService.training.listRuns(experimentId, pageNum);
      
      if (!response || typeof response !== 'object') {
        throw new Error('Invalid response format');
      }
      
      setRuns(response.runs || []);
      setTotalPages(response.total_pages || 1);
    } catch (error: any) {
      console.error('Failed to fetch runs:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to load training runs';
      setFetchError(errorMessage);
      setRuns([]);
    } finally {
      setLoading(false);
    }
  };

  const fetchRunDetails = async (runId: string) => {
    try {
      setDetailsLoading(true);
      setFetchError(null);
      const details = await apiService.training.getRunDetails(runId);
      
      if (!details || typeof details !== 'object') {
        throw new Error('Invalid response format');
      }
      
      // Validate required fields
      if (!details.results || !details.config_snapshot) {
        throw new Error('Incomplete run data');
      }
      
      setSelectedRun(details);
    } catch (error: any) {
      console.error('Failed to fetch run details:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to load run details';
      setFetchError(errorMessage);
    } finally {
      setDetailsLoading(false);
    }
  };

  const formatDuration = (seconds?: number) => {
    if (!seconds) return 'N/A';
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <Badge className="bg-green-500">Completed</Badge>;
      case 'running':
        return <Badge className="bg-blue-500">Running</Badge>;
      case 'failed':
        return <Badge variant="destructive">Failed</Badge>;
      default:
        return <Badge variant="secondary">Pending</Badge>;
    }
  };

  const getOverfittingIndicator = (trainScore: number, testScore: number, isClassification: boolean) => {
    const gap = Math.abs(trainScore - testScore);
    const gapPercent = (gap * 100).toFixed(1);
    
    if (gap < 0.05) {
      return (
        <div className="flex items-center gap-2 text-green-600">
          <CheckCircle2 className="w-4 h-4" />
          <span>Good generalization (gap: {gapPercent}%)</span>
        </div>
      );
    } else if (gap < 0.15) {
      return (
        <div className="flex items-center gap-2 text-yellow-600">
          <AlertCircle className="w-4 h-4" />
          <span>Slight overfitting (gap: {gapPercent}%)</span>
        </div>
      );
    } else {
      return (
        <div className="flex items-center gap-2 text-red-600">
          <AlertTriangle className="w-4 h-4" />
          <span>High overfitting (gap: {gapPercent}%)</span>
        </div>
      );
    }
  };

  if (selectedRun) {
    // Detail View - Determine task type for dynamic metrics
    // Priority: results.task_type (what was actually used) > config_snapshot.taskType (what was configured)
    const taskType = selectedRun.results?.task_type || selectedRun.config_snapshot?.taskType || 'classification';
    const isClassification = taskType === 'classification';
    
    // Safety check for results
    if (!selectedRun.results || !selectedRun.results.models) {
      return (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold">Run #{selectedRun.run_number}</h2>
            <Button variant="outline" onClick={() => setSelectedRun(null)}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to List
            </Button>
          </div>
          <Card className="border-destructive">
            <CardContent className="py-6 text-center">
              <p className="text-destructive">No results available for this run</p>
            </CardContent>
          </Card>
        </div>
      );
    }
    
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold">Run #{selectedRun.run_number}</h2>
            <p className="text-sm text-muted-foreground">
              {new Date(selectedRun.created_at).toLocaleString()}
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => setSelectedRun(null)}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to List
            </Button>
            {getStatusBadge(selectedRun.status)}
          </div>
        </div>

        {/* Results Summary */}
        <Card>
          <CardHeader>
            <CardTitle>Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">Total Models</p>
                <p className="text-2xl font-bold">{selectedRun.results.summary?.total_models || 0}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Successful</p>
                <p className="text-2xl font-bold text-green-600">{selectedRun.results.summary?.successful || 0}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Failed</p>
                <p className="text-2xl font-bold text-red-600">{selectedRun.results.summary?.failed || 0}</p>
              </div>
            </div>
            
            {selectedRun.results.best_model && (
              <div className="mt-4 p-4 bg-primary/10 rounded-lg">
                <p className="text-sm font-medium">Best Model</p>
                <p className="text-xl font-bold">{selectedRun.results.best_model.display_name || 'Unknown'}</p>
                <p className="text-sm text-muted-foreground">
                  {selectedRun.results.best_model.metric || 'Score'}: {selectedRun.results.best_model.value?.toFixed(4) || 'N/A'}
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Model Results */}
        <Card>
          <CardHeader>
            <CardTitle>Model Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model</TableHead>
                  {isClassification ? (
                    <>
                      <TableHead>Accuracy</TableHead>
                      <TableHead>Precision</TableHead>
                      <TableHead>Recall</TableHead>
                      <TableHead>F1 Score</TableHead>
                    </>
                  ) : (
                    <>
                      <TableHead>R² Score</TableHead>
                      <TableHead>MAE</TableHead>
                      <TableHead>RMSE</TableHead>
                      <TableHead>MSE</TableHead>
                    </>
                  )}
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {selectedRun.results.models && selectedRun.results.models.length > 0 ? (
                  selectedRun.results.models.map((model) => (
                    <TableRow key={model.model_type}>
                      <TableCell className="font-medium">{model.display_name || model.model_type}</TableCell>
                      {model.error ? (
                        <>
                          <TableCell colSpan={4} className="text-destructive">{model.error}</TableCell>
                          <TableCell><Badge variant="destructive">Failed</Badge></TableCell>
                        </>
                      ) : model.metrics ? (
                        // Support both new structure (metrics.test) and old structure (metrics.accuracy)
                        isClassification ? (
                          <>
                            <TableCell>{(model.metrics.test?.accuracy ?? model.metrics.accuracy)?.toFixed(4) || 'N/A'}</TableCell>
                            <TableCell>{(model.metrics.test?.precision ?? model.metrics.precision)?.toFixed(4) || 'N/A'}</TableCell>
                            <TableCell>{(model.metrics.test?.recall ?? model.metrics.recall)?.toFixed(4) || 'N/A'}</TableCell>
                            <TableCell>{(model.metrics.test?.f1_score ?? model.metrics.f1_score)?.toFixed(4) || 'N/A'}</TableCell>
                            <TableCell><Badge className="bg-green-500">Success</Badge></TableCell>
                          </>
                        ) : (
                          <>
                            <TableCell>{(model.metrics.test?.r2_score ?? model.metrics.r2_score)?.toFixed(4) || 'N/A'}</TableCell>
                            <TableCell>{(model.metrics.test?.mae ?? model.metrics.mae)?.toFixed(4) || 'N/A'}</TableCell>
                            <TableCell>{(model.metrics.test?.rmse ?? model.metrics.rmse)?.toFixed(4) || 'N/A'}</TableCell>
                            <TableCell>{(model.metrics.test?.mse ?? model.metrics.mse)?.toFixed(4) || 'N/A'}</TableCell>
                            <TableCell><Badge className="bg-green-500">Success</Badge></TableCell>
                          </>
                        )
                      ) : (
                        <>
                          <TableCell colSpan={4}>No metrics available</TableCell>
                          <TableCell><Badge variant="secondary">Unknown</Badge></TableCell>
                        </>
                      )}
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={6} className="text-center text-muted-foreground">
                      No model results available
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        {/* Model Analysis Section - Loop through each successful model */}
        {selectedRun.results.models && selectedRun.results.models.filter(m => m.metrics && !m.error).map((model) => {
          // Extract train and test scores with null safety (support old and new structure)
          const trainScore = isClassification 
            ? (model.metrics?.train?.accuracy ?? 0)
            : (model.metrics?.train?.r2_score ?? 0);
          const testScore = isClassification 
            ? (model.metrics?.test?.accuracy ?? model.metrics?.accuracy ?? 0)
            : (model.metrics?.test?.r2_score ?? model.metrics?.r2_score ?? 0);
          
          return (
          <Card key={model.model_type} className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>{model.display_name} - Detailed Analysis</span>
                {model.model_type === selectedRun.results.best_model?.model_type && (
                  <Badge className="bg-yellow-500">Best Model</Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Hyperparameter Optimization Results - Transparent Expert System */}
              {model.hyperparameter_tuning?.enabled && (
                <div className="space-y-4">
                  {/* Expert System Header */}
                  <div className="p-4 bg-gradient-to-r from-primary/10 via-accent/10 to-primary-blue/10 rounded-lg border border-primary/30">
                    <div className="flex items-center gap-2 mb-2">
                      <Sparkles className="w-5 h-5 text-primary" />
                      <div className="font-semibold text-sm bg-gradient-to-r from-primary via-accent to-primary-blue bg-clip-text text-transparent">
                        Expert System — Optimization Summary
                      </div>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Our Expert System adjusted your model based on your data's specific characteristics
                      {model.hyperparameter_tuning.dataset_info && (
                        <span className="font-medium text-foreground">
                          {' '}(Size: {model.hyperparameter_tuning.dataset_info.n_samples?.toLocaleString()} samples, Features: {model.hyperparameter_tuning.dataset_info.n_features})
                        </span>
                      )}.
                    </p>
                  </div>

                  {/* Summary Metrics */}
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                    <div className="p-3 bg-card/50 backdrop-blur-sm rounded-lg border border-border hover:border-accent/50 transition-colors">
                      <div className="text-xs text-muted-foreground">Engine</div>
                      <div className="text-sm font-bold text-accent">
                        {model.hyperparameter_tuning.engine || 'NoCodeML Rules Engine'}
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        {model.hyperparameter_tuning.method}
                      </div>
                    </div>

                    <div className="p-3 bg-card/50 backdrop-blur-sm rounded-lg border border-border hover:border-primary-blue/50 transition-colors">
                      <div className="text-xs text-muted-foreground">Rules Evaluated</div>
                      <div className="text-lg font-bold text-primary-blue">
                        {model.hyperparameter_tuning.rules_evaluated ?? model.hyperparameter_tuning.applied_rules?.length ?? 0}
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        {model.hyperparameter_tuning.cv_strategy}
                      </div>
                    </div>

                    <div className="p-3 bg-card/50 backdrop-blur-sm rounded-lg border border-border hover:border-success/50 transition-colors">
                      <div className="text-xs text-muted-foreground">Final Configuration</div>
                      <div className="text-lg font-bold text-success">
                        {model.hyperparameter_tuning.test_score?.toFixed(4)}
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        Test Score
                      </div>
                    </div>

                    <div className="p-3 bg-card/50 backdrop-blur-sm rounded-lg border border-border hover:border-warning/50 transition-colors">
                      <div className="text-xs text-muted-foreground">Rules Applied</div>
                      <div className="text-lg font-bold text-warning">
                        {model.hyperparameter_tuning.rules_applied ?? model.hyperparameter_tuning.applied_rules?.filter((r: any) => r.parameter !== '—').length ?? 0}
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        Parameter adjustments
                      </div>
                    </div>
                  </div>

                  {/* Reasoning Table — Step-by-Step Logic */}
                  {model.hyperparameter_tuning.applied_rules && model.hyperparameter_tuning.applied_rules.length > 0 && (
                    <div className="rounded-lg border border-border overflow-hidden">
                      <div className="p-3 bg-primary-blue/10 border-b border-primary-blue/30">
                        <div className="font-semibold text-sm flex items-center gap-2 text-primary-blue">
                          <Info className="w-4 h-4" />
                          Step-by-Step Reasoning — Why Each Parameter Was Changed
                        </div>
                      </div>
                      <Table>
                        <TableHeader>
                          <TableRow className="bg-secondary/30">
                            <TableHead className="w-[140px] text-xs font-semibold">Parameter</TableHead>
                            <TableHead className="w-[90px] text-xs font-semibold text-center">Before</TableHead>
                            <TableHead className="w-[90px] text-xs font-semibold text-center">After</TableHead>
                            <TableHead className="text-xs font-semibold">The "Why" (Reasoning)</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {model.hyperparameter_tuning.applied_rules.map((rule: any, idx: number) => (
                            <TableRow key={idx} className="hover:bg-accent/5 transition-colors">
                              <TableCell className="font-mono text-xs font-semibold text-primary">
                                {rule.parameter}
                              </TableCell>
                              <TableCell className="text-xs text-center text-muted-foreground">
                                {typeof rule.original_value === 'object' ? JSON.stringify(rule.original_value) : String(rule.original_value)}
                              </TableCell>
                              <TableCell className="text-xs text-center font-semibold text-success">
                                {typeof rule.value === 'object' ? JSON.stringify(rule.value) : String(rule.value)}
                              </TableCell>
                              <TableCell className="text-xs text-muted-foreground leading-relaxed">
                                {rule.reason}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  )}

                  {/* Optimized Parameters Details */}
                  <details className="group">
                    <summary className="cursor-pointer text-sm font-semibold p-3 bg-secondary/50 backdrop-blur-sm rounded-lg hover:bg-secondary hover:border-primary/30 border border-border transition-all">
                      <span className="inline-flex items-center gap-2">
                        <span className="transform group-open:rotate-90 transition-transform text-primary">▶</span>
                        <span className="text-foreground">View Final Hyperparameters</span>
                      </span>
                    </summary>
                    <div className="mt-2 p-4 bg-secondary/50 backdrop-blur-sm rounded-lg border border-border">
                      <pre className="text-xs overflow-x-auto text-foreground">
                        {JSON.stringify(model.hyperparameter_tuning.best_params, null, 2)}
                      </pre>
                    </div>
                  </details>
                </div>
              )}

              {/* Train vs Test Comparison */}
              <div>
                <h3 className="font-semibold mb-3 flex items-center gap-2">
                  Overfitting Check
                  <Badge variant="outline" className="text-xs">Generalization</Badge>
                </h3>
                <div className="grid grid-cols-2 gap-4 mb-3">
                  <div className="p-5 bg-blue-500/10 border border-blue-500/20 rounded-lg shadow-sm">
                    <p className="text-sm font-medium text-blue-600 dark:text-blue-400 mb-1">Train Score</p>
                    <p className="text-3xl font-bold text-foreground">
                      {trainScore > 0 ? trainScore.toFixed(4) : 'N/A'}
                    </p>
                  </div>
                  <div className="p-5 bg-green-500/10 border border-green-500/20 rounded-lg shadow-sm">
                    <p className="text-sm font-medium text-green-600 dark:text-green-400 mb-1">Test Score</p>
                    <p className="text-3xl font-bold text-foreground">
                      {testScore > 0 ? testScore.toFixed(4) : 'N/A'}
                    </p>
                  </div>
                </div>
                {trainScore > 0 && testScore > 0 && getOverfittingIndicator(trainScore, testScore, isClassification)}
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Show message if no visualizations available */}
                {!model.confusion_matrix && !model.feature_importance && (
                  <div className="col-span-2 text-center py-10 bg-muted/30 border-2 border-dashed border-muted-foreground/30 rounded-lg">
                    <AlertCircle className="mx-auto h-14 w-14 mb-4 text-muted-foreground/50" />
                    <p className="font-semibold text-lg text-foreground">Advanced Visualizations Not Available</p>
                    <p className="text-sm mt-3 text-muted-foreground max-w-md mx-auto">
                      This training run was completed before visualization features were added.
                    </p>
                    <div className="mt-4 inline-flex items-center gap-2 px-4 py-2 bg-primary/10 text-primary rounded-md text-sm font-medium">
                      <CheckCircle2 className="h-4 w-4" />
                      Start a new training run to see confusion matrix & feature importance
                    </div>
                  </div>
                )}
                
                {/* Confusion Matrix */}
                {isClassification && model.confusion_matrix && (
                  <div>
                    <h3 className="font-semibold mb-3 flex items-center gap-2">
                      Confusion Matrix
                      <Badge variant="outline" className="text-xs">Predictions</Badge>
                    </h3>
                    <div className="bg-card border rounded-lg p-4 shadow-sm">
                      <div className="overflow-x-auto">
                        <table className="w-full border-collapse">
                          <thead>
                            <tr>
                              <th className="p-3 border border-border bg-muted/30"></th>
                              {model.confusion_matrix.labels.map((label) => (
                                <th key={`pred-${label}`} className="p-3 border border-border bg-blue-500/10 text-xs font-semibold">
                                  <div className="text-blue-600 dark:text-blue-400">Predicted</div>
                                  <div className="text-foreground mt-1">{label}</div>
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {model.confusion_matrix.matrix.map((row, i) => (
                              <tr key={`actual-${i}`}>
                                <th className="p-3 border border-border bg-purple-500/10 text-xs font-semibold">
                                  <div className="text-purple-600 dark:text-purple-400">Actual</div>
                                  <div className="text-foreground mt-1">{model.confusion_matrix!.labels[i]}</div>
                                </th>
                                {row.map((val, j) => {
                                  const total = row.reduce((a, b) => a + b, 0);
                                  const percentage = total > 0 ? ((val / total) * 100).toFixed(0) : '0';
                                  const isCorrect = i === j;
                                  return (
                                    <td
                                      key={`cell-${i}-${j}`}
                                      className={`p-3 border border-border text-center font-semibold transition-colors ${
                                        isCorrect 
                                          ? 'bg-green-500/20 hover:bg-green-500/30' 
                                          : val > 0 
                                          ? 'bg-red-500/20 hover:bg-red-500/30' 
                                          : 'bg-muted/30 hover:bg-muted/50'
                                      }`}
                                    >
                                      <div className="text-lg font-bold text-foreground">{val}</div>
                                      <div className="text-xs font-medium text-muted-foreground mt-0.5">({percentage}%)</div>
                                    </td>
                                  );
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <div className="flex items-center justify-center gap-4 mt-4 text-xs">
                        <div className="flex items-center gap-1.5">
                          <div className="w-3 h-3 rounded bg-green-500/20 border border-green-500/50"></div>
                          <span className="text-muted-foreground">Correct predictions</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <div className="w-3 h-3 rounded bg-red-500/20 border border-red-500/50"></div>
                          <span className="text-muted-foreground">Incorrect predictions</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Feature Importance */}
                {model.feature_importance && 
                 model.feature_importance.features && 
                 model.feature_importance.importance &&
                 model.feature_importance.features.length > 0 && (
                  <div>
                    <h3 className="font-semibold mb-3 flex items-center gap-2">
                      Feature Importance
                      <Badge variant="outline" className="text-xs">Top 10</Badge>
                    </h3>
                    <div className="bg-card border rounded-lg p-4 shadow-sm">
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart
                          data={model.feature_importance.features.slice(0, 10).map((feature, idx) => ({
                            feature: feature && feature.length > 20 ? feature.substring(0, 17) + '...' : feature || `Feature ${idx}`,
                            importance: model.feature_importance!.importance[idx] || 0
                          }))}
                          layout="vertical"
                          margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                          <XAxis type="number" className="text-xs" />
                          <YAxis dataKey="feature" type="category" width={90} style={{ fontSize: '12px' }} />
                          <Tooltip 
                            contentStyle={{ 
                              backgroundColor: 'hsl(var(--card))', 
                              border: '1px solid hsl(var(--border))',
                              borderRadius: '8px'
                            }}
                          />
                          <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                            {model.feature_importance!.features.slice(0, 10).map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={index < 3 ? '#22c55e' : '#8b5cf6'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                      <div className="flex items-center justify-center gap-4 mt-3 text-xs">
                        <div className="flex items-center gap-1.5">
                          <div className="w-3 h-3 rounded bg-green-500"></div>
                          <span className="text-muted-foreground">Top 3 features</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <div className="w-3 h-3 rounded bg-purple-500"></div>
                          <span className="text-muted-foreground">Other features</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
          );
        })}
      </div>
    );
  }

  // List View
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Training Runs</h2>
          <p className="text-sm text-muted-foreground">
            View and compare all training runs for this experiment
          </p>
        </div>
        <Button variant="outline" onClick={onBack}>
          Back to Training
        </Button>
      </div>

      {fetchError && (
        <Card className="border-destructive">
          <CardContent className="py-6 text-center">
            <p className="text-destructive">{fetchError}</p>
            <Button onClick={() => fetchRuns(page)} variant="outline" className="mt-4">
              Retry
            </Button>
          </CardContent>
        </Card>
      )}

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
        </div>
      ) : runs.length === 0 && !fetchError ? (
        <Card>
          <CardContent className="py-12 text-center">
            <p className="text-muted-foreground">No training runs yet. Start training to see results here.</p>
          </CardContent>
        </Card>
      ) : !fetchError && runs.length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle>All Runs ({runs.length})</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Run #</TableHead>
                  <TableHead>Date</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Duration</TableHead>
                  <TableHead>Best Model</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {runs.map((run) => (
                  <TableRow key={run.id}>
                    <TableCell className="font-medium">#{run.run_number}</TableCell>
                    <TableCell>{new Date(run.created_at).toLocaleDateString()}</TableCell>
                    <TableCell>{getStatusBadge(run.status)}</TableCell>
                    <TableCell>{formatDuration(run.duration_seconds)}</TableCell>
                    <TableCell>
                      {run.results_summary?.best_model ? (
                        <div>
                          <p className="font-medium">{run.results_summary.best_model.display_name}</p>
                          <p className="text-xs text-muted-foreground">
                            {run.results_summary.best_model.value.toFixed(3)}
                          </p>
                        </div>
                      ) : (
                        'N/A'
                      )}
                    </TableCell>
                    <TableCell>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => fetchRunDetails(run.id)}
                        disabled={detailsLoading}
                      >
                        <Eye className="w-4 h-4 mr-1" />
                        View
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between mt-4 pt-4 border-t">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={page === 1 || loading}
                >
                  Previous
                </Button>
                <span className="text-sm text-muted-foreground">
                  Page {page} of {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                  disabled={page === totalPages || loading}
                >
                  Next
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
};

export default ResultsStep;
