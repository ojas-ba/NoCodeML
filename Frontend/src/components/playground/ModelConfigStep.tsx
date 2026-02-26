import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { Settings, Target, AlertCircle, CheckCircle2, Loader2, Sparkles, TrendingUp, Zap, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Slider } from "@/components/ui/slider";
import { FeatureSelector } from "./FeatureSelector";
import { experimentAPI } from "@/services/apiService";
import { useToast } from "@/hooks/use-toast";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { ExperimentConfig, ModelConfig, ColumnInfo, EDAResponse } from '@/types/experiment';
import { useModels } from "@/contexts/ModelsContext";
import { useExperiment } from "@/contexts/ExperimentContext";

interface ModelConfigStepProps {
  experiment: any;
  edaData: any;
  onNext: () => void;
  onBack?: () => void;
}

const ModelConfigStep = ({ experiment, edaData, onNext, onBack }: ModelConfigStepProps) => {
  const { toast } = useToast();
  const { updateExperiment } = useExperiment();
  const { getModelsByTask, loading: modelsLoading } = useModels();
  const [isSaving, setIsSaving] = useState(false);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  
  // Config state
  const [taskType, setTaskType] = useState<"classification" | "regression">(
    experiment?.config?.taskType || "classification"
  );
  const [targetColumn, setTargetColumn] = useState(
    experiment?.config?.targetColumn || ""
  );
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>(
    experiment?.config?.selectedFeatures || []
  );
  const [trainSplit, setTrainSplit] = useState([
    experiment?.config?.trainTestSplit ? experiment.config.trainTestSplit * 100 : 80
  ]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [optimizationEnabled, setOptimizationEnabled] = useState<boolean>(
    experiment?.config?.enableOptimization || false
  );
  
  // Get available columns from EDA (memoized to prevent re-renders)
  const allColumns = useMemo(() => edaData?.columns || [], [edaData?.columns]);
  const numericColumns = useMemo(() => 
    edaData?.numeric_columns?.filter(
      (col: string) => !edaData.id_columns.includes(col)
    ) || []
  , [edaData?.numeric_columns, edaData?.id_columns]);
  
  const categoricalColumns = useMemo(() => 
    edaData?.categorical_columns?.filter(
      (col: string) => !edaData.id_columns.includes(col)
    ) || []
  , [edaData?.categorical_columns, edaData?.id_columns]);
  
  const idColumns = useMemo(() => edaData?.id_columns || [], [edaData?.id_columns]);
  const availableColumns = useMemo(() => 
    [...numericColumns, ...categoricalColumns]
  , [numericColumns, categoricalColumns]);

  // Valid target columns based on task type
  const validTargetColumns = useMemo(() => {
    if (taskType === "classification") {
      // Classification: only categorical columns
      return categoricalColumns;
    } else {
      // Regression: only numerical columns
      return numericColumns;
    }
  }, [taskType, categoricalColumns, numericColumns]);

  // Available features (exclude target column and ID columns)
  const availableFeatures = useMemo(() => {
    return availableColumns.filter(col => col !== targetColumn);
  }, [availableColumns, targetColumn]);
  
  // Get models from the global context based on task type
  const models = useMemo(() => {
    const modelsData = getModelsByTask(taskType);
    return modelsData.map(model => ({
      id: model.model_type,
      name: model.display_name,
      description: model.description
    }));
  }, [taskType, getModelsByTask]);
  
  // Sync state with experiment config when it changes (e.g., after save or tab switch)
  useEffect(() => {
    if (experiment?.config) {
      // Update task type if different
      if (experiment.config.taskType && experiment.config.taskType !== taskType) {
        setTaskType(experiment.config.taskType);
      }
      
      // Update target column if different
      if (experiment.config.targetColumn && experiment.config.targetColumn !== targetColumn) {
        setTargetColumn(experiment.config.targetColumn);
      }
      
      // Update selected features if different
      if (experiment.config.selectedFeatures && experiment.config.selectedFeatures.length > 0) {
        setSelectedFeatures(experiment.config.selectedFeatures);
      } else if (selectedFeatures.length === 0 && availableColumns.length > 0) {
        // Auto-select all non-ID columns by default if no features selected yet
        setSelectedFeatures(availableColumns);
      }
      
      // Update train split if different
      if (experiment.config.trainTestSplit && experiment.config.trainTestSplit !== trainSplit[0] / 100) {
        setTrainSplit([experiment.config.trainTestSplit * 100]);
      }
      
      // Update selected models if different
      if (experiment.config.models && experiment.config.models.length > 0) {
        const modelTypes = experiment.config.models.map((m: any) => m.model_type);
        if (JSON.stringify(modelTypes) !== JSON.stringify(selectedModels)) {
          setSelectedModels(modelTypes);
        }
      }
    }
  }, [experiment?.config, availableColumns.length]);

  // Clear invalid models and target when task type changes
  const prevTaskType = useRef(taskType);
  useEffect(() => {
    if (prevTaskType.current !== taskType && prevTaskType.current !== undefined) {
      // Clear invalid models
      const availableModelIds = models.map(m => m.id);
      const validModels = selectedModels.filter(modelId => 
        availableModelIds.includes(modelId)
      );
      
      if (validModels.length !== selectedModels.length) {
        setSelectedModels(validModels);
        
        if (selectedModels.length > 0 && validModels.length === 0) {
          toast({
            title: "Model Selection Cleared",
            description: `Previous model selections were cleared because they're not available for ${taskType}.`,
          });
        }
      }

      // Clear target if it's not valid for new task type
      if (targetColumn && !validTargetColumns.includes(targetColumn)) {
        setTargetColumn("");
        toast({
          title: "Target Column Cleared",
          description: `The previous target column is not valid for ${taskType}. Please select a ${taskType === "classification" ? "categorical" : "numerical"} column.`,
        });
      }
    }
    prevTaskType.current = taskType;
  }, [taskType, models, selectedModels, validTargetColumns, targetColumn, toast]);
  
  // Handle target column change
  const handleTargetChange = (value: string) => {
    setTargetColumn(value);
    
    // Automatically remove target from selected features
    if (selectedFeatures.includes(value)) {
      setSelectedFeatures(prev => prev.filter(f => f !== value));
      toast({
        title: "Feature Removed",
        description: "Target column was automatically removed from selected features.",
      });
    }
  };

  const toggleModel = (modelId: string) => {
    setSelectedModels(prev =>
      prev.includes(modelId)
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    );
  };
  
  const handleFeatureToggle = (feature: string) => {
    // Prevent selecting target as feature
    if (feature === targetColumn) {
      toast({
        title: "Invalid Selection",
        description: "Target column cannot be used as a feature.",
        variant: "destructive"
      });
      return;
    }

    setSelectedFeatures(prev =>
      prev.includes(feature)
        ? prev.filter(f => f !== feature)
        : [...prev, feature]
    );
  };
  
  const handleSelectAllFeatures = () => {
    // Exclude target column from selection
    setSelectedFeatures(availableFeatures);
  };
  
  const handleClearAllFeatures = () => {
    setSelectedFeatures([]);
  };

  const handleSelectRecommended = () => {
    const recommended: string[] = [];
    
    // All numeric columns (excluding target)
    numericColumns.forEach(col => {
      if (col !== targetColumn) {
        recommended.push(col);
      }
    });
    
    // Categorical with reasonable cardinality and low missing (excluding target)
    categoricalColumns.forEach(col => {
      if (col !== targetColumn) {
        const info = allColumns.find((c: any) => c.name === col);
        if (info && info.unique_count <= 50 && info.missing_percent < 50) {
          recommended.push(col);
        }
      }
    });
    
    setSelectedFeatures(recommended);
  };
  
  // Track unsaved changes
  useEffect(() => {
    setHasUnsavedChanges(true);
  }, [taskType, targetColumn, selectedFeatures.length, trainSplit, selectedModels.length]);
  
  // Clear validation errors when user makes changes
  useEffect(() => {
    if (validationErrors.length > 0) {
      setValidationErrors([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskType, targetColumn, selectedFeatures.length, selectedModels.length]);

  // Manual save function
  const handleSaveConfig = async () => {
    // Comprehensive frontend validation
    const errors: string[] = [];
    
    if (!taskType) {
      errors.push("Task type is required");
    }
    
    if (!targetColumn) {
      errors.push("Target column is required");
    } else {
      // Validate target column is appropriate for task type
      if (taskType === "classification" && !categoricalColumns.includes(targetColumn)) {
        errors.push("For classification, target must be a categorical column");
      } else if (taskType === "regression" && !numericColumns.includes(targetColumn)) {
        errors.push("For regression, target must be a numerical column");
      }
      
      // Check target is not in features
      if (selectedFeatures.includes(targetColumn)) {
        errors.push("Target column cannot be a feature");
      }
    }
    
    if (selectedFeatures.length === 0) {
      errors.push("Select at least one feature");
    }
    
    // Check for ID columns in features
    const idFeaturesSelected = selectedFeatures.filter(f => idColumns.includes(f));
    if (idFeaturesSelected.length > 0) {
      errors.push(`Warning: ID columns detected in features: ${idFeaturesSelected.join(", ")}. Consider removing them.`);
    }
    
    if (selectedModels.length === 0) {
      errors.push("Select at least one model");
    }
    
    if (errors.length > 0) {
      setValidationErrors(errors);
      toast({
        title: "Validation Error",
        description: "Please fix the errors before saving",
        variant: "destructive"
      });
      return;
    }
    
    setValidationErrors([]);
    setIsSaving(true);
    
    try {
      const featureTypes = {
        numerical: selectedFeatures.filter(f => numericColumns.includes(f)),
        categorical: selectedFeatures.filter(f => categoricalColumns.includes(f))
      };
      
      const modelConfigs = selectedModels.map(modelType => {
        const modelInfo = models.find(m => m.id === modelType);
        return {
          model_type: modelType,
          display_name: modelInfo?.name || modelType,
          preset: "default", // Changed from "balanced" to valid preset
          hyperparameters: {},
          custom_hyperparameters: null
        };
      });
      
      const config = {
        taskType,
        targetColumn,
        selectedFeatures,
        featureTypes,
        excludedColumns: idColumns,
        trainTestSplit: trainSplit[0] / 100,
        randomSeed: 42,
        models: modelConfigs,
        enableOptimization: optimizationEnabled
      };
      
      // Use context to update experiment - this updates both backend AND context state
      await updateExperiment(experiment.id, { config });
      
      setHasUnsavedChanges(false);
      toast({
        title: "Success",
        description: "Configuration saved successfully"
      });
    } catch (error: any) {
      console.error("Save failed:", error);
      
      // Handle different error response structures
      const errorDetail = error.response?.data?.detail;
      
      // Parse validation errors from FastAPI/Pydantic
      let errors: string[] = [];
      
      if (Array.isArray(errorDetail)) {
        // Pydantic validation errors format: [{type, loc, msg, input, ctx}, ...]
        errors = errorDetail.map(err => {
          if (typeof err === 'object' && err.msg) {
            const location = err.loc ? ` (${err.loc.join(' -> ')})` : '';
            return `${err.msg}${location}`;
          }
          return typeof err === 'string' ? err : JSON.stringify(err);
        });
      } else if (typeof errorDetail === 'string') {
        errors = [errorDetail];
      } else if (typeof errorDetail === 'object' && errorDetail?.message) {
        errors = [errorDetail.message];
      } else {
        errors = [error.message || "Failed to save configuration"];
      }
      
      setValidationErrors(errors);
      toast({
        title: "Error",
        description: errors[0] || "Failed to save configuration",
        variant: "destructive"
      });
    } finally {
      setIsSaving(false);
    }
  };

  const validateAndContinue = async () => {
    // Save first, then continue
    await handleSaveConfig();
    
    // If there are validation errors after save, don't continue
    if (validationErrors.length > 0) {
      return;
    }
    
    // Continue to next step
    onNext();
  };
  
  return (
    <div className="space-y-6">
      {/* Validation Errors */}
      {validationErrors.length > 0 && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <ul className="list-disc list-inside">
              {validationErrors.map((error, idx) => (
                <li key={idx}>{error}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}
      
      {/* Basic Configuration */}
      <div className="bg-card border border-border rounded-xl p-6">
        <h2 className="text-2xl font-bold mb-6 flex items-center">
          <Settings className="w-6 h-6 mr-2 text-primary" />
          Model Configuration
        </h2>
        
        <div className="space-y-6">
          {/* Task Type */}
          <div>
            <Label className="text-base mb-2 block">Task Type</Label>
            <Select value={taskType} onValueChange={(v: "classification" | "regression") => setTaskType(v)}>
              <SelectTrigger>
                <SelectValue placeholder="Select task type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="classification">Classification</SelectItem>
                <SelectItem value="regression">Regression</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          {/* Target Variable */}
          <div>
            <Label className="text-base mb-2 flex items-center">
              <Target className="w-4 h-4 mr-2 text-primary" />
              Target Variable
            </Label>
            <Select value={targetColumn} onValueChange={handleTargetChange}>
              <SelectTrigger>
                <SelectValue placeholder={`Select ${taskType === "classification" ? "categorical" : "numerical"} column`} />
              </SelectTrigger>
              <SelectContent>
                {validTargetColumns.length > 0 ? (
                  validTargetColumns.map((col: string) => {
                    const colInfo = allColumns.find((c: any) => c.name === col);
                    return (
                      <SelectItem key={col} value={col}>
                        <div className="flex items-center gap-2">
                          <span>{col}</span>
                          {colInfo && (
                            <span className="text-xs text-muted-foreground">
                              ({colInfo.dtype})
                            </span>
                          )}
                        </div>
                      </SelectItem>
                    );
                  })
                ) : (
                  <SelectItem value="__no_columns__" disabled>
                    No valid {taskType === "classification" ? "categorical" : "numerical"} columns available
                  </SelectItem>
                )}
              </SelectContent>
            </Select>
            {targetColumn && (
              <p className="text-xs text-muted-foreground mt-1">
                This is the column you want to predict
              </p>
            )}
            {validTargetColumns.length === 0 && (
              <p className="text-xs text-red-500 mt-1">
                ⚠️ No valid target columns for {taskType}. Try switching task type.
              </p>
            )}
          </div>
          
          {/* Train/Test Split */}
          <div>
            <Label className="text-base mb-2 block">
              Train/Test Split: {trainSplit[0]}% / {100 - trainSplit[0]}%
            </Label>
            <Slider
              value={trainSplit}
              onValueChange={setTrainSplit}
              min={60}
              max={90}
              step={5}
              className="mt-2"
            />
            <p className="text-xs text-muted-foreground mt-1">
              {trainSplit[0]}% of data will be used for training, {100 - trainSplit[0]}% for testing
            </p>
          </div>
        </div>
      </div>
      
      {/* Feature Selection */}
      <FeatureSelector
        numericColumns={numericColumns.filter(col => col !== targetColumn)}
        categoricalColumns={categoricalColumns.filter(col => col !== targetColumn)}
        idColumns={idColumns}
        allColumns={allColumns}
        selectedFeatures={selectedFeatures}
        onFeatureToggle={handleFeatureToggle}
        onSelectAll={handleSelectAllFeatures}
        onClearAll={handleClearAllFeatures}
        onSelectRecommended={handleSelectRecommended}
        targetColumn={targetColumn}
      />
      
      {/* Model Selection */}
      <div className="bg-card border border-border rounded-xl p-6">
        <h3 className="text-xl font-bold mb-4">Select Models to Train</h3>
        {modelsLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin mr-2" />
            <span>Loading available models...</span>
          </div>
        ) : models.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <AlertCircle className="h-8 w-8 mx-auto mb-2" />
            No models available for {taskType}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {models.map(model => (
              <div
                key={model.id}
                className={`p-4 rounded-lg border-2 transition-all cursor-pointer ${
                  selectedModels.includes(model.id)
                    ? "border-primary bg-primary/10"
                    : "border-border hover:border-primary/50"
                }`}
                onClick={() => toggleModel(model.id)}
                title={model.description}
              >
                <div className="flex items-center space-x-3">
                  <Checkbox
                    checked={selectedModels.includes(model.id)}
                    onCheckedChange={() => toggleModel(model.id)}
                  />
                  <div className="flex-1">
                    <span className="font-medium">{model.name}</span>
                    {model.description && (
                      <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                        {model.description}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
        <p className="text-sm text-muted-foreground mt-4">
          {selectedModels.length} model{selectedModels.length !== 1 ? 's' : ''} selected
        </p>
      </div>

      {/* Hyperparameter Optimization Card */}
      <Card className="border-2 border-primary/30 bg-gradient-to-br from-card via-card to-accent/5">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-primary via-accent to-primary-blue rounded-lg shadow-lg shadow-primary/20">
                <Sparkles className="w-5 h-5 text-background" />
              </div>
              <div>
                <CardTitle className="text-lg bg-gradient-to-r from-primary via-accent to-primary-blue bg-clip-text text-transparent">
                  Expert System Optimization
                </CardTitle>
                <CardDescription className="text-sm mt-1">
                  Transparent, rule-based parameter tuning with full reasoning visibility
                </CardDescription>
              </div>
            </div>
            <Switch
              checked={optimizationEnabled}
              onCheckedChange={setOptimizationEnabled}
              className="data-[state=checked]:bg-primary"
            />
          </div>
        </CardHeader>

        {optimizationEnabled && (
          <CardContent className="space-y-4 pt-0">
            <Alert className="bg-gradient-to-r from-primary/10 via-accent/10 to-primary-blue/10 border-primary/30">
              <Info className="h-4 w-4 text-primary" />
              <AlertDescription className="text-sm">
                <strong className="text-primary">Expert System enabled:</strong> Parameters will be adjusted by our 
                rules engine based on your dataset characteristics, with every decision 
                explained transparently.
              </AlertDescription>
            </Alert>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex items-start gap-3 p-4 bg-card/50 backdrop-blur-sm rounded-lg border border-border hover:border-primary/50 transition-colors">
                <TrendingUp className="w-5 h-5 text-success mt-0.5 flex-shrink-0" />
                <div>
                  <div className="font-semibold text-sm text-foreground">Dataset-Aware</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Automatically adjusts for dataset size, features, and class imbalance
                  </div>
                </div>
              </div>

              <div className="flex items-start gap-3 p-4 bg-card/50 backdrop-blur-sm rounded-lg border border-border hover:border-warning/50 transition-colors">
                <Zap className="w-5 h-5 text-warning mt-0.5 flex-shrink-0" />
                <div>
                  <div className="font-semibold text-sm text-foreground">Overfitting Prevention</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Applies regularization and complexity controls based on data
                  </div>
                </div>
              </div>

              <div className="flex items-start gap-3 p-4 bg-card/50 backdrop-blur-sm rounded-lg border border-border hover:border-accent/50 transition-colors">
                <Sparkles className="w-5 h-5 text-accent mt-0.5 flex-shrink-0" />
                <div>
                  <div className="font-semibold text-sm text-foreground">Research-Backed</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Uses proven parameters from academic research and competitions
                  </div>
                </div>
              </div>
            </div>

            <div className="text-xs text-muted-foreground bg-secondary/50 backdrop-blur-sm p-3 rounded-lg border border-border">
              <strong className="text-foreground">How it works:</strong> Our transparent Expert System uses research-backed 
              heuristic rules to analyze your dataset's characteristics (size, features, class balance) 
              and adjusts each hyperparameter with a clear, step-by-step explanation of every decision made.
            </div>
          </CardContent>
        )}
      </Card>
      
      {/* Summary */}
      <div className="bg-gradient-to-r from-primary/10 via-primary-purple/10 to-primary-blue/10 border border-primary/30 rounded-xl p-6">
        <h3 className="text-lg font-bold mb-3">Configuration Summary</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-muted-foreground">Task</p>
            <p className="font-semibold capitalize">{taskType || "Not set"}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Target</p>
            <p className="font-semibold">{targetColumn || "Not set"}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Features</p>
            <p className="font-semibold">{selectedFeatures.length}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Models</p>
            <p className="font-semibold">{selectedModels.length}</p>
          </div>
        </div>
      </div>
      
      <div className="flex justify-between items-center">
        {onBack && (
          <Button variant="outline" onClick={onBack}>
            Back to Data Analysis
          </Button>
        )}
        <div className="flex gap-3 ml-auto">
          <Button 
            variant="outline"
            onClick={handleSaveConfig}
            disabled={isSaving || selectedModels.length === 0 || selectedFeatures.length === 0 || !targetColumn}
          >
            {isSaving ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                {hasUnsavedChanges && <span className="w-2 h-2 bg-orange-500 rounded-full mr-2" />}
                Save Configuration
              </>
            )}
          </Button>
          <Button 
            onClick={validateAndContinue}
            disabled={isSaving || selectedModels.length === 0 || selectedFeatures.length === 0 || !targetColumn || hasUnsavedChanges}
            className="gradient-primary text-background"
          >
            Continue to Training
          </Button>
        </div>
      </div>
    </div>
  );
};

export default ModelConfigStep;
