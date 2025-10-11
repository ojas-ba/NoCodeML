import { useState } from "react";
import { Sparkles, Upload, Download, AlertCircle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import { useExperiment } from "@/contexts/ExperimentContext";
import { predictionAPI } from "@/services/apiService";

const PredictionStep = () => {
  const { currentExperiment } = useExperiment();
  const { toast } = useToast();
  
  const [loading, setLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [batchResult, setBatchResult] = useState<any>(null);
  const [inputValues, setInputValues] = useState<Record<string, any>>({});
  const [error, setError] = useState<string | null>(null);
  const [predictionHistory, setPredictionHistory] = useState<any[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  
  // Get feature names from experiment config
  const selectedFeatures = currentExperiment?.config?.selectedFeatures || [];
  
  const handleSinglePredict = async () => {
    if (!currentExperiment) {
      setError("No experiment selected");
      return;
    }
    
    // Validate all features are filled
    const missingFeatures = selectedFeatures.filter(feature => !inputValues[feature]);
    if (missingFeatures.length > 0) {
      setError(`Please fill in all features: ${missingFeatures.join(", ")}`);
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      // Convert values to numbers where appropriate
      const features = Object.fromEntries(
        Object.entries(inputValues).map(([key, value]) => [key, isNaN(Number(value)) ? value : Number(value)])
      );
      
      const result = await predictionAPI.single(currentExperiment.id, features);
      setPredictionResult(result);
      
      toast({ 
        title: "Prediction Complete!", 
        description: `Result: ${result.prediction}` 
      });
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || "Unknown error";
      setError(errorMsg);
      toast({ 
        title: "Prediction Failed", 
        description: errorMsg,
        variant: "destructive" 
      });
    } finally {
      setLoading(false);
    }
  };
  
  const handleBatchPredict = async (file: File) => {
    if (!currentExperiment) {
      setError("No experiment selected");
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await predictionAPI.batch(currentExperiment.id, file);
      setBatchResult(result);
      
      toast({ 
        title: "Batch Prediction Complete!", 
        description: `${result.total_predictions} predictions made` 
      });
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || "Unknown error";
      setError(errorMsg);
      toast({ 
        title: "Batch Prediction Failed", 
        description: errorMsg,
        variant: "destructive" 
      });
    } finally {
      setLoading(false);
    }
  };
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.name.endsWith('.csv')) {
        setError("Please upload a CSV file");
        return;
      }
      handleBatchPredict(file);
    }
  };
  
  const handleDownload = async (predictionId?: string) => {
    const id = predictionId || batchResult?.prediction_id;
    if (id) {
      try {
        await predictionAPI.download(id);
      } catch (err) {
        console.error('Download failed:', err);
        setError('Failed to download predictions. Please try again.');
      }
    }
  };
  
  const loadPredictionHistory = async () => {
    if (!currentExperiment) return;
    
    try {
      const result = await predictionAPI.getHistory(currentExperiment.id);
      setPredictionHistory(result.predictions || []);
      setShowHistory(true);
    } catch (err) {
      console.error('Failed to load history:', err);
    }
  };
  
  if (!currentExperiment) {
    return (
      <div className="flex items-center justify-center h-64">
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            No experiment selected. Please create an experiment first.
          </AlertDescription>
        </Alert>
      </div>
    );
  }
  
  return (
    <div className="space-y-6">
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      
      <div className="bg-card border border-border rounded-xl p-6">
        <h2 className="text-2xl font-bold mb-6 flex items-center">
          <Sparkles className="w-6 h-6 mr-2 text-primary" />
          Make Predictions
        </h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Single Prediction */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Single Prediction</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Enter feature values to get a prediction from your trained model
            </p>
            
            <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
              {selectedFeatures.length === 0 ? (
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    No features configured. Please configure your experiment first.
                  </AlertDescription>
                </Alert>
              ) : (
                selectedFeatures.map((feature: string) => (
                  <div key={feature}>
                    <Label>{feature}</Label>
                    <Input
                      type="text"
                      placeholder={`Enter ${feature}`}
                      value={inputValues[feature] || ''}
                      onChange={(e) => setInputValues({...inputValues, [feature]: e.target.value})}
                      disabled={loading}
                    />
                  </div>
                ))
              )}
            </div>
            
            <Button 
              onClick={handleSinglePredict} 
              disabled={loading || selectedFeatures.length === 0}
              className="w-full gradient-primary text-background"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Predicting...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 mr-2" />
                  Predict
                </>
              )}
            </Button>
          </div>
          
          {/* Batch Prediction */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Batch Prediction</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Upload a CSV file with the same features as your training data
            </p>
            
            <div className="border-2 border-dashed border-border rounded-lg p-8 text-center space-y-4">
              <Upload className="w-12 h-12 mx-auto text-primary" />
              <div>
                <p className="font-medium mb-1">Upload CSV for batch predictions</p>
                <p className="text-sm text-muted-foreground">
                  File must contain columns: {selectedFeatures.join(", ")}
                </p>
              </div>
              
              <div className="relative">
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  disabled={loading}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  id="batch-file-input"
                />
                <Button 
                  variant="outline" 
                  disabled={loading}
                  asChild
                >
                  <label htmlFor="batch-file-input" className="cursor-pointer">
                    {loading ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Upload className="w-4 h-4 mr-2" />
                        Choose CSV File
                      </>
                    )}
                  </label>
                </Button>
              </div>
            </div>
            
            {batchResult && (
              <div className="mt-4 p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
                <h4 className="font-bold mb-2 text-green-700 dark:text-green-400">
                  ✓ Batch Prediction Complete!
                </h4>
                <p className="text-sm">
                  <strong>{batchResult.total_predictions}</strong> predictions generated successfully
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Single Prediction Result */}
      {predictionResult && (
        <div className="bg-gradient-to-r from-primary/10 via-primary-purple/10 to-primary-blue/10 border border-primary/30 rounded-xl p-6 animate-fade-in">
          <h3 className="text-xl font-bold mb-4">Prediction Result</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Prediction Value */}
            <div className="text-center p-6 bg-card rounded-lg shadow-sm">
              <p className="text-sm text-muted-foreground mb-2">Prediction</p>
              <p className="text-4xl font-bold text-primary">{predictionResult.prediction}</p>
            </div>
            
            {/* Confidence Score */}
            {predictionResult.confidence !== null && predictionResult.confidence !== undefined && (
              <div className="text-center p-6 bg-card rounded-lg shadow-sm">
                <p className="text-sm text-muted-foreground mb-2">Confidence</p>
                <p className="text-4xl font-bold">
                  {(predictionResult.confidence * 100).toFixed(1)}%
                </p>
                <div className="mt-3 w-full bg-muted rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-primary to-primary-purple h-2 rounded-full transition-all"
                    style={{ width: `${predictionResult.confidence * 100}%` }}
                  />
                </div>
              </div>
            )}
            
            {/* Class Probabilities */}
            {predictionResult.probabilities && (
              <div className="text-center p-6 bg-card rounded-lg shadow-sm">
                <p className="text-sm text-muted-foreground mb-3">Class Probabilities</p>
                <div className="space-y-2 text-sm">
                  {Object.entries(predictionResult.probabilities).map(([key, value]) => (
                    <div key={key} className="flex justify-between items-center">
                      <span className="font-medium">{key}:</span>
                      <div className="flex items-center gap-2">
                        <div className="w-20 bg-muted rounded-full h-1.5">
                          <div 
                            className="bg-primary h-1.5 rounded-full"
                            style={{ width: `${(value as number) * 100}%` }}
                          />
                        </div>
                        <span className="font-semibold w-12 text-right">
                          {((value as number) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          {/* Input Summary */}
          <div className="mt-6 p-4 bg-card/50 rounded-lg">
            <p className="text-sm font-semibold mb-2 text-muted-foreground">Input Features:</p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
              {Object.entries(inputValues).map(([key, value]) => (
                <div key={key} className="flex justify-between">
                  <span className="text-muted-foreground">{key}:</span>
                  <span className="font-medium ml-2">{value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
      
      {/* Prediction History */}
      <div className="bg-card border border-border rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center">
            <Download className="w-5 h-5 mr-2 text-primary" />
            Prediction History
          </h3>
          <Button 
            onClick={loadPredictionHistory}
            variant="outline"
            size="sm"
          >
            {showHistory ? 'Refresh' : 'Load History'}
          </Button>
        </div>
        
        {showHistory && (
          <div className="space-y-2">
            {predictionHistory.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-4">
                No prediction history yet. Make batch predictions to see them here.
              </p>
            ) : (
              predictionHistory.map((pred) => (
                <div 
                  key={pred.id} 
                  className="flex items-center justify-between p-3 bg-card/50 rounded-lg border border-border hover:border-primary/50 transition-colors"
                >
                  <div className="flex-1">
                    <p className="text-sm font-medium">
                      {pred.total_predictions} predictions
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {new Date(pred.created_at).toLocaleString()}
                    </p>
                  </div>
                  <Button
                    onClick={() => handleDownload(pred.id)}
                    size="sm"
                    variant="ghost"
                  >
                    <Download className="w-4 h-4 mr-1" />
                    Download
                  </Button>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionStep;
