import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Save } from "lucide-react";
import { useExperiment } from "@/contexts/ExperimentContext";
import { useEDA } from "@/hooks/useEDA";
import DataAnalysisStep from "@/components/playground/DataAnalysisStep";
import ModelConfigStep from "@/components/playground/ModelConfigStep";
import TrainingStep from "@/components/playground/TrainingStep";
import ResultsStep from "@/components/playground/ResultsStep";
import PredictionStep from "@/components/playground/PredictionStep";
import { DataScienceAssistant } from "@/components/experiments/DataScienceAssistant";
import { toast } from "sonner";

const Playground = () => {
  const { experimentId } = useParams();
  const navigate = useNavigate();
  const { currentExperiment, loadExperiment, saveExperimentConfig } = useExperiment();
  const [activeTab, setActiveTab] = useState("analysis");
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  
  // Load EDA data for the experiment's dataset
  const { edaData, isLoading: edaLoading } = useEDA(currentExperiment?.datasetId);
  
  // Refresh experiment data when switching to config or training tabs to ensure fresh data
  useEffect(() => {
    if (experimentId && (activeTab === "config" || activeTab === "training")) {
      loadExperiment(experimentId);
    }
  }, [activeTab, experimentId]);

  useEffect(() => {
    if (experimentId) {
      loadExperiment(experimentId);
    } else {
      navigate("/experiments");
    }
  }, [experimentId]);
  
  const handleSave = async () => {
    if (!currentExperiment) return;
    
    try {
      await saveExperimentConfig(currentExperiment.config);
      setHasUnsavedChanges(false);
    } catch (error) {
      // Error handled in context
    }
  };

  const tabs = [
    { value: "analysis", label: "1. Data Analysis", component: DataAnalysisStep },
    { value: "config", label: "2. Model Config", component: ModelConfigStep },
    { value: "training", label: "3. Training", component: TrainingStep },
    { value: "results", label: "4. Results", component: ResultsStep },
    { value: "predict", label: "5. Prediction", component: PredictionStep }
  ];

  if (!currentExperiment) {
    return (
      <div className="min-h-screen py-8 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-2">Loading experiment...</h2>
          <p className="text-muted-foreground">Please wait</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="min-h-screen py-8">
      <div className="container mx-auto px-4">
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-4">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => navigate("/experiments")}
              >
                <ArrowLeft className="w-5 h-5" />
              </Button>
              <div>
                <h1 className="text-4xl font-bold gradient-text">{currentExperiment.name}</h1>
                <p className="text-muted-foreground">
                  Dataset: {currentExperiment.datasetName || "Loading..."}
                </p>
              </div>
            </div>
            <Button
              onClick={handleSave}
              disabled={!hasUnsavedChanges}
              variant="outline"
              className="gap-2"
            >
              <Save className="w-4 h-4" />
              Save Progress
            </Button>
          </div>
        </div>
        
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid grid-cols-5 w-full bg-card border border-border p-1 h-auto">
            {tabs.map((tab) => (
              <TabsTrigger
                key={tab.value}
                value={tab.value}
                className="data-[state=active]:bg-primary data-[state=active]:text-background py-3 text-xs sm:text-sm"
              >
                {tab.label}
              </TabsTrigger>
            ))}
          </TabsList>
          
          <TabsContent value="analysis" className="animate-fade-in">
            <DataAnalysisStep 
              onNext={() => setActiveTab("config")}
            />
          </TabsContent>
          
          <TabsContent value="config" className="animate-fade-in">
            {edaLoading ? (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
                  <p className="text-muted-foreground">Loading dataset information...</p>
                </div>
              </div>
            ) : edaData ? (
              <ModelConfigStep 
                experiment={currentExperiment}
                edaData={edaData}
                onNext={() => setActiveTab("training")}
                onBack={() => setActiveTab("analysis")}
              />
            ) : (
              <div className="text-center py-12">
                <p className="text-muted-foreground">Please complete Data Analysis first</p>
                <Button onClick={() => setActiveTab("analysis")} className="mt-4">
                  Go to Data Analysis
                </Button>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="training" className="animate-fade-in">
            {currentExperiment?.config?.taskType && currentExperiment?.config?.targetColumn && currentExperiment?.config?.selectedFeatures ? (
              <TrainingStep 
                experimentId={currentExperiment.id}
                experimentConfig={currentExperiment.config}
                onComplete={() => {
                  // Training complete, move to results
                  setActiveTab("results");
                }}
              />
            ) : (
              <div className="text-center py-12">
                <p className="text-muted-foreground">Please complete Model Configuration first</p>
                <Button onClick={() => setActiveTab("config")} className="mt-4">
                  Go to Model Config
                </Button>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="results" className="animate-fade-in">
            <ResultsStep 
              experimentId={currentExperiment.id}
              onBack={() => setActiveTab("training")}
            />
          </TabsContent>
          
          <TabsContent value="predict" className="animate-fade-in">
            <PredictionStep />
          </TabsContent>
        </Tabs>
      </div>

      {/* AI Assistant - Always available across all phases */}
      <DataScienceAssistant
        datasetId={currentExperiment?.datasetId}
        edaData={edaData}
        currentPhase={activeTab as 'analysis' | 'config' | 'training' | 'results' | 'predict'}
        experimentConfig={currentExperiment?.config}
        trainingData={currentExperiment?.trainingRuns}
        resultsData={currentExperiment?.results}
      />
    </div>
  );
};

export default Playground;
