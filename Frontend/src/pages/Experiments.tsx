import { useState, useEffect } from "react";
import { Plus, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useExperiment } from "@/contexts/ExperimentContext";
import { useNavigate } from "react-router-dom";
import ExperimentCard from "@/components/experiments/ExperimentCard";
import CreateExperimentModal from "@/components/experiments/CreateExperimentModal";
import { experimentAPI } from "@/services/apiService";

const Experiments = () => {
  const { experiments, fetchExperiments, deleteExperiment, createExperiment } = useExperiment();
  const [searchQuery, setSearchQuery] = useState("");
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    fetchExperiments();
  }, []);

  const handleCreate = async (name: string, datasetId: string) => {
    const experiment = await createExperiment(name, datasetId);
    navigate(`/playground/${experiment.id}`);
  };

  const handleDuplicate = async (id: string) => {
    try {
      await experimentAPI.duplicate(id);
      fetchExperiments();
    } catch (error) {
      // Error handled in API service
    }
  };

  const filteredExperiments = experiments.filter(exp =>
    exp.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen py-8">
      <div className="container mx-auto px-4">
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-4xl font-bold gradient-text mb-2">Experiments</h1>
              <p className="text-muted-foreground">Manage your ML experiments</p>
            </div>
            <Button 
              onClick={() => setCreateModalOpen(true)}
              className="gradient-primary text-background gap-2"
            >
              <Plus className="w-4 h-4" />
              New Experiment
            </Button>
          </div>

          <div className="relative max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search experiments..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>

        {filteredExperiments.length === 0 ? (
          <div className="bg-card border border-border rounded-xl p-12 text-center">
            <div className="max-w-md mx-auto">
              <h3 className="text-xl font-semibold mb-2">
                {searchQuery ? "No experiments found" : "No experiments yet"}
              </h3>
              <p className="text-muted-foreground mb-6">
                {searchQuery 
                  ? "Try adjusting your search query"
                  : "Create your first experiment to start training ML models"}
              </p>
              {!searchQuery && (
                <Button 
                  onClick={() => setCreateModalOpen(true)}
                  className="gradient-primary text-background gap-2"
                >
                  <Plus className="w-4 h-4" />
                  Create Experiment
                </Button>
              )}
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {filteredExperiments.map((exp) => (
              <ExperimentCard
                key={exp.id}
                experiment={exp}
                onDelete={deleteExperiment}
                onDuplicate={handleDuplicate}
              />
            ))}
          </div>
        )}
      </div>

      <CreateExperimentModal
        open={createModalOpen}
        onOpenChange={setCreateModalOpen}
        onCreate={handleCreate}
      />
    </div>
  );
};

export default Experiments;
