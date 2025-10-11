import { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { datasetAPI } from "@/services/apiService";
import { toast } from "sonner";

interface CreateExperimentModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreate: (name: string, datasetId: string) => Promise<void>;
  preselectedDatasetId?: string;
}

const CreateExperimentModal = ({ open, onOpenChange, onCreate, preselectedDatasetId }: CreateExperimentModalProps) => {
  const [name, setName] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [datasets, setDatasets] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  // Handle modal open/close and preselected dataset
  useEffect(() => {
    if (open) {
      // Set preselected dataset when modal opens
      if (preselectedDatasetId) {
        setDatasetId(preselectedDatasetId);
      } else {
        setDatasetId("");
      }
      loadDatasets();
    } else {
      // Reset form when modal closes
      setName("");
      setDatasetId("");
    }
  }, [open, preselectedDatasetId]);

  const loadDatasets = async () => {
    setLoading(true);
    try {
      const data = await datasetAPI.list();
      setDatasets(data);
    } catch (error: any) {
      toast.error(error.message || "Failed to load datasets");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!name.trim()) {
      toast.error("Please enter an experiment name");
      return;
    }
    
    if (!datasetId) {
      toast.error("Please select a dataset");
      return;
    }

    setSubmitting(true);
    try {
      await onCreate(name.trim(), datasetId);
      setName("");
      setDatasetId("");
      onOpenChange(false);
    } catch (error) {
      // Error handled in onCreate
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <form onSubmit={handleSubmit}>
          <DialogHeader>
            <DialogTitle>Create New Experiment</DialogTitle>
            <DialogDescription>
              Give your experiment a name and select a dataset to get started.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="name">Experiment Name</Label>
              <Input
                id="name"
                placeholder="My ML Experiment"
                value={name}
                onChange={(e) => setName(e.target.value)}
                disabled={submitting}
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="dataset">Dataset</Label>
              <Select 
                value={datasetId} 
                onValueChange={setDatasetId} 
                disabled={submitting || loading || !!preselectedDatasetId}
              >
                <SelectTrigger id="dataset">
                  <SelectValue placeholder={loading ? "Loading datasets..." : "Select a dataset"} />
                </SelectTrigger>
                <SelectContent>
                  {datasets.length === 0 && !loading ? (
                    <div className="p-2 text-sm text-muted-foreground text-center">
                      No datasets available. Upload one first.
                    </div>
                  ) : (
                    datasets.map((dataset) => (
                      <SelectItem key={dataset.id} value={dataset.id}>
                        {dataset.name}
                      </SelectItem>
                    ))
                  )}
                </SelectContent>
              </Select>
              {preselectedDatasetId && (
                <p className="text-xs text-muted-foreground">
                  Dataset has been pre-selected for this experiment
                </p>
              )}
            </div>
          </div>
          
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={submitting}>
              Cancel
            </Button>
            <Button type="submit" disabled={submitting || !name.trim() || !datasetId}>
              {submitting ? "Creating..." : "Create Experiment"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default CreateExperimentModal;
