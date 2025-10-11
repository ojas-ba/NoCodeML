import { useState } from "react";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { datasetAPI } from "@/services/apiService";
import { toast } from "sonner";

interface DatasetRenameModalProps {
  dataset: { id: string; name: string } | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onRenameSuccess: () => void;
}

const DatasetRenameModal = ({ dataset, open, onOpenChange, onRenameSuccess }: DatasetRenameModalProps) => {
  const [newName, setNewName] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!newName.trim() || !dataset) {
      toast.error("Please enter a valid name");
      return;
    }
    
    setSubmitting(true);
    try {
      await datasetAPI.update(dataset.id, { 
        name: newName.trim(),
        description: null // Keep existing description or set to null
      });
      toast.success("Dataset renamed successfully");
      setNewName("");
      onRenameSuccess();
      onOpenChange(false);
    } catch (error: any) {
      toast.error(error.message || "Failed to rename dataset");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <form onSubmit={handleSubmit}>
          <DialogHeader>
            <DialogTitle>Rename Dataset</DialogTitle>
            <DialogDescription>
              Enter a new name for "{dataset?.name}"
            </DialogDescription>
          </DialogHeader>
          
          <div className="py-4">
            <Label htmlFor="newName">New Name</Label>
            <Input
              id="newName"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder={dataset?.name}
              disabled={submitting}
              className="mt-2"
            />
          </div>
          
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={submitting}>
              Cancel
            </Button>
            <Button type="submit" disabled={submitting || !newName.trim()}>
              {submitting ? "Renaming..." : "Rename"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default DatasetRenameModal;
