import { useState, useCallback } from "react";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Upload, X, FileUp } from "lucide-react";
import { datasetAPI } from "@/services/apiService";
import { toast } from "sonner";

interface DatasetUploadModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onUploadSuccess: () => void;
}

const DatasetUploadModal = ({ open, onOpenChange, onUploadSuccess }: DatasetUploadModalProps) => {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);

  const validateFile = (file: File): string | null => {
    const maxSize = 100 * 1024 * 1024; // 100MB
    
    if (!file.name.endsWith('.csv')) {
      return "Only CSV files are supported";
    }
    
    if (file.size > maxSize) {
      return "File size must be less than 100MB";
    }
    
    if (file.size === 0) {
      return "File is empty";
    }
    
    return null;
  };

  const handleFileSelect = (selectedFile: File) => {
    const error = validateFile(selectedFile);
    
    if (error) {
      toast.error(error);
      return;
    }
    
    setFile(selectedFile);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      handleFileSelect(droppedFile);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleUpload = async () => {
    if (!file) return;
    
    setUploading(true);
    try {
      // Extract name from filename without extension
      const fileName = file.name.replace(/\.[^/.]+$/, "");
      await datasetAPI.upload(file, fileName);
      toast.success("Dataset uploaded successfully");
      setFile(null);
      onUploadSuccess();
      onOpenChange(false);
    } catch (error: any) {
      toast.error(error.message || "Failed to upload dataset");
    } finally {
      setUploading(false);
    }
  };

  const handleClose = () => {
    if (!uploading) {
      setFile(null);
      onOpenChange(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Upload Dataset</DialogTitle>
          <DialogDescription>
            Upload a CSV file to create a new dataset (max 100MB)
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {!file ? (
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${
                isDragging
                  ? "border-primary bg-primary/10 scale-[1.02]"
                  : "border-border hover:border-primary/50 hover:bg-card/50"
              }`}
            >
              <div className="flex flex-col items-center gap-4">
                <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center">
                  <Upload className="w-8 h-8 text-primary" />
                </div>
                
                <div>
                  <p className="text-lg font-semibold mb-1">
                    Drop your CSV file here
                  </p>
                  <p className="text-sm text-muted-foreground mb-4">
                    or click to browse
                  </p>
                  
                  <Button
                    variant="outline"
                    onClick={() => document.getElementById('file-input')?.click()}
                    disabled={uploading}
                  >
                    <FileUp className="w-4 h-4 mr-2" />
                    Choose File
                  </Button>
                  
                  <input
                    id="file-input"
                    type="file"
                    accept=".csv"
                    onChange={(e) => {
                      const selectedFile = e.target.files?.[0];
                      if (selectedFile) handleFileSelect(selectedFile);
                    }}
                    className="hidden"
                  />
                </div>
              </div>
            </div>
          ) : (
            <div className="border border-border rounded-xl p-4">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3 flex-1">
                  <div className="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center flex-shrink-0">
                    <FileUp className="w-5 h-5 text-primary" />
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <p className="font-medium truncate">{file.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                
                {!uploading && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setFile(null)}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="flex justify-end gap-2">
          <Button
            variant="outline"
            onClick={handleClose}
            disabled={uploading}
          >
            Cancel
          </Button>
          <Button
            onClick={handleUpload}
            disabled={!file || uploading}
            className="gradient-primary text-background"
          >
            {uploading ? "Uploading..." : "Upload Dataset"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default DatasetUploadModal;
