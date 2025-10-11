import { useState, useEffect } from "react";
import { Database, Trash2, Eye, Edit2, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import { datasetAPI } from "@/services/apiService";
import { toast } from "sonner";
import DatasetUploadModal from "@/components/datasets/DatasetUploadModal";
import DatasetPreviewModal from "@/components/datasets/DatasetPreviewModal";
import DatasetRenameModal from "@/components/datasets/DatasetRenameModal";

// Helper function to format backend dataset response for frontend display
const formatDataset = (dataset: any) => ({
  ...dataset,
  rows: dataset.row_count,
  columns: dataset.column_count,
  size: dataset.file_size_bytes 
    ? `${(dataset.file_size_bytes / (1024 * 1024)).toFixed(2)} MB`
    : 'N/A',
  uploaded: dataset.created_at 
    ? new Date(dataset.created_at).toLocaleDateString()
    : 'N/A'
});

const Datasets = () => {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  
  // Modals state
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [previewModalOpen, setPreviewModalOpen] = useState(false);
  const [renameModalOpen, setRenameModalOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  
  // Selected dataset for actions
  const [selectedDataset, setSelectedDataset] = useState<any>(null);

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    setLoading(true);
    try {
      const datasets = await datasetAPI.list();
      // datasetAPI.list() already extracts the array from response
      const formattedDatasets = (datasets || []).map(formatDataset);
      setDatasets(formattedDatasets);
    } catch (error: any) {
      toast.error(error.message || "Failed to load datasets");
    } finally {
      setLoading(false);
    }
  };

  const [deleteError, setDeleteError] = useState<{ message: string; dependencies: any[] } | null>(null);

  const handleDelete = async () => {
    if (!selectedDataset) return;
    
    try {
      await datasetAPI.delete(selectedDataset.id);
      toast.success("Dataset deleted successfully");
      loadDatasets();
      setDeleteDialogOpen(false);
      setSelectedDataset(null);
      setDeleteError(null);
    } catch (error: any) {
      // Handle 409 Conflict - dataset has dependencies
      if (error.statusCode === 409 && error.dependencies) {
        setDeleteError({
          message: error.message,
          dependencies: error.dependencies
        });
      } else {
        toast.error(error.message || "Failed to delete dataset");
        setDeleteDialogOpen(false);
      }
    }
  };

  const filteredDatasets = datasets.filter(ds =>
    ds.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen py-8">
      <div className="container mx-auto px-4">
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-4xl font-bold gradient-text mb-2">Datasets</h1>
              <p className="text-muted-foreground">Manage your uploaded datasets and create experiments</p>
            </div>
            <Button 
              onClick={() => setUploadModalOpen(true)}
              className="gradient-primary text-background gap-2"
            >
              <Plus className="w-4 h-4" />
              Upload Dataset
            </Button>
          </div>

          <div className="max-w-md">
            <Input
              placeholder="Search datasets..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </div>
        
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[1, 2, 3].map(i => (
              <div key={i} className="bg-card border border-border rounded-xl p-6 animate-pulse">
                <div className="h-6 bg-muted rounded w-3/4 mb-4"></div>
                <div className="h-4 bg-muted rounded w-1/2 mb-6"></div>
                <div className="grid grid-cols-3 gap-2 mb-4">
                  <div className="h-12 bg-muted rounded"></div>
                  <div className="h-12 bg-muted rounded"></div>
                  <div className="h-12 bg-muted rounded"></div>
                </div>
                <div className="h-10 bg-muted rounded"></div>
              </div>
            ))}
          </div>
        ) : filteredDatasets.length === 0 ? (
          <div className="bg-card border border-border rounded-xl p-12 text-center">
            <div className="max-w-md mx-auto">
              <Database className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-xl font-semibold mb-2">
                {searchQuery ? "No datasets found" : "No datasets yet"}
              </h3>
              <p className="text-muted-foreground mb-6">
                {searchQuery 
                  ? "Try adjusting your search query"
                  : "Upload your first dataset to start creating ML experiments"}
              </p>
              {!searchQuery && (
                <Button 
                  onClick={() => setUploadModalOpen(true)}
                  className="gradient-primary text-background gap-2"
                >
                  <Plus className="w-4 h-4" />
                  Upload Dataset
                </Button>
              )}
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredDatasets.map((dataset) => (
              <div key={dataset.id} className="card-hover bg-card border border-border rounded-xl p-6 space-y-4">
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-3 flex-1">
                    <div className="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center flex-shrink-0">
                      <Database className="w-5 h-5 text-primary" />
                    </div>
                    <div className="min-w-0">
                      <h3 className="font-semibold truncate">{dataset.name}</h3>
                      <p className="text-sm text-muted-foreground">{dataset.uploaded}</p>
                    </div>
                  </div>
                </div>
                
                <div className="grid grid-cols-3 gap-2 py-2 border-t border-b border-border">
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground">Rows</p>
                    <p className="font-semibold">{dataset.rows?.toLocaleString()}</p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground">Columns</p>
                    <p className="font-semibold">{dataset.columns}</p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground">Size</p>
                    <p className="font-semibold">{dataset.size}</p>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex gap-2">
                    <Button 
                      variant="outline" 
                      size="sm" 
                      className="flex-1"
                      onClick={() => {
                        setSelectedDataset(dataset);
                        setPreviewModalOpen(true);
                      }}
                    >
                      <Eye className="w-4 h-4 mr-2" />
                      Preview
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => {
                        setSelectedDataset(dataset);
                        setRenameModalOpen(true);
                      }}
                    >
                      <Edit2 className="w-4 h-4" />
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => {
                        setSelectedDataset(dataset);
                        setDeleteDialogOpen(true);
                      }}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Upload Modal */}
      <DatasetUploadModal
        open={uploadModalOpen}
        onOpenChange={setUploadModalOpen}
        onUploadSuccess={loadDatasets}
      />

      {/* Preview Modal */}
      <DatasetPreviewModal
        datasetId={selectedDataset?.id}
        datasetName={selectedDataset?.name || ""}
        open={previewModalOpen}
        onOpenChange={setPreviewModalOpen}
      />

      {/* Rename Modal */}
      <DatasetRenameModal
        dataset={selectedDataset}
        open={renameModalOpen}
        onOpenChange={setRenameModalOpen}
        onRenameSuccess={loadDatasets}
      />

      {/* Delete Confirmation */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={(open) => {
        setDeleteDialogOpen(open);
        if (!open) setDeleteError(null);
      }}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Dataset</AlertDialogTitle>
            <AlertDialogDescription>
              {deleteError ? (
                <div className="space-y-3">
                  <p className="text-destructive font-medium">{deleteError.message}</p>
                  <div className="bg-muted rounded-lg p-3">
                    <p className="text-sm font-medium mb-2">Dependent Experiments:</p>
                    <ul className="space-y-1">
                      {deleteError.dependencies.map((dep: any) => (
                        <li key={dep.id} className="text-sm">
                          • {dep.name}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <p className="text-sm">Please delete or reassign these experiments before deleting this dataset.</p>
                </div>
              ) : (
                `Are you sure you want to delete "${selectedDataset?.name}"? This action cannot be undone.`
              )}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setDeleteError(null)}>
              {deleteError ? 'Close' : 'Cancel'}
            </AlertDialogCancel>
            {!deleteError && (
              <AlertDialogAction onClick={handleDelete} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
                Delete
              </AlertDialogAction>
            )}
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default Datasets;
