import { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { datasetAPI } from "@/services/apiService";
import { toast } from "sonner";

interface DatasetPreviewModalProps {
  datasetId: string | null;
  datasetName: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const DatasetPreviewModal = ({ datasetId, datasetName, open, onOpenChange }: DatasetPreviewModalProps) => {
  const [loading, setLoading] = useState(false);
  const [previewData, setPreviewData] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    if (open && datasetId) {
      loadData();
    }
  }, [open, datasetId]);

  const loadData = async () => {
    if (!datasetId) return;
    
    setLoading(true);
    try {
      // Backend preview endpoint returns DatasetPreviewResponse:
      // { columns: [...], data: [[...], [...]], row_count: number, preview_rows: number }
      const preview = await datasetAPI.preview(datasetId, 50);
      setPreviewData(preview);
      
      // Set stats from preview response
      setStats({
        totalRows: preview.row_count,
        totalColumns: preview.columns?.length || 0,
        missingValues: 0, // Not provided in preview response
        numericColumns: 0, // Not provided in preview response
        columnTypes: {} // Not provided in preview response
      });
    } catch (error: any) {
      toast.error(error.message || "Failed to load dataset preview");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-6xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>{datasetName}</DialogTitle>
          <DialogDescription>
            Dataset preview and statistics
          </DialogDescription>
        </DialogHeader>

        {loading && !previewData ? (
          <div className="space-y-4 py-4">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-64 w-full" />
          </div>
        ) : (
          <div className="flex-1 overflow-auto space-y-6">
            {/* Statistics */}
            {stats && (
              <div className="grid grid-cols-4 gap-4">
                <div className="bg-card border border-border rounded-lg p-4">
                  <p className="text-sm text-muted-foreground mb-1">Total Rows</p>
                  <p className="text-2xl font-bold">{stats.totalRows?.toLocaleString()}</p>
                </div>
                <div className="bg-card border border-border rounded-lg p-4">
                  <p className="text-sm text-muted-foreground mb-1">Total Columns</p>
                  <p className="text-2xl font-bold">{stats.totalColumns}</p>
                </div>
                <div className="bg-card border border-border rounded-lg p-4">
                  <p className="text-sm text-muted-foreground mb-1">Missing Values</p>
                  <p className="text-2xl font-bold">{stats.missingValues || 0}</p>
                </div>
                <div className="bg-card border border-border rounded-lg p-4">
                  <p className="text-sm text-muted-foreground mb-1">Numeric Columns</p>
                  <p className="text-2xl font-bold">{stats.numericColumns || 0}</p>
                </div>
              </div>
            )}

            {/* Data Table */}
            {previewData?.data && (
              <div>
                <h3 className="font-semibold mb-3">Data Preview</h3>
                <div className="border border-border rounded-lg overflow-auto max-h-96">
                  <table className="w-full text-sm">
                    <thead className="bg-muted/50 sticky top-0">
                      <tr>
                        {previewData.columns?.map((col: string, idx: number) => (
                          <th key={idx} className="px-4 py-3 text-left font-medium border-b border-border whitespace-nowrap">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {previewData.data.map((row: any[], rowIdx: number) => (
                        <tr key={rowIdx} className="hover:bg-muted/20 transition-colors">
                          {row.map((cell: any, cellIdx: number) => (
                            <td key={cellIdx} className="px-4 py-3 border-b border-border whitespace-nowrap">
                              {cell !== null && cell !== undefined ? String(cell) : (
                                <span className="text-muted-foreground italic">null</span>
                              )}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Preview Info */}
                <div className="flex items-center justify-center mt-4">
                  <p className="text-sm text-muted-foreground">
                    Showing {previewData.preview_rows} of {stats?.totalRows?.toLocaleString() || 0} total rows
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default DatasetPreviewModal;
