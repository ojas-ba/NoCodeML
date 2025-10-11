import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { PlotControls } from "./PlotControls";
import { EDAPlot } from "./EDAPlot";

interface VisualizationPanelProps {
  datasetId: string;
  numericColumns: string[];
  categoricalColumns: string[];
  idColumns: string[];
}

export const VisualizationPanel = ({ 
  datasetId, 
  numericColumns, 
  categoricalColumns,
  idColumns 
}: VisualizationPanelProps) => {
  const [plotType, setPlotType] = useState("histogram");
  const [xColumn, setXColumn] = useState("none");
  const [yColumn, setYColumn] = useState("none");
  const [groupBy, setGroupBy] = useState("none");
  const [shouldGeneratePlot, setShouldGeneratePlot] = useState(false);

  // Available columns (excluding ID columns for cleaner analysis)
  const availableColumns = {
    numeric: numericColumns.filter(col => !idColumns.includes(col)),
    categorical: categoricalColumns.filter(col => !idColumns.includes(col)),
    excluded: idColumns
  };

  // Set default columns when component mounts
  useEffect(() => {
    if (xColumn === 'none' && availableColumns.numeric.length > 0) {
      setXColumn(availableColumns.numeric[0]);
      if (availableColumns.numeric.length > 1) {
        setYColumn(availableColumns.numeric[1]);
      }
    }
  }, [availableColumns.numeric, xColumn]);

  // Reset columns when plot type changes
  const handlePlotTypeChange = (newType: string) => {
    setPlotType(newType);
    setShouldGeneratePlot(false);

    // Set appropriate defaults based on plot type
    if (newType === 'histogram' || newType === 'box') {
      if (availableColumns.numeric.length > 0) {
        setXColumn(availableColumns.numeric[0]);
      }
      setYColumn('none');
    } else if (newType === 'scatter') {
      if (availableColumns.numeric.length >= 2) {
        setXColumn(availableColumns.numeric[0]);
        setYColumn(availableColumns.numeric[1]);
      }
    } else if (newType === 'bar') {
      if (availableColumns.categorical.length > 0) {
        setXColumn(availableColumns.categorical[0]);
      } else if (availableColumns.numeric.length > 0) {
        setXColumn(availableColumns.numeric[0]);
      }
      setYColumn('none');
    } else if (newType === 'correlation') {
      // Correlation doesn't need specific columns
      setXColumn('none');
      setYColumn('none');
    }

    setGroupBy('none');
  };

  const handleGeneratePlot = () => {
    setShouldGeneratePlot(true);
  };

  return (
    <div className="space-y-6">
      {/* Plot Controls */}
      <PlotControls
        plotType={plotType}
        xColumn={xColumn}
        yColumn={yColumn}
        groupBy={groupBy}
        availableColumns={availableColumns}
        onPlotTypeChange={handlePlotTypeChange}
        onXColumnChange={setXColumn}
        onYColumnChange={setYColumn}
        onGroupByChange={setGroupBy}
        onGeneratePlot={handleGeneratePlot}
        isGenerating={false}
      />

      {/* Plot Display - EDAPlot handles its own loading and error states */}
      {shouldGeneratePlot ? (
        <EDAPlot 
          datasetId={datasetId}
          plotType={plotType}
          xColumn={xColumn}
          yColumn={yColumn}
          groupBy={groupBy}
          shouldGenerate={shouldGeneratePlot}
        />
      ) : (
        <Card>
          <CardContent className="pt-6">
            <div className="flex flex-col items-center justify-center py-12 space-y-4">
              <div className="text-center">
                <p className="text-lg font-medium">No plot generated yet</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Select plot type and columns, then click "Generate Plot"
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
