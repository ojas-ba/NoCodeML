import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { BarChart3, ScatterChart, Box, Maximize, TrendingUp, LineChart } from "lucide-react";
import { Info } from "lucide-react";

interface AvailableColumns {
  numeric: string[];
  categorical: string[];
  excluded: string[];
}

interface PlotControlsProps {
  plotType: string;
  xColumn: string;
  yColumn?: string;
  groupBy?: string;
  availableColumns: AvailableColumns;
  onPlotTypeChange: (type: string) => void;
  onXColumnChange: (col: string) => void;
  onYColumnChange?: (col: string) => void;
  onGroupByChange?: (col: string) => void;
  onGeneratePlot: () => void;
  isGenerating?: boolean;
}

const plotTypes = [
  { value: 'histogram', label: 'Histogram', icon: BarChart3, description: 'Distribution of a single numeric column' },
  { value: 'scatter', label: 'Scatter Plot', icon: ScatterChart, description: 'Relationship between two numeric columns' },
  { value: 'box', label: 'Box Plot', icon: Box, description: 'Distribution with outliers' },
  { value: 'correlation', label: 'Correlation Heatmap', icon: TrendingUp, description: 'Correlations between all numeric columns' },
  { value: 'bar', label: 'Bar Chart', icon: Maximize, description: 'Distribution of a categorical column' },
];

export const PlotControls = ({
  plotType,
  xColumn,
  yColumn,
  groupBy,
  availableColumns,
  onPlotTypeChange,
  onXColumnChange,
  onYColumnChange,
  onGroupByChange,
  onGeneratePlot,
  isGenerating = false
}: PlotControlsProps) => {
  const needsYColumn = plotType === 'scatter';
  const needsXColumn = plotType !== 'correlation';
  const supportsGroupBy = plotType === 'box' || plotType === 'scatter';
  const shouldUseNumeric = ['histogram', 'scatter', 'box'].includes(plotType);
  const shouldUseCategorical = plotType === 'bar';

  const availableXColumns = shouldUseNumeric 
    ? availableColumns.numeric 
    : shouldUseCategorical 
    ? availableColumns.categorical 
    : [...availableColumns.numeric, ...availableColumns.categorical];

  // Check if configuration is valid for plotting
  const canGenerate = () => {
    if (plotType === 'correlation') return true;
    if (!xColumn || xColumn === 'none') return false;
    if (plotType === 'scatter' && (!yColumn || yColumn === 'none')) return false;
    return true;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Plot Configuration</CardTitle>
        <CardDescription>
          Select plot type and columns to visualize
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Plot Type Selector */}
          <div className="space-y-2">
            <Label htmlFor="plot-type">Plot Type</Label>
            <Select value={plotType} onValueChange={onPlotTypeChange}>
              <SelectTrigger id="plot-type">
                <SelectValue placeholder="Select plot type" />
              </SelectTrigger>
              <SelectContent>
                {plotTypes.map((type) => (
                  <SelectItem key={type.value} value={type.value}>
                    <div className="flex items-center gap-2">
                      <type.icon className="w-4 h-4" />
                      <span>{type.label}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              {plotTypes.find(t => t.value === plotType)?.description}
            </p>
          </div>

          {/* X-Axis Column Selector */}
          {needsXColumn && (
            <div className="space-y-2">
              <Label htmlFor="x-column">
                {plotType === 'scatter' ? 'X-Axis Column' : 'Column'}
              </Label>
              <Select value={xColumn} onValueChange={onXColumnChange}>
                <SelectTrigger id="x-column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableXColumns.length === 0 ? (
                    <SelectItem value="none" disabled>
                      No columns available
                    </SelectItem>
                  ) : (
                    <>
                      <SelectItem value="none">Select column...</SelectItem>
                      {availableXColumns.map((col) => (
                        <SelectItem key={col} value={col}>
                          {col}
                        </SelectItem>
                      ))}
                    </>
                  )}
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Y-Axis Column Selector (for scatter) */}
          {needsYColumn && onYColumnChange && (
            <div className="space-y-2">
              <Label htmlFor="y-column">Y-Axis Column</Label>
              <Select value={yColumn || 'none'} onValueChange={onYColumnChange}>
                <SelectTrigger id="y-column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {availableColumns.numeric.length === 0 ? (
                    <SelectItem value="none" disabled>
                      No numeric columns available
                    </SelectItem>
                  ) : (
                    <>
                      <SelectItem value="none">Select column...</SelectItem>
                      {availableColumns.numeric.map((col) => (
                        <SelectItem key={col} value={col}>
                          {col}
                        </SelectItem>
                      ))}
                    </>
                  )}
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Group By Selector (for box and scatter) */}
          {supportsGroupBy && onGroupByChange && (
            <div className="space-y-2">
              <Label htmlFor="group-by">
                Group By (Optional)
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="w-3 h-3 ml-1 inline" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="text-xs">Group data by a categorical column</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </Label>
              <Select value={groupBy || 'none'} onValueChange={onGroupByChange}>
                <SelectTrigger id="group-by">
                  <SelectValue placeholder="No grouping" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">No grouping</SelectItem>
                  {availableColumns.categorical.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </div>

        {/* Generate Plot Button */}
        <div className="mt-6 flex items-center justify-between border-t border-border pt-4">
          <p className="text-sm text-muted-foreground">
            {canGenerate() 
              ? 'Configuration is valid. Click to generate plot.' 
              : 'Please select required columns to enable plot generation.'}
          </p>
          <Button 
            onClick={onGeneratePlot}
            disabled={!canGenerate() || isGenerating}
            size="lg"
            className="gap-2 min-w-[160px]"
          >
            <LineChart className="w-4 h-4" />
            {isGenerating ? 'Generating...' : 'Generate Plot'}
          </Button>
        </div>

        {/* Excluded Columns Info */}
        {availableColumns.excluded.length > 0 && (
          <div className="mt-4 p-3 bg-warning/10 border border-warning/30 rounded-lg">
            <div className="flex items-start gap-2">
              <Info className="w-4 h-4 text-warning mt-0.5 flex-shrink-0" />
              <div className="text-xs space-y-1">
                <p className="font-semibold">ID columns excluded from analysis:</p>
                <p className="text-muted-foreground">
                  {availableColumns.excluded.join(', ')}
                </p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
