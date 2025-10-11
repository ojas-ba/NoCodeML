import { useEffect, useState, Component, ErrorInfo, ReactNode } from "react";
import Plot from "react-plotly.js";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircle, AlertTriangle, TrendingUp } from "lucide-react";
import { useEDA } from "@/hooks/useEDA";

// Error Boundary Component
class PlotErrorBoundary extends Component<
  { children: ReactNode; resetKey?: string | number },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: ReactNode; resetKey?: string | number }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    console.error('Plot Error Caught:', error);
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Plot Error Details:', error, errorInfo);
  }

  componentDidUpdate(prevProps: { resetKey?: string | number }) {
    // Reset error state when resetKey changes
    if (this.state.hasError && prevProps.resetKey !== this.props.resetKey) {
      this.setState({ hasError: false, error: null });
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <Card>
          <CardContent className="pt-6">
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Failed to render plot: {this.state.error?.message || 'Unknown error'}
                <br />
                <span className="text-xs mt-2 block">Try selecting different columns or plot type.</span>
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>
      );
    }

    return this.props.children;
  }
}

interface EDAPlotProps {
  datasetId: string;
  plotType: string;
  xColumn: string;
  yColumn?: string;
  groupBy?: string;
  shouldGenerate?: boolean;
}

const EDAPlotContent = ({ datasetId, plotType, xColumn, yColumn, groupBy, shouldGenerate }: EDAPlotProps) => {
  const { plotData, plotLoading, plotError, generatePlot } = useEDA(datasetId);
  const [plotKey, setPlotKey] = useState(0);
  const [localError, setLocalError] = useState<string | null>(null);

  // Log component mount/unmount for debugging
  useEffect(() => {
    console.log('EDAPlotContent mounted with props:', { datasetId, plotType, xColumn, yColumn, groupBy, shouldGenerate });
    return () => {
      console.log('EDAPlotContent unmounting');
    };
  }, []);

  // Only generate plot when shouldGenerate trigger changes
  useEffect(() => {
    if (!shouldGenerate || !datasetId) {
      console.log('Skipping plot generation:', { shouldGenerate, datasetId });
      return;
    }

    // Skip if required columns are not selected (still set to 'none')
    if (plotType !== 'correlation' && (!xColumn || xColumn === 'none')) {
      console.log('Skipping: X column not selected');
      return;
    }
    if (plotType === 'scatter' && (!yColumn || yColumn === 'none')) {
      console.log('Skipping: Y column not selected for scatter plot');
      return;
    }

    // Clear local error
    setLocalError(null);

    // Filter out 'none' placeholder values
    const plotConfig = {
      plot_type: plotType,
      x_column: xColumn !== 'none' ? xColumn : undefined,
      y_column: yColumn !== 'none' ? yColumn : undefined,
      group_by: groupBy !== 'none' ? groupBy : undefined,
    };

    console.log('Generating plot with config:', plotConfig);

    try {
      generatePlot(plotConfig);
    } catch (err: any) {
      console.error('Error generating plot:', err);
      setLocalError(err.message || 'Failed to generate plot');
    }
  }, [shouldGenerate, datasetId, generatePlot, plotType, xColumn, yColumn, groupBy]);

  // Force re-render of plot when data changes
  useEffect(() => {
    if (plotData) {
      setPlotKey(prev => prev + 1);
    }
  }, [plotData]);

  if (plotLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-4 w-64 mt-2" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-96 w-full" />
        </CardContent>
      </Card>
    );
  }

  if (plotError || localError) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>{plotError || localError}</AlertDescription>
      </Alert>
    );
  }

  if (!plotData && !plotLoading) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="text-center text-muted-foreground py-12">
            <TrendingUp className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>Configure your plot and click "Generate Plot" to visualize</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Validate plot data structure
  const isValidPlotData = plotData && 
    plotData.data && 
    Array.isArray(plotData.data) && 
    plotData.data.length > 0 &&
    plotData.layout;

  if (!isValidPlotData) {
    return (
      <Card>
        <CardContent className="pt-6">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>Invalid plot data received from server</AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="text-lg">Visualization</CardTitle>
            <CardDescription>
              {plotData.plot_type?.charAt(0).toUpperCase() + plotData.plot_type?.slice(1) || 'Plot'} - 
              {plotData.is_sampled 
                ? ` Displaying ${(plotData.displayed_rows || 0).toLocaleString()} of ${(plotData.total_rows || 0).toLocaleString()} rows`
                : ` ${(plotData.total_rows || 0).toLocaleString()} rows`
              }
            </CardDescription>
          </div>
          {plotData.is_sampled && (
            <Badge variant="outline" className="border-warning text-warning">
              <AlertTriangle className="w-3 h-3 mr-1" />
              Sampled for Performance
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {plotData.is_sampled && (
          <Alert className="mb-4 border-warning/50 bg-warning/10">
            <AlertTriangle className="h-4 w-4 text-warning" />
            <AlertDescription className="text-sm">
              <strong>Performance Optimization:</strong> Displaying {(plotData.displayed_rows || 0).toLocaleString()} randomly sampled rows 
              out of {(plotData.total_rows || 0).toLocaleString()} total rows for better performance. 
              Statistics are computed on the full dataset.
            </AlertDescription>
          </Alert>
        )}

        <div className="bg-secondary/30 rounded-lg p-4 border border-border">
          {isValidPlotData ? (
            <Plot
              key={plotKey}
              data={plotData.data || []}
              layout={{
                ...(plotData.layout || {}),
                autosize: true,
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { 
                  color: 'hsl(var(--foreground))',
                  family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
                },
                xaxis: { 
                  ...(plotData.layout?.xaxis || {}),
                  gridcolor: 'hsl(var(--border))',
                  zerolinecolor: 'hsl(var(--border))'
                },
                yaxis: { 
                  ...(plotData.layout?.yaxis || {}),
                  gridcolor: 'hsl(var(--border))',
                  zerolinecolor: 'hsl(var(--border))'
                },
                margin: { t: 40, r: 40, b: 60, l: 60 },
                hovermode: 'closest',
              }}
              config={{ 
                responsive: true, 
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                toImageButtonOptions: {
                  format: 'png',
                  filename: `eda_${plotData.plot_type || 'plot'}_${Date.now()}`,
                  height: 800,
                  width: 1200,
                  scale: 2
                }
              }}
              style={{ width: '100%', height: '500px' }}
              useResizeHandler={true}
              onError={(err) => {
                console.error('Plotly rendering error:', err);
                setLocalError('Failed to render plot');
              }}
            />
          ) : (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>No plot data available</AlertDescription>
            </Alert>
          )}
        </div>

        {/* Strong Correlations Info (if available) */}
        {plotData.plot_type === 'correlation' && plotData.data[0]?.z && (
          <div className="mt-4 space-y-2">
            <p className="text-sm font-semibold">Strong Correlations (|r| &gt; 0.7):</p>
            {(() => {
              const correlations: Array<{ col1: string; col2: string; value: number }> = [];
              const matrix = plotData.data[0].z as number[][];
              const columns = plotData.data[0].x as string[];
              
              for (let i = 0; i < matrix.length; i++) {
                for (let j = i + 1; j < matrix[i].length; j++) {
                  const value = matrix[i][j];
                  if (Math.abs(value) > 0.7) {
                    correlations.push({
                      col1: columns[i],
                      col2: columns[j],
                      value: value
                    });
                  }
                }
              }
              
              if (correlations.length === 0) {
                return <p className="text-xs text-muted-foreground">No strong correlations detected</p>;
              }
              
              return (
                <div className="space-y-1">
                  {correlations.map((corr, idx) => (
                    <div key={idx} className="text-xs p-2 bg-primary/10 border border-primary/30 rounded">
                      <strong>{corr.col1}</strong> ↔ <strong>{corr.col2}</strong>: {corr.value.toFixed(3)}
                    </div>
                  ))}
                </div>
              );
            })()}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

// Export wrapped component with error boundary
export const EDAPlot = (props: EDAPlotProps) => (
  <PlotErrorBoundary resetKey={`${props.plotType}-${props.shouldGenerate}`}>
    <EDAPlotContent {...props} />
  </PlotErrorBoundary>
);
