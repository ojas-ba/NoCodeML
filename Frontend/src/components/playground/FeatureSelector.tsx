import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AlertCircle } from "lucide-react";

interface ColumnInfo {
  name: string;
  dtype: string;
  missing_count: number;
  missing_percent: number;
  unique_count: number;
  is_id_column: boolean;
  sample_values?: any[];
}

interface FeatureSelectorProps {
  numericColumns: string[];
  categoricalColumns: string[];
  idColumns: string[];
  allColumns: ColumnInfo[];
  selectedFeatures: string[];
  onFeatureToggle: (feature: string) => void;
  onSelectAll: () => void;
  onClearAll: () => void;
  onSelectRecommended?: () => void;
  targetColumn?: string;
}

export const FeatureSelector = ({
  numericColumns,
  categoricalColumns,
  idColumns,
  allColumns,
  selectedFeatures,
  onFeatureToggle,
  onSelectAll,
  onClearAll,
  onSelectRecommended,
  targetColumn
}: FeatureSelectorProps) => {
  
  const getColumnInfo = (colName: string) => {
    return allColumns.find(col => col.name === colName);
  };

  const formatNumber = (num: number): string => {
    return new Intl.NumberFormat().format(num);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Feature Selection</CardTitle>
        <CardDescription>
          Select features to use for model training. ID columns are automatically excluded.
          {targetColumn && (
            <span className="block mt-1 text-primary font-medium">
              Target column "{targetColumn}" is excluded from features.
            </span>
          )}
        </CardDescription>
        <div className="flex gap-2 pt-2">
          <Button variant="outline" size="sm" onClick={onSelectRecommended || onSelectAll}>
            Select Recommended
          </Button>
          <Button variant="outline" size="sm" onClick={onSelectAll}>
            Select All
          </Button>
          <Button variant="outline" size="sm" onClick={onClearAll}>
            Clear All
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        
        {/* Numerical Features */}
        {numericColumns.length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-3">
              <span className="text-sm font-semibold">
                📊 Numerical Features ({selectedFeatures.filter(f => numericColumns.includes(f)).length}/{numericColumns.length})
              </span>
            </div>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {numericColumns.map((col) => {
                const info = getColumnInfo(col);
                const isSelected = selectedFeatures.includes(col);
                
                return (
                  <div
                    key={col}
                    className="flex items-center justify-between p-3 border rounded-lg hover:bg-accent/50 transition-colors"
                  >
                    <div className="flex items-center gap-3 flex-1">
                      <Checkbox
                        checked={isSelected}
                        onCheckedChange={() => onFeatureToggle(col)}
                        id={`feature-${col}`}
                      />
                      <label
                        htmlFor={`feature-${col}`}
                        className="flex-1 cursor-pointer"
                      >
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{col}</span>
                          <Badge variant="outline" className="text-xs">
                            🔢 {info?.dtype || 'numeric'}
                          </Badge>
                        </div>
                        {info && (
                          <div className="text-xs text-muted-foreground mt-1">
                            {info.unique_count} unique • {info.missing_count > 0 ? `${info.missing_count} missing` : 'No missing'}
                          </div>
                        )}
                      </label>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Categorical Features */}
        {categoricalColumns.length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-3">
              <span className="text-sm font-semibold">
                📝 Categorical Features ({selectedFeatures.filter(f => categoricalColumns.includes(f)).length}/{categoricalColumns.length})
              </span>
            </div>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {categoricalColumns.map((col) => {
                const info = getColumnInfo(col);
                const isSelected = selectedFeatures.includes(col);
                const highCardinality = info && info.unique_count > 50;
                
                return (
                  <div
                    key={col}
                    className="flex items-center justify-between p-3 border rounded-lg hover:bg-accent/50 transition-colors"
                  >
                    <div className="flex items-center gap-3 flex-1">
                      <Checkbox
                        checked={isSelected}
                        onCheckedChange={() => onFeatureToggle(col)}
                        id={`feature-${col}`}
                      />
                      <label
                        htmlFor={`feature-${col}`}
                        className="flex-1 cursor-pointer"
                      >
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{col}</span>
                          <Badge variant="outline" className="text-xs">
                            📝 {info?.dtype || 'categorical'}
                          </Badge>
                          {highCardinality && (
                            <Badge variant="secondary" className="text-xs">
                              ⚠️ High cardinality
                            </Badge>
                          )}
                        </div>
                        {info && (
                          <div className="text-xs text-muted-foreground mt-1">
                            {formatNumber(info.unique_count)} unique values • {info.missing_count > 0 ? `${info.missing_count} missing` : 'No missing'}
                          </div>
                        )}
                      </label>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Excluded Columns */}
        {idColumns.length > 0 && (
          <div className="border-t pt-4">
            <details className="group">
              <summary className="cursor-pointer flex items-center gap-2 text-sm font-semibold text-muted-foreground hover:text-foreground">
                <span className="group-open:rotate-90 transition-transform">▶</span>
                🔒 Excluded Columns ({idColumns.length})
              </summary>
              <div className="mt-3 space-y-2">
                {idColumns.map((col) => {
                  const info = getColumnInfo(col);
                  return (
                    <div
                      key={col}
                      className="flex items-center justify-between p-3 border rounded-lg bg-muted/30 opacity-60"
                    >
                      <div className="flex items-center gap-3 flex-1">
                        <Checkbox
                          checked={false}
                          disabled
                          className="cursor-not-allowed"
                        />
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className="font-medium">{col}</span>
                            <Badge variant="secondary" className="text-xs">
                              🆔 ID Column
                            </Badge>
                          </div>
                          <div className="text-xs text-muted-foreground mt-1 flex items-center gap-1">
                            <AlertCircle className="w-3 h-3" />
                            Auto-excluded: ID columns have no predictive value
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </details>
          </div>
        )}

        {/* Selection Summary */}
        <div className="border-t pt-4">
          <div className="text-sm text-muted-foreground">
            <strong>{selectedFeatures.length}</strong> features selected
            {selectedFeatures.length === 0 && (
              <span className="text-warning ml-2">⚠️ Select at least one feature</span>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
