import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Database, FileText, AlertCircle, Info } from "lucide-react";

interface DatasetInfo {
  id: string;
  name: string;
  row_count: number;
  column_count: number;
  file_size_bytes: number;
  file_name: string;
  memory_usage_bytes: number;
}

interface Statistics {
  [column: string]: {
    count: number;
    mean: number;
    std: number;
    min: number;
    '25%': number;
    '50%': number;
    '75%': number;
    max: number;
  };
}

interface EDAStatisticsProps {
  datasetInfo: DatasetInfo;
  statistics: Statistics;
  numericColumns: string[];
  categoricalColumns: string[];
  missingDataSummary: {
    total_missing: number;
    total_cells: number;
    missing_percent: number;
    columns_with_missing: Array<{
      column: string;
      missing_count: number;
      missing_percent: number;
    }>;
  };
}

const formatBytes = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
};

const formatNumber = (num: number): string => {
  return new Intl.NumberFormat().format(num);
};

export const EDAStatistics = ({ 
  datasetInfo, 
  statistics, 
  numericColumns, 
  categoricalColumns,
  missingDataSummary 
}: EDAStatisticsProps) => {
  return (
    <div className="space-y-4">
      {/* Dataset Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center">
              <Database className="w-4 h-4 mr-2 text-primary" />
              Rows
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatNumber(datasetInfo.row_count)}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center">
              <FileText className="w-4 h-4 mr-2 text-primary" />
              Columns
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{datasetInfo.column_count}</div>
            <p className="text-xs text-muted-foreground mt-1">
              {numericColumns.length} numeric, {categoricalColumns.length} categorical
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center">
              <Info className="w-4 h-4 mr-2 text-primary" />
              File Size
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatBytes(datasetInfo.file_size_bytes)}</div>
            <p className="text-xs text-muted-foreground mt-1">
              {formatBytes(datasetInfo.memory_usage_bytes)} in memory
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center">
              <AlertCircle className="w-4 h-4 mr-2 text-warning" />
              Missing Data
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{missingDataSummary.missing_percent.toFixed(2)}%</div>
            <p className="text-xs text-muted-foreground mt-1">
              {formatNumber(missingDataSummary.total_missing)} of {formatNumber(missingDataSummary.total_cells)} cells
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Missing Data Details */}
      {missingDataSummary.columns_with_missing.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Columns with Missing Values</CardTitle>
            <CardDescription>
              {missingDataSummary.columns_with_missing.length} columns have missing data
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {missingDataSummary.columns_with_missing.slice(0, 10).map((col) => (
                <div key={col.column} className="flex items-center justify-between p-2 bg-secondary/30 rounded">
                  <span className="text-sm font-medium">{col.column}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">
                      {formatNumber(col.missing_count)} missing
                    </span>
                    <span className="text-xs font-semibold text-warning">
                      {col.missing_percent.toFixed(2)}%
                    </span>
                  </div>
                </div>
              ))}
              {missingDataSummary.columns_with_missing.length > 10 && (
                <p className="text-xs text-muted-foreground text-center pt-2">
                  + {missingDataSummary.columns_with_missing.length - 10} more columns
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Numeric Statistics Summary */}
      {numericColumns.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Numeric Columns Summary</CardTitle>
            <CardDescription>
              Statistical overview of {numericColumns.length} numeric columns
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left p-2 font-semibold">Column</th>
                    <th className="text-right p-2 font-semibold">Mean</th>
                    <th className="text-right p-2 font-semibold">Std</th>
                    <th className="text-right p-2 font-semibold">Min</th>
                    <th className="text-right p-2 font-semibold">Max</th>
                  </tr>
                </thead>
                <tbody>
                  {numericColumns.slice(0, 5).map((col) => {
                    const stats = statistics[col];
                    if (!stats) return null;
                    return (
                      <tr key={col} className="border-b">
                        <td className="p-2 font-medium">{col}</td>
                        <td className="text-right p-2">{stats.mean?.toFixed(2) ?? 'N/A'}</td>
                        <td className="text-right p-2">{stats.std?.toFixed(2) ?? 'N/A'}</td>
                        <td className="text-right p-2">{stats.min?.toFixed(2) ?? 'N/A'}</td>
                        <td className="text-right p-2">{stats.max?.toFixed(2) ?? 'N/A'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {numericColumns.length > 5 && (
                <p className="text-xs text-muted-foreground text-center pt-2">
                  + {numericColumns.length - 5} more numeric columns
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
