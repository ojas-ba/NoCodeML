import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Hash, Type, Key, Info } from "lucide-react";

interface ColumnInfo {
  name: string;
  dtype: string;
  missing_count: number;
  missing_percent: number;
  unique_count: number;
  is_id_column: boolean;
  sample_values?: any[];
}

interface EDAColumnInfoProps {
  columns: ColumnInfo[];
  idColumns: string[];
}

const getDtypeIcon = (dtype: string) => {
  if (dtype.includes('int') || dtype.includes('float')) {
    return <Hash className="w-4 h-4 text-blue-500" />;
  }
  if (dtype.includes('object') || dtype.includes('string')) {
    return <Type className="w-4 h-4 text-green-500" />;
  }
  return <Info className="w-4 h-4 text-gray-500" />;
};

const getDtypeColor = (dtype: string) => {
  if (dtype.includes('int') || dtype.includes('float')) {
    return 'bg-blue-500/10 text-blue-500 border-blue-500/20';
  }
  if (dtype.includes('object') || dtype.includes('string')) {
    return 'bg-green-500/10 text-green-500 border-green-500/20';
  }
  return 'bg-gray-500/10 text-gray-500 border-gray-500/20';
};

export const EDAColumnInfo = ({ columns, idColumns }: EDAColumnInfoProps) => {
  const regularColumns = columns.filter(col => !col.is_id_column);
  const idColumnDetails = columns.filter(col => col.is_id_column);

  return (
    <div className="space-y-4">
      {/* ID Columns (if any) */}
      {idColumnDetails.length > 0 && (
        <Card className="border-warning/50">
          <CardHeader>
            <CardTitle className="text-lg flex items-center">
              <Key className="w-5 h-5 mr-2 text-warning" />
              Excluded Columns (ID Columns)
            </CardTitle>
            <CardDescription>
              These columns are automatically excluded from analysis as they appear to be identifiers
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {idColumnDetails.map((col) => (
                <TooltipProvider key={col.name}>
                  <Tooltip>
                    <TooltipTrigger>
                      <Badge variant="outline" className="border-warning/50 text-warning">
                        <Key className="w-3 h-3 mr-1" />
                        {col.name}
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent>
                      <div className="text-xs space-y-1">
                        <p><strong>Type:</strong> {col.dtype}</p>
                        <p><strong>Unique:</strong> {col.unique_count.toLocaleString()}</p>
                        <p><strong>Missing:</strong> {col.missing_percent.toFixed(2)}%</p>
                      </div>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Regular Columns */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Column Details</CardTitle>
          <CardDescription>
            Overview of {regularColumns.length} columns used in analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {regularColumns.map((col) => (
              <div key={col.name} className="p-3 bg-secondary/30 rounded-lg border border-border">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {getDtypeIcon(col.dtype)}
                    <span className="font-semibold">{col.name}</span>
                  </div>
                  <Badge variant="outline" className={getDtypeColor(col.dtype)}>
                    {col.dtype}
                  </Badge>
                </div>
                
                <div className="grid grid-cols-3 gap-2 text-xs text-muted-foreground">
                  <div>
                    <span className="font-medium">Unique:</span> {col.unique_count.toLocaleString()}
                  </div>
                  <div>
                    <span className="font-medium">Missing:</span> {col.missing_count.toLocaleString()} 
                    {col.missing_percent > 0 && (
                      <span className="text-warning ml-1">({col.missing_percent.toFixed(1)}%)</span>
                    )}
                  </div>
                  <div>
                    <span className="font-medium">Type:</span> {
                      col.dtype.includes('int') || col.dtype.includes('float') ? 'Numeric' : 'Categorical'
                    }
                  </div>
                </div>

                {col.sample_values && col.sample_values.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-border/50">
                    <span className="text-xs font-medium text-muted-foreground">Sample values:</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {col.sample_values.map((val, idx) => (
                        <Badge key={idx} variant="secondary" className="text-xs">
                          {String(val).substring(0, 20)}{String(val).length > 20 ? '...' : ''}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
