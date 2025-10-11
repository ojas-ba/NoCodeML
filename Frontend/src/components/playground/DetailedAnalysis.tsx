import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { 
  Pagination, 
  PaginationContent, 
  PaginationItem, 
  PaginationLink, 
  PaginationNext, 
  PaginationPrevious 
} from "@/components/ui/pagination";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ChevronDown, Hash, Type, Key, AlertTriangle, TrendingUp, TrendingDown } from "lucide-react";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface ColumnInfo {
  name: string;
  dtype: string;
  missing_count: number;
  missing_percent: number;
  unique_count: number;
  is_id_column: boolean;
  sample_values?: any[];
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

interface Correlations {
  columns: string[];
  matrix: number[][];
  pairs: Array<{
    col1: string;
    col2: string;
    correlation: number;
  }>;
}

interface DetailedAnalysisProps {
  columns: ColumnInfo[];
  statistics: Statistics;
  correlations: Correlations | null;
  numericColumns: string[];
  idColumns: string[];
}

const getDtypeIcon = (dtype: string) => {
  if (dtype.includes('int') || dtype.includes('float')) {
    return <Hash className="w-4 h-4 text-blue-500" />;
  }
  if (dtype.includes('object') || dtype.includes('string')) {
    return <Type className="w-4 h-4 text-green-500" />;
  }
  return <Hash className="w-4 h-4 text-gray-500" />;
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

const getQualityBadge = (missingPercent: number) => {
  if (missingPercent === 0) {
    return <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/20">Excellent</Badge>;
  } else if (missingPercent < 5) {
    return <Badge variant="outline" className="bg-yellow-500/10 text-yellow-500 border-yellow-500/20">Good</Badge>;
  } else if (missingPercent < 20) {
    return <Badge variant="outline" className="bg-orange-500/10 text-orange-500 border-orange-500/20">Fair</Badge>;
  } else {
    return <Badge variant="outline" className="bg-red-500/10 text-red-500 border-red-500/20">Poor</Badge>;
  }
};

export const DetailedAnalysis = ({ 
  columns, 
  statistics, 
  correlations, 
  numericColumns,
  idColumns 
}: DetailedAnalysisProps) => {
  const [columnPage, setColumnPage] = useState(1);
  const [statsPage, setStatsPage] = useState(1);
  const [idColumnsOpen, setIdColumnsOpen] = useState(false);
  const itemsPerPage = 10;

  // Filter columns
  const regularColumns = useMemo(() => columns.filter(col => !col.is_id_column), [columns]);
  const idColumnDetails = useMemo(() => columns.filter(col => col.is_id_column), [columns]);

  // Paginate regular columns
  const totalColumnPages = Math.ceil(regularColumns.length / itemsPerPage);
  const paginatedColumns = useMemo(() => {
    const startIndex = (columnPage - 1) * itemsPerPage;
    return regularColumns.slice(startIndex, startIndex + itemsPerPage);
  }, [regularColumns, columnPage, itemsPerPage]);

  // Paginate statistics
  const numericStats = useMemo(() => 
    Object.entries(statistics).filter(([col]) => numericColumns.includes(col) && !idColumns.includes(col)),
    [statistics, numericColumns, idColumns]
  );
  const totalStatsPages = Math.ceil(numericStats.length / itemsPerPage);
  const paginatedStats = useMemo(() => {
    const startIndex = (statsPage - 1) * itemsPerPage;
    return numericStats.slice(startIndex, startIndex + itemsPerPage);
  }, [numericStats, statsPage, itemsPerPage]);

  // Filter top correlations
  const topCorrelations = useMemo(() => {
    if (!correlations || !correlations.pairs) return [];
    return correlations.pairs
      .filter(pair => Math.abs(pair.correlation) > 0.5)
      .sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation))
      .slice(0, 10);
  }, [correlations]);

  const getPageNumbers = (currentPage: number, totalPages: number) => {
    const pages = [];
    const maxPages = 5;
    let startPage = Math.max(1, currentPage - Math.floor(maxPages / 2));
    let endPage = Math.min(totalPages, startPage + maxPages - 1);
    
    if (endPage - startPage < maxPages - 1) {
      startPage = Math.max(1, endPage - maxPages + 1);
    }
    
    for (let i = startPage; i <= endPage; i++) {
      pages.push(i);
    }
    return pages;
  };

  return (
    <div className="space-y-6">
      {/* Section 1: Column Details */}
      <Card>
        <CardHeader>
          <CardTitle>Column Details</CardTitle>
          <CardDescription>
            Showing {((columnPage - 1) * itemsPerPage) + 1} to {Math.min(columnPage * itemsPerPage, regularColumns.length)} of {regularColumns.length} columns
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* ID Columns (Collapsible) */}
          {idColumnDetails.length > 0 && (
            <Collapsible open={idColumnsOpen} onOpenChange={setIdColumnsOpen}>
              <Card className="border-yellow-500/50">
                <CollapsibleTrigger className="w-full">
                  <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Key className="w-4 h-4 text-yellow-500" />
                        <CardTitle className="text-sm">ID Columns ({idColumnDetails.length})</CardTitle>
                      </div>
                      <ChevronDown className={`w-4 h-4 transition-transform ${idColumnsOpen ? 'rotate-180' : ''}`} />
                    </div>
                  </CardHeader>
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <CardContent className="pt-0">
                    <div className="space-y-2">
                      {idColumnDetails.map((col) => (
                        <div key={col.name} className="flex items-center justify-between p-2 rounded-lg bg-muted/50">
                          <span className="font-mono text-sm">{col.name}</span>
                          <Badge variant="outline" className={getDtypeColor(col.dtype)}>
                            {col.dtype}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </CollapsibleContent>
              </Card>
            </Collapsible>
          )}

          {/* Regular Columns */}
          <div className="grid gap-4">
            {paginatedColumns.map((col) => (
              <Card key={col.name} className="border">
                <CardContent className="pt-6">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div className="flex-1 space-y-2">
                      <div className="flex items-center gap-2">
                        {getDtypeIcon(col.dtype)}
                        <span className="font-mono font-semibold">{col.name}</span>
                        <Badge variant="outline" className={getDtypeColor(col.dtype)}>
                          {col.dtype}
                        </Badge>
                      </div>
                      
                      <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger>
                              <span>Unique: <strong>{col.unique_count.toLocaleString()}</strong></span>
                            </TooltipTrigger>
                            <TooltipContent>Number of unique values</TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                        
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger>
                              <span>Missing: <strong>{col.missing_count.toLocaleString()} ({col.missing_percent.toFixed(2)}%)</strong></span>
                            </TooltipTrigger>
                            <TooltipContent>Missing values count and percentage</TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </div>

                      {col.sample_values && col.sample_values.length > 0 && (
                        <div className="text-xs text-muted-foreground">
                          <span className="font-semibold">Sample: </span>
                          {col.sample_values.slice(0, 3).map((val, idx) => (
                            <span key={idx}>
                              {idx > 0 && ", "}
                              <code className="bg-muted px-1 rounded">{String(val)}</code>
                            </span>
                          ))}
                        </div>
                      )}
                    </div>

                    <div className="flex items-center gap-2">
                      {getQualityBadge(col.missing_percent)}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Column Pagination */}
          {totalColumnPages > 1 && (
            <Pagination>
              <PaginationContent>
                <PaginationItem>
                  <PaginationPrevious
                    onClick={() => setColumnPage(p => Math.max(1, p - 1))}
                    className={columnPage === 1 ? "pointer-events-none opacity-50" : "cursor-pointer"}
                  />
                </PaginationItem>
                
                {getPageNumbers(columnPage, totalColumnPages).map((pageNum) => (
                  <PaginationItem key={pageNum}>
                    <PaginationLink
                      onClick={() => setColumnPage(pageNum)}
                      isActive={columnPage === pageNum}
                      className="cursor-pointer"
                    >
                      {pageNum}
                    </PaginationLink>
                  </PaginationItem>
                ))}
                
                <PaginationItem>
                  <PaginationNext
                    onClick={() => setColumnPage(p => Math.min(totalColumnPages, p + 1))}
                    className={columnPage === totalColumnPages ? "pointer-events-none opacity-50" : "cursor-pointer"}
                  />
                </PaginationItem>
              </PaginationContent>
            </Pagination>
          )}
        </CardContent>
      </Card>

      {/* Section 2: Numeric Statistics */}
      {numericStats.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Numeric Statistics</CardTitle>
            <CardDescription>
              Statistical summary for {numericStats.length} numeric columns
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="w-full whitespace-nowrap rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="font-bold">Column</TableHead>
                    <TableHead className="font-bold text-right">Mean</TableHead>
                    <TableHead className="font-bold text-right">Std Dev</TableHead>
                    <TableHead className="font-bold text-right">Min</TableHead>
                    <TableHead className="font-bold text-right">25%</TableHead>
                    <TableHead className="font-bold text-right">Median</TableHead>
                    <TableHead className="font-bold text-right">75%</TableHead>
                    <TableHead className="font-bold text-right">Max</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {paginatedStats.map(([column, stats]) => (
                    <TableRow key={column}>
                      <TableCell className="font-mono font-medium">{column}</TableCell>
                      <TableCell className="text-right">{stats.mean?.toFixed(2) ?? 'N/A'}</TableCell>
                      <TableCell className="text-right">{stats.std?.toFixed(2) ?? 'N/A'}</TableCell>
                      <TableCell className="text-right">{stats.min?.toFixed(2) ?? 'N/A'}</TableCell>
                      <TableCell className="text-right">{stats['25%']?.toFixed(2) ?? 'N/A'}</TableCell>
                      <TableCell className="text-right">{stats['50%']?.toFixed(2) ?? 'N/A'}</TableCell>
                      <TableCell className="text-right">{stats['75%']?.toFixed(2) ?? 'N/A'}</TableCell>
                      <TableCell className="text-right">{stats.max?.toFixed(2) ?? 'N/A'}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              <ScrollBar orientation="horizontal" />
            </ScrollArea>

            {/* Stats Pagination */}
            {totalStatsPages > 1 && (
              <div className="mt-4">
                <Pagination>
                  <PaginationContent>
                    <PaginationItem>
                      <PaginationPrevious
                        onClick={() => setStatsPage(p => Math.max(1, p - 1))}
                        className={statsPage === 1 ? "pointer-events-none opacity-50" : "cursor-pointer"}
                      />
                    </PaginationItem>
                    
                    {getPageNumbers(statsPage, totalStatsPages).map((pageNum) => (
                      <PaginationItem key={pageNum}>
                        <PaginationLink
                          onClick={() => setStatsPage(pageNum)}
                          isActive={statsPage === pageNum}
                          className="cursor-pointer"
                        >
                          {pageNum}
                        </PaginationLink>
                      </PaginationItem>
                    ))}
                    
                    <PaginationItem>
                      <PaginationNext
                        onClick={() => setStatsPage(p => Math.min(totalStatsPages, p + 1))}
                        className={statsPage === totalStatsPages ? "pointer-events-none opacity-50" : "cursor-pointer"}
                      />
                    </PaginationItem>
                  </PaginationContent>
                </Pagination>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Section 3: Top Correlations */}
      {topCorrelations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Top Correlations</CardTitle>
            <CardDescription>
              Showing strongest correlations (|r| {'>'} 0.5)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {topCorrelations.map((pair, idx) => {
                const absCorr = Math.abs(pair.correlation);
                const isPositive = pair.correlation > 0;
                const percentage = absCorr * 100;
                
                return (
                  <div key={idx} className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        {isPositive ? (
                          <TrendingUp className="w-4 h-4 text-green-500" />
                        ) : (
                          <TrendingDown className="w-4 h-4 text-red-500" />
                        )}
                        <span className="font-mono">{pair.col1}</span>
                        <span className="text-muted-foreground">↔</span>
                        <span className="font-mono">{pair.col2}</span>
                      </div>
                      <Badge variant="outline" className={isPositive ? "bg-green-500/10 text-green-500" : "bg-red-500/10 text-red-500"}>
                        {pair.correlation.toFixed(3)}
                      </Badge>
                    </div>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <div
                        className={`h-full ${isPositive ? 'bg-green-500' : 'bg-red-500'}`}
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
