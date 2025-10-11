import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { 
  Pagination, 
  PaginationContent, 
  PaginationItem, 
  PaginationLink, 
  PaginationNext, 
  PaginationPrevious 
} from "@/components/ui/pagination";
import { Database, FileText, AlertCircle, Rows3 } from "lucide-react";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";

interface DatasetInfo {
  id: string;
  name: string;
  row_count: number;
  column_count: number;
  file_size_bytes: number;
  file_name: string;
  memory_usage_bytes: number;
}

interface PreviewData {
  columns: string[];
  rows: Array<Record<string, any>>;
  total_rows: number;
  page_size: number;
}

interface MissingDataSummary {
  total_missing: number;
  total_cells: number;
  missing_percent: number;
  columns_with_missing: Array<{
    column: string;
    missing_count: number;
    missing_percent: number;
  }>;
}

interface DatasetOverviewProps {
  datasetInfo: DatasetInfo;
  previewData: PreviewData;
  missingDataSummary: MissingDataSummary;
}

const formatBytes = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
};

const formatNumber = (num: number): string => {
  return num.toLocaleString();
};

export const DatasetOverview = ({ datasetInfo, previewData, missingDataSummary }: DatasetOverviewProps) => {
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 10;

  // Calculate pagination
  const totalPages = Math.ceil(previewData.rows.length / rowsPerPage);
  const paginatedRows = useMemo(() => {
    const startIndex = (currentPage - 1) * rowsPerPage;
    const endIndex = startIndex + rowsPerPage;
    return previewData.rows.slice(startIndex, endIndex);
  }, [previewData.rows, currentPage, rowsPerPage]);

  // Generate page numbers to display
  const getPageNumbers = () => {
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
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Rows</CardTitle>
            <Rows3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatNumber(datasetInfo.row_count)}</div>
            <p className="text-xs text-muted-foreground mt-1">
              Dataset records
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Columns</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{datasetInfo.column_count}</div>
            <p className="text-xs text-muted-foreground mt-1">
              Features available
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">File Size</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatBytes(datasetInfo.file_size_bytes)}</div>
            <p className="text-xs text-muted-foreground mt-1">
              {formatBytes(datasetInfo.memory_usage_bytes)} in memory
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Missing Data</CardTitle>
            <AlertCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {missingDataSummary.missing_percent.toFixed(2)}%
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {formatNumber(missingDataSummary.total_missing)} of {formatNumber(missingDataSummary.total_cells)} cells
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Data Preview Table */}
      <Card>
        <CardHeader>
          <CardTitle>Data Preview</CardTitle>
          <p className="text-sm text-muted-foreground">
            Showing {((currentPage - 1) * rowsPerPage) + 1} to {Math.min(currentPage * rowsPerPage, previewData.rows.length)} of {previewData.rows.length} rows
          </p>
        </CardHeader>
        <CardContent>
          <ScrollArea className="w-full whitespace-nowrap rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-12 font-bold">#</TableHead>
                  {previewData.columns.map((col) => (
                    <TableHead key={col} className="font-bold min-w-[120px]">
                      {col}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {paginatedRows.map((row, idx) => (
                  <TableRow key={idx}>
                    <TableCell className="font-medium text-muted-foreground">
                      {((currentPage - 1) * rowsPerPage) + idx + 1}
                    </TableCell>
                    {previewData.columns.map((col) => (
                      <TableCell key={col}>
                        {row[col] === null || row[col] === undefined ? (
                          <span className="text-muted-foreground italic">null</span>
                        ) : (
                          <span>{String(row[col])}</span>
                        )}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            <ScrollBar orientation="horizontal" />
          </ScrollArea>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="mt-4">
              <Pagination>
                <PaginationContent>
                  <PaginationItem>
                    <PaginationPrevious
                      onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                      className={currentPage === 1 ? "pointer-events-none opacity-50" : "cursor-pointer"}
                    />
                  </PaginationItem>
                  
                  {getPageNumbers().map((pageNum) => (
                    <PaginationItem key={pageNum}>
                      <PaginationLink
                        onClick={() => setCurrentPage(pageNum)}
                        isActive={currentPage === pageNum}
                        className="cursor-pointer"
                      >
                        {pageNum}
                      </PaginationLink>
                    </PaginationItem>
                  ))}
                  
                  <PaginationItem>
                    <PaginationNext
                      onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                      className={currentPage === totalPages ? "pointer-events-none opacity-50" : "cursor-pointer"}
                    />
                  </PaginationItem>
                </PaginationContent>
              </Pagination>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
