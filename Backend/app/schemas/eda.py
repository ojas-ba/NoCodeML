"""EDA (Exploratory Data Analysis) schema definitions."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from uuid import UUID


class PlotRequest(BaseModel):
    """Request model for generating plots."""
    plot_type: str = Field(..., description="Type of plot: histogram, scatter, box, correlation, bar")
    x_column: str = Field(..., description="Column for x-axis")
    y_column: Optional[str] = Field(None, description="Column for y-axis (for scatter plots)")
    group_by: Optional[str] = Field(None, description="Column to group by (for box plots)")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional plot options")


class ColumnInfo(BaseModel):
    """Information about a single column."""
    name: str
    dtype: str
    missing_count: int
    missing_percent: float
    unique_count: int
    is_id_column: bool
    sample_values: Optional[List[Any]] = None


class ColumnStatistics(BaseModel):
    """Statistics for numeric columns."""
    column: str
    count: float
    mean: float
    std: float
    min: float
    q25: float
    median: float
    q75: float
    max: float


class MissingDataSummary(BaseModel):
    """Summary of missing data in the dataset."""
    total_missing: int
    total_cells: int
    missing_percent: float
    columns_with_missing: List[Dict[str, Any]]


class PreviewData(BaseModel):
    """Preview data with pagination metadata."""
    columns: List[str] = Field(..., description="Column names")
    rows: List[Dict[str, Any]] = Field(..., description="Preview rows (up to 100)")
    total_rows: int = Field(..., description="Total number of rows in dataset")
    page_size: int = Field(..., description="Number of rows included in preview")


class EDAResponse(BaseModel):
    """Comprehensive EDA summary response."""
    dataset_info: Dict[str, Any] = Field(..., description="Basic dataset information")
    columns: List[ColumnInfo] = Field(..., description="Detailed column information")
    numeric_columns: List[str] = Field(..., description="List of numeric column names")
    categorical_columns: List[str] = Field(..., description="List of categorical column names")
    id_columns: List[str] = Field(..., description="Auto-detected ID columns")
    statistics: Dict[str, Any] = Field(..., description="Statistical summary")
    correlations: Optional[Dict[str, Any]] = Field(None, description="Correlation matrix for numeric columns")
    missing_data_summary: MissingDataSummary = Field(..., description="Missing data analysis")
    preview_data: PreviewData = Field(..., description="Dataset preview with pagination")


class PlotResponse(BaseModel):
    """Response model for plot generation."""
    data: List[Dict[str, Any]] = Field(..., description="Plotly data traces")
    layout: Dict[str, Any] = Field(..., description="Plotly layout configuration")
    is_sampled: bool = Field(..., description="Whether data was sampled for performance")
    total_rows: int = Field(..., description="Total rows in dataset")
    displayed_rows: int = Field(..., description="Number of rows displayed in plot")
    plot_type: str = Field(..., description="Type of plot generated")
