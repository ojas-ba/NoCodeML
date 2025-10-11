"""EDA service for exploratory data analysis operations."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException, status

from app.models.dataset import Dataset


# Constants
SAMPLE_THRESHOLD = 10000
RANDOM_SEED = 42
MAX_SAMPLE_VALUES = 5


async def load_dataset(dataset_id: UUID, user_id: int, db: AsyncSession) -> Tuple[pd.DataFrame, Dataset]:
    """Load dataset from storage and verify ownership."""
    query = select(Dataset).where(
        Dataset.id == dataset_id,
        Dataset.user_id == user_id
    )
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found or access denied"
        )
    
    storage_path = Path(dataset.storage_path)
    if not storage_path.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Dataset file not found on storage"
        )
    
    try:
        # Read file based on extension
        file_ext = storage_path.suffix.lower()
        if file_ext == '.csv':
            df = pd.read_csv(storage_path)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(storage_path)
        elif file_ext == '.parquet':
            df = pd.read_parquet(storage_path)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format: {file_ext}"
            )
        
        return df, dataset
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read dataset: {str(e)}"
        )


def detect_id_columns(df: pd.DataFrame) -> List[str]:
    """Auto-detect ID columns based on name patterns and uniqueness."""
    id_columns = []
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Name-based detection
        if any(pattern in col_lower for pattern in ['id', '_id', 'id_', 'index', 'key']):
            id_columns.append(col)
            continue
        
        # Uniqueness-based detection (100% unique and numeric/string type)
        if df[col].nunique() == len(df):
            if df[col].dtype in ['int64', 'int32', 'object', 'string']:
                id_columns.append(col)
    
    return id_columns


def get_column_info(df: pd.DataFrame, id_columns: List[str]) -> List[Dict[str, Any]]:
    """Extract detailed information for each column."""
    columns_info = []
    
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_percent = (missing_count / len(df)) * 100
        unique_count = int(df[col].nunique())
        
        # Get sample values (non-null)
        sample_values = df[col].dropna().head(MAX_SAMPLE_VALUES).tolist()
        
        # Convert numpy types to native Python types
        sample_values = [
            int(x) if isinstance(x, (np.integer, np.int64)) else
            float(x) if isinstance(x, (np.floating, np.float64)) else
            str(x)
            for x in sample_values
        ]
        
        columns_info.append({
            'name': col,
            'dtype': str(df[col].dtype),
            'missing_count': missing_count,
            'missing_percent': round(missing_percent, 2),
            'unique_count': unique_count,
            'is_id_column': col in id_columns,
            'sample_values': sample_values
        })
    
    return columns_info


def categorize_columns(df: pd.DataFrame, id_columns: List[str]) -> Tuple[List[str], List[str]]:
    """Categorize columns into numeric and categorical (keeping ID columns for now)."""
    numeric_cols = []
    categorical_cols = []
    
    for col in df.columns:
        # Keep ID columns in the analysis - user requested this
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    return numeric_cols, categorical_cols


def compute_statistics(df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
    """Compute statistical summary for numeric columns."""
    if not numeric_columns:
        return {}
    
    stats_df = df[numeric_columns].describe()
    
    # Convert to dictionary with proper type conversion
    statistics = {}
    for col in numeric_columns:
        col_stats = {}
        for stat_name in stats_df.index:
            value = stats_df.loc[stat_name, col]
            if pd.notna(value):
                col_stats[stat_name] = float(value)
            else:
                col_stats[stat_name] = None
        statistics[col] = col_stats
    
    return statistics


def compute_correlations(df: pd.DataFrame, numeric_columns: List[str]) -> Optional[Dict[str, Any]]:
    """Compute correlation matrix for numeric columns."""
    if len(numeric_columns) < 2:
        return None
    
    try:
        corr_matrix = df[numeric_columns].corr()
        
        # Convert to dictionary format
        correlations = {
            'columns': numeric_columns,
            'matrix': corr_matrix.values.tolist(),
            'pairs': []
        }
        
        # Extract strong correlations (|r| > 0.7, excluding diagonal)
        for i, col1 in enumerate(numeric_columns):
            for j, col2 in enumerate(numeric_columns):
                if i < j:  # Avoid duplicates
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        correlations['pairs'].append({
                            'col1': col1,
                            'col2': col2,
                            'correlation': round(float(corr_value), 3)
                        })
        
        return correlations
    except Exception:
        return None


def compute_missing_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary of missing data."""
    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isna().sum().sum())
    missing_percent = (total_missing / total_cells) * 100 if total_cells > 0 else 0
    
    columns_with_missing = []
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        if missing_count > 0:
            columns_with_missing.append({
                'column': col,
                'missing_count': missing_count,
                'missing_percent': round((missing_count / len(df)) * 100, 2)
            })
    
    # Sort by missing count descending
    columns_with_missing.sort(key=lambda x: x['missing_count'], reverse=True)
    
    return {
        'total_missing': total_missing,
        'total_cells': total_cells,
        'missing_percent': round(missing_percent, 2),
        'columns_with_missing': columns_with_missing
    }


def get_preview_data(df: pd.DataFrame, max_rows: int = 100) -> Dict[str, Any]:
    """
    Get preview data for the dataset with pagination metadata.
    
    Args:
        df: DataFrame to preview
        max_rows: Maximum number of rows to include (default 100)
    
    Returns:
        Dictionary with columns, rows, total_rows, and page_size
    """
    preview_df = df.head(max_rows)
    
    # Convert DataFrame to list of dictionaries
    # Handle NaN/None values and convert numpy types
    rows = []
    for _, row in preview_df.iterrows():
        row_dict = {}
        for col in df.columns:
            value = row[col]
            
            # Handle missing values
            if pd.isna(value):
                row_dict[col] = None
            # Convert numpy types to Python native types
            elif isinstance(value, (np.integer, np.int64)):
                row_dict[col] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                # Round to 2 decimal places for display
                row_dict[col] = round(float(value), 2)
            else:
                row_dict[col] = str(value)
        
        rows.append(row_dict)
    
    return {
        'columns': df.columns.tolist(),
        'rows': rows,
        'total_rows': len(df),
        'page_size': max_rows
    }


async def get_eda_summary(dataset_id: UUID, user_id: int, db: AsyncSession) -> Dict[str, Any]:
    """
    Generate comprehensive EDA summary for a dataset.
    
    Returns:
        Dictionary containing dataset info, column info, statistics, correlations, 
        missing data summary, and preview data.
    """
    # Load dataset
    df, dataset = await load_dataset(dataset_id, user_id, db)
    
    # Detect ID columns
    id_columns = detect_id_columns(df)
    
    # Categorize columns
    numeric_columns, categorical_columns = categorize_columns(df, id_columns)
    
    # Get column information
    columns_info = get_column_info(df, id_columns)
    
    # Compute statistics
    statistics = compute_statistics(df, numeric_columns)
    
    # Compute correlations
    correlations = compute_correlations(df, numeric_columns)
    
    # Missing data summary
    missing_data_summary = compute_missing_data_summary(df)
    
    # Get preview data (first 100 rows for pagination)
    preview_data = get_preview_data(df, max_rows=100)
    
    # Dataset info
    dataset_info = {
        'id': str(dataset.id),
        'name': dataset.name,
        'row_count': len(df),
        'column_count': len(df.columns),
        'file_size_bytes': dataset.file_size_bytes,
        'file_name': dataset.file_name,
        'memory_usage_bytes': int(df.memory_usage(deep=True).sum())
    }
    
    return {
        'dataset_info': dataset_info,
        'columns': columns_info,
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'id_columns': id_columns,
        'statistics': statistics,
        'correlations': correlations,
        'missing_data_summary': missing_data_summary,
        'preview_data': preview_data
    }


def sample_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool, int, int]:
    """Sample dataframe if it exceeds threshold."""
    total_rows = len(df)
    
    if total_rows > SAMPLE_THRESHOLD:
        df_sampled = df.sample(n=SAMPLE_THRESHOLD, random_state=RANDOM_SEED)
        return df_sampled, True, total_rows, SAMPLE_THRESHOLD
    
    return df, False, total_rows, total_rows


def create_histogram(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Generate histogram plot data."""
    data = df[column].dropna()
    
    trace = {
        'x': data.tolist(),
        'type': 'histogram',
        'name': column,
        'marker': {
            'color': 'rgba(99, 102, 241, 0.7)',
            'line': {'color': 'rgba(99, 102, 241, 1)', 'width': 1}
        },
        'autobinx': True
    }
    
    # Calculate mean and median for reference lines
    mean_val = float(data.mean())
    median_val = float(data.median())
    
    layout = {
        'title': f'Distribution of {column}',
        'xaxis': {'title': column},
        'yaxis': {'title': 'Frequency'},
        'shapes': [
            {
                'type': 'line',
                'x0': mean_val,
                'x1': mean_val,
                'y0': 0,
                'y1': 1,
                'yref': 'paper',
                'line': {'color': 'red', 'width': 2, 'dash': 'dash'}
            },
            {
                'type': 'line',
                'x0': median_val,
                'x1': median_val,
                'y0': 0,
                'y1': 1,
                'yref': 'paper',
                'line': {'color': 'green', 'width': 2, 'dash': 'dash'}
            }
        ],
        'annotations': [
            {
                'x': mean_val,
                'y': 0.95,
                'yref': 'paper',
                'text': f'Mean: {mean_val:.2f}',
                'showarrow': False,
                'font': {'color': 'red'}
            },
            {
                'x': median_val,
                'y': 0.85,
                'yref': 'paper',
                'text': f'Median: {median_val:.2f}',
                'showarrow': False,
                'font': {'color': 'green'}
            }
        ]
    }
    
    return {'data': [trace], 'layout': layout}


def create_scatter(df: pd.DataFrame, x_column: str, y_column: str, group_by: Optional[str] = None) -> Dict[str, Any]:
    """Generate scatter plot data."""
    # Remove rows with missing values in relevant columns
    cols_to_check = [x_column, y_column]
    if group_by:
        cols_to_check.append(group_by)
    
    df_clean = df[cols_to_check].dropna()
    
    if group_by and group_by in df_clean.columns:
        # Group by category
        traces = []
        for category in df_clean[group_by].unique():
            mask = df_clean[group_by] == category
            traces.append({
                'x': df_clean.loc[mask, x_column].tolist(),
                'y': df_clean.loc[mask, y_column].tolist(),
                'type': 'scatter',
                'mode': 'markers',
                'name': str(category),
                'marker': {'size': 8}
            })
        data = traces
    else:
        # Single scatter plot
        data = [{
            'x': df_clean[x_column].tolist(),
            'y': df_clean[y_column].tolist(),
            'type': 'scatter',
            'mode': 'markers',
            'marker': {
                'size': 8,
                'color': 'rgba(99, 102, 241, 0.7)',
                'line': {'color': 'rgba(99, 102, 241, 1)', 'width': 1}
            }
        }]
    
    # Calculate correlation
    try:
        correlation = df_clean[[x_column, y_column]].corr().iloc[0, 1]
        corr_text = f'Correlation: {correlation:.3f}'
    except Exception:
        corr_text = ''
    
    layout = {
        'title': f'{y_column} vs {x_column}',
        'xaxis': {'title': x_column},
        'yaxis': {'title': y_column},
        'annotations': [
            {
                'x': 0.05,
                'y': 0.95,
                'xref': 'paper',
                'yref': 'paper',
                'text': corr_text,
                'showarrow': False,
                'font': {'size': 12}
            }
        ] if corr_text else []
    }
    
    return {'data': data, 'layout': layout}


def create_box_plot(df: pd.DataFrame, column: str, group_by: Optional[str] = None) -> Dict[str, Any]:
    """Generate box plot data."""
    if group_by and group_by in df.columns:
        # Group by category
        traces = []
        for category in df[group_by].dropna().unique():
            mask = df[group_by] == category
            traces.append({
                'y': df.loc[mask, column].dropna().tolist(),
                'type': 'box',
                'name': str(category),
                'boxmean': 'sd'
            })
        data = traces
        title = f'{column} by {group_by}'
    else:
        # Single box plot
        data = [{
            'y': df[column].dropna().tolist(),
            'type': 'box',
            'name': column,
            'marker': {'color': 'rgba(99, 102, 241, 0.7)'},
            'boxmean': 'sd'
        }]
        title = f'Box Plot of {column}'
    
    layout = {
        'title': title,
        'yaxis': {'title': column}
    }
    
    return {'data': data, 'layout': layout}


def create_correlation_heatmap(df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
    """Generate correlation heatmap data."""
    if len(numeric_columns) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Need at least 2 numeric columns for correlation heatmap"
        )
    
    corr_matrix = df[numeric_columns].corr()
    
    data = [{
        'z': corr_matrix.values.tolist(),
        'x': numeric_columns,
        'y': numeric_columns,
        'type': 'heatmap',
        'colorscale': 'RdBu',
        'zmid': 0,
        'zmin': -1,
        'zmax': 1,
        'colorbar': {'title': 'Correlation'}
    }]
    
    layout = {
        'title': 'Correlation Heatmap',
        'xaxis': {'title': '', 'tickangle': -45},
        'yaxis': {'title': ''},
        'height': 500 + len(numeric_columns) * 20
    }
    
    return {'data': data, 'layout': layout}


def create_bar_chart(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Generate bar chart for categorical column."""
    value_counts = df[column].value_counts().head(20)  # Top 20 categories
    
    data = [{
        'x': value_counts.index.tolist(),
        'y': value_counts.values.tolist(),
        'type': 'bar',
        'marker': {
            'color': 'rgba(99, 102, 241, 0.7)',
            'line': {'color': 'rgba(99, 102, 241, 1)', 'width': 1}
        }
    }]
    
    layout = {
        'title': f'Distribution of {column}',
        'xaxis': {'title': column, 'tickangle': -45},
        'yaxis': {'title': 'Count'}
    }
    
    return {'data': data, 'layout': layout}


async def generate_plot_data(
    dataset_id: UUID,
    plot_type: str,
    x_column: str,
    y_column: Optional[str],
    group_by: Optional[str],
    user_id: int,
    db: AsyncSession
) -> Dict[str, Any]:
    """
    Generate plot data based on plot type and column selections.
    
    Args:
        dataset_id: UUID of the dataset
        plot_type: Type of plot (histogram, scatter, box, correlation, bar)
        x_column: Column for x-axis
        y_column: Column for y-axis (optional)
        group_by: Column to group by (optional)
        user_id: User ID for authorization
        db: Database session
        
    Returns:
        Dictionary with plot data, layout, and sampling information
    """
    # Load dataset
    df, dataset = await load_dataset(dataset_id, user_id, db)
    
    # Sample if necessary
    df_plot, is_sampled, total_rows, displayed_rows = sample_dataframe(df)
    
    # Validate columns
    if x_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Column '{x_column}' not found in dataset"
        )
    
    if y_column and y_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Column '{y_column}' not found in dataset"
        )
    
    # Generate plot based on type
    try:
        if plot_type == 'histogram':
            plot_result = create_histogram(df_plot, x_column)
        elif plot_type == 'scatter':
            if not y_column:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Scatter plot requires both x_column and y_column"
                )
            plot_result = create_scatter(df_plot, x_column, y_column, group_by)
        elif plot_type == 'box':
            plot_result = create_box_plot(df_plot, x_column, group_by)
        elif plot_type == 'correlation':
            # For correlation, use all numeric columns (excluding IDs)
            id_columns = detect_id_columns(df)
            numeric_columns, _ = categorize_columns(df, id_columns)
            plot_result = create_correlation_heatmap(df_plot, numeric_columns)
        elif plot_type == 'bar':
            plot_result = create_bar_chart(df_plot, x_column)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported plot type: {plot_type}"
            )
        
        return {
            'data': plot_result['data'],
            'layout': plot_result['layout'],
            'is_sampled': is_sampled,
            'total_rows': total_rows,
            'displayed_rows': displayed_rows,
            'plot_type': plot_type
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate plot: {str(e)}"
        )
