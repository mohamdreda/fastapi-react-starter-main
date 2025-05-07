import pandas as pd
import dask.dataframe as dd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats
import json
import os
from typing import Dict, List, Any, Tuple, Optional

async def generate_visualizations(file_path: str, file_type: str, viz_type: str):
    """Main visualization generation controller"""
    df = read_data_file(file_path, file_type)
    
    # Improved sampling for large datasets
    if isinstance(df, dd.DataFrame):
        try:
            nrows = df.shape[0].compute()
            sample_size = min(max(int(nrows * 0.01), 10000), nrows)
            df = df.sample(frac=sample_size/nrows)
        except:
            df = df.head(10000)
    
    # Basic visualizations
    if viz_type == "missing":
        return await generate_missing_analysis(df)
    elif viz_type == "duplicates":
        return await generate_duplicates_analysis(df)
    elif viz_type == "categorical":
        return await generate_categorical_analysis(df)
    elif viz_type == "outliers":
        return await generate_outlier_analysis(df)
    
    # New advanced visualizations
    elif viz_type == "structure":
        return await generate_structure_analysis(df)
    elif viz_type == "distribution":
        return await generate_distribution_analysis(df)
    elif viz_type == "correlation":
        return await generate_correlation_analysis(df)
    elif viz_type == "summary":
        return await generate_summary_statistics(df)
    elif viz_type == "categorical_consistency":
        return await generate_categorical_consistency(df)
    else:
        raise ValueError(f"Unsupported visualization type: {viz_type}")

def read_data_file(file_path: str, file_type: str):
    from app.services.upload import read_dataframe
    from pathlib import Path
    return read_dataframe(Path(file_path), file_type)

async def generate_missing_analysis(df) -> dict:
    """Generate missing values visualization and stats"""
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Convert 'NaN' strings to actual NaN values
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            # Replace 'NaN', 'nan', 'NA', empty strings with NaN
            if isinstance(df_copy, dd.DataFrame):
                df_copy[col] = df_copy[col].map(lambda x: pd.NA if pd.isna(x) or 
                                              (isinstance(x, str) and x.lower() in ['nan', 'na', '']) else x, 
                                              meta=(col, df_copy[col].dtype))
            else:
                df_copy[col] = df_copy[col].replace(['nan', 'NaN', 'NA', ''], pd.NA)
                df_copy[col] = df_copy[col].apply(lambda x: pd.NA if pd.isna(x) else x)
    
    # Calculate missing values
    if isinstance(df_copy, dd.DataFrame):
        missing_by_column = df_copy.isnull().sum().compute()
        total_missing = missing_by_column.sum()
        total_cells = df_copy.size.compute()
    else:
        missing_by_column = df_copy.isnull().sum()
        total_missing = missing_by_column.sum()
        total_cells = df_copy.size
    
    missing_percentage = (total_missing / total_cells * 100) if total_cells > 0 else 0.0
    
    # Create missing values heatmap
    if isinstance(df_copy, dd.DataFrame):
        sample_df = df_copy.head(1000)
    else:
        # Handle case where dataset has fewer than 1000 rows
        sample_size = min(1000, len(df_copy))
        sample_df = df_copy.sample(n=sample_size) if sample_size > 0 else df_copy
    
    # Create heatmap
    fig = px.imshow(sample_df.isnull().T,
                   title="Missing Values Distribution",
                   color_continuous_scale='Viridis',
                   labels=dict(x="Row Index", y="Column", color="Is Missing"))
    
    # Create column-wise missing values bar chart
    missing_pct = (missing_by_column / (df_copy.shape[0] if isinstance(df_copy, pd.DataFrame) else df_copy.shape[0].compute())) * 100
    missing_pct = missing_pct.sort_values(ascending=False)
    
    # Only include columns with missing values
    missing_pct = missing_pct[missing_pct > 0]
    
    if len(missing_pct) > 0:
        fig2 = px.bar(
            x=missing_pct.index.tolist(),
            y=missing_pct.values.tolist(),
            title="Missing Values by Column (%)",
            labels={"x": "Column", "y": "Missing (%)"}
        )
        fig2.update_layout(xaxis_tickangle=-45)
    else:
        # Create empty figure if no missing values
        fig2 = go.Figure()
        fig2.update_layout(title="No Missing Values Found")
    
    # Prepare per-column missing data
    per_column_data = {}
    for col, count in missing_by_column.items():
        if count > 0:
            per_column_data[col] = {
                "count": int(count),
                "percentage": round(float(missing_pct.get(col, 0)), 2)
            }
    
    return {
        "plot": plotly_to_json(fig),
        "column_plot": plotly_to_json(fig2),
        "stats": {
            "total_missing": int(total_missing),
            "missing_percentage": round(missing_percentage, 2),
            "total_cells": int(total_cells),
            "per_column": per_column_data
        }
    }

async def generate_duplicates_analysis(df) -> dict:
    """Analyze and visualize duplicate rows"""
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Detect exact duplicates (keep=False marks all duplicates, not just second occurrences)
    if isinstance(df_copy, dd.DataFrame):
        exact_duplicates = df_copy.duplicated(keep=False).sum().compute()
        total_rows = df_copy.shape[0].compute()
    else:
        exact_duplicates = df_copy.duplicated(keep=False).sum()
        total_rows = len(df_copy)
    
    # Prepare data for visualization
    duplicate_info = {
        "exact_duplicates": int(exact_duplicates),
        "total_rows": int(total_rows)
    }
    
    # Find near-duplicates (focusing on common data columns, not IDs)
    try:
        # Identify potential data columns (exclude ID-like columns)
        exclude_cols = [col for col in df_copy.columns if col.lower() in ['id', 'index', 'key', 'uuid']]
        data_cols = [col for col in df_copy.columns if col not in exclude_cols]
        
        # For near-duplicates, focus on string columns that might have case differences
        string_cols = df_copy.select_dtypes(include=['object']).columns.tolist()
        
        # Find case-insensitive duplicates in string columns
        near_duplicates = 0
        duplicate_examples = []
        
        if not isinstance(df_copy, dd.DataFrame) and len(string_cols) > 0:
            # Create lowercase version for string columns
            df_lower = df_copy.copy()
            for col in string_cols:
                if col in df_lower.columns:
                    df_lower[col] = df_lower[col].astype(str).str.lower()
            
            # Find rows that are duplicates when case-insensitive but not exact duplicates
            case_insensitive_dupes = df_lower.duplicated(subset=string_cols, keep=False)
            exact_dupes = df_copy.duplicated(subset=string_cols, keep=False)
            near_dupes = case_insensitive_dupes & ~exact_dupes
            near_duplicates = near_dupes.sum()
            
            # Get examples of near-duplicates for the report
            if near_duplicates > 0:
                near_dupe_indices = df_copy[near_dupes].index[:5]  # Limit to 5 examples
                for idx in near_dupe_indices:
                    row = df_copy.loc[idx]
                    # Find the matching case-insensitive rows
                    matches = []
                    for col in string_cols:
                        if pd.notna(row[col]):
                            matches_col = df_copy[df_copy[col].str.lower() == str(row[col]).lower()]
                            if len(matches_col) > 1:  # More than 1 means there's a match besides itself
                                matches.append({
                                    "column": col,
                                    "value": row[col],
                                    "matches": matches_col[col].tolist()[:3]  # Limit to 3 matches
                                })
                    if matches:
                        duplicate_examples.append({
                            "row_index": int(idx),
                            "matches": matches
                        })
        
        duplicate_info["near_duplicates"] = int(near_duplicates)
        duplicate_info["total_duplicates"] = int(exact_duplicates + near_duplicates)
        duplicate_info["duplicate_examples"] = duplicate_examples[:5]  # Limit examples
    except Exception as e:
        # If near-duplicate detection fails, just use exact duplicates
        duplicate_info["near_duplicates"] = 0
        duplicate_info["total_duplicates"] = int(exact_duplicates)
        duplicate_info["error"] = str(e)
    
    # Create visualization
    categories = ["Unique Rows", "Exact Duplicates"]
    values = [total_rows - exact_duplicates, exact_duplicates]
    
    if "near_duplicates" in duplicate_info and duplicate_info["near_duplicates"] > 0:
        categories.append("Near Duplicates")
        values.append(duplicate_info["near_duplicates"])
    
    fig = px.bar(
        x=categories,
        y=values,
        title="Duplicate Rows Analysis",
        labels={"x": "Row Type", "y": "Count"},
        color=categories,
        color_discrete_map={
            "Unique Rows": "green",
            "Exact Duplicates": "red",
            "Near Duplicates": "orange"
        }
    )
    
    # Add percentage labels on top of bars
    for i, value in enumerate(values):
        percentage = (value / total_rows) * 100
        fig.add_annotation(
            x=categories[i],
            y=value,
            text=f"{percentage:.1f}%",
            showarrow=False,
            yshift=10
        )
    
    return {
        "plot": plotly_to_json(fig),
        "stats": {
            "exact_duplicates": duplicate_info["exact_duplicates"],
            "near_duplicates": duplicate_info.get("near_duplicates", 0),
            "total_duplicates": duplicate_info.get("total_duplicates", duplicate_info["exact_duplicates"]),
            "duplicate_percentage": round((duplicate_info.get("total_duplicates", duplicate_info["exact_duplicates"]) / total_rows) * 100, 2),
            "total_rows": duplicate_info["total_rows"],
            "examples": duplicate_info.get("duplicate_examples", [])
        }
    }

async def generate_categorical_analysis(df) -> dict:
    """Analyze categorical columns distribution"""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    analysis = {}
    
    for col in cat_cols:
        if isinstance(df, dd.DataFrame):
            counts = df[col].value_counts().compute()
        else:
            counts = df[col].value_counts()
        
        analysis[col] = {
            "unique_count": len(counts),
            "top_values": counts.head(5).to_dict()
        }
    
    plot_data = {}
    if cat_cols:
        col = cat_cols[0]
        # Get top 10 categories or all if fewer than 10
        sample_size = min(10, len(counts))
        sample_counts = counts.head(sample_size)
        
        if not sample_counts.empty:
            fig = px.pie(
                names=sample_counts.index,
                values=sample_counts.values,
                title=f"Category Distribution: {col}"
            )
            plot_data = plotly_to_json(fig)
    
    return {
        "plot": plot_data,
        "analysis": analysis
    }

async def generate_outlier_analysis(df) -> dict:
    """Detect and visualize outliers in numerical columns"""
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Select only numeric columns, excluding ID-like columns
    exclude_cols = [col for col in df_copy.columns if col.lower() in ['id', 'index', 'key', 'uuid']]
    num_cols = df_copy.select_dtypes(include=['number']).columns.tolist()
    num_cols = [col for col in num_cols if col not in exclude_cols]
    
    if not num_cols:
        return {"error": "No numerical columns found in the dataset"}
    
    outliers_data = {}
    plots = {}
    
    for col in num_cols:
        if isinstance(df_copy, dd.DataFrame):
            # For Dask DataFrames, compute the column first
            col_data = df_copy[col].compute()
        else:
            col_data = df_copy[col]
        
        # Skip columns with all NaN
        if col_data.isna().all():
            continue
        
        # Drop NaN for outlier detection
        col_data = col_data.dropna()
        
        # Skip if not enough data
        if len(col_data) < 5:  # Reduced minimum to catch outliers in small datasets
            continue
        
        # Use multiple methods to detect outliers
        outliers_combined = pd.Series(False, index=col_data.index)
        outlier_methods = {}
        
        # Method 1: IQR method (more robust than z-score for skewed data)
        try:
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = (col_data < lower_bound) | (col_data > upper_bound)
            outliers_combined = outliers_combined | iqr_outliers
            outlier_methods["IQR"] = {
                "count": int(iqr_outliers.sum()),
                "bounds": (float(lower_bound), float(upper_bound))
            }
        except Exception as e:
            print(f"Error in IQR calculation for {col}: {str(e)}")
        
        # Method 2: Z-score method (good for normally distributed data)
        try:
            z_scores = np.abs(stats.zscore(col_data))
            z_outliers = z_scores > 3.0  # Values more than 3 standard deviations away
            outliers_combined = outliers_combined | z_outliers
            outlier_methods["Z-score"] = {
                "count": int(z_outliers.sum()),
                "threshold": 3.0
            }
        except Exception as e:
            print(f"Error in Z-score calculation for {col}: {str(e)}")
        
        # Method 3: Isolation Forest (good for complex distributions)
        try:
            # Reshape for isolation forest
            X = col_data.values.reshape(-1, 1)
            
            # Fit isolation forest with adaptive contamination based on dataset size
            contamination = min(0.1, max(0.01, 10/len(col_data)))  # Between 1% and 10%
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            iso_predictions = iso_forest.fit_predict(X)
            iso_outliers = pd.Series(iso_predictions == -1, index=col_data.index)
            outliers_combined = outliers_combined | iso_outliers
            outlier_methods["Isolation Forest"] = {
                "count": int(iso_outliers.sum()),
                "contamination": contamination
            }
        except Exception as e:
            print(f"Error in Isolation Forest for {col}: {str(e)}")
        
        # Get final outlier values (combining all methods)
        outlier_values = col_data[outliers_combined].tolist()
        
        # Create box plot with outliers highlighted
        fig = px.box(col_data, title=f"Outlier Detection for {col}")
        
        # Add outlier points with hover information
        if outlier_values:
            # Add scatter points for outliers
            fig.add_trace(go.Scatter(
                x=[col] * len(outlier_values),
                y=outlier_values,
                mode='markers',
                marker=dict(color='red', size=8, symbol='circle'),
                name='Outliers',
                hovertemplate='Value: %{y}<br>Column: ' + col
            ))
            
            # Add a horizontal line for the mean
            fig.add_shape(
                type="line",
                x0=-0.5, x1=0.5,
                y0=col_data.mean(), y1=col_data.mean(),
                line=dict(color="green", width=2, dash="dash"),
                xref='x', yref='y'
            )
            
            # Add annotations for IQR bounds if available
            if "IQR" in outlier_methods:
                lower, upper = outlier_methods["IQR"]["bounds"]
                for bound, label in [(lower, "Lower IQR Bound"), (upper, "Upper IQR Bound")]:
                    fig.add_shape(
                        type="line",
                        x0=-0.5, x1=0.5,
                        y0=bound, y1=bound,
                        line=dict(color="orange", width=1.5, dash="dot"),
                        xref='x', yref='y'
                    )
        
        # Update layout for better visualization
        fig.update_layout(
            showlegend=True,
            xaxis_title="",
            yaxis_title=col,
            height=400
        )
        
        plots[col] = plotly_to_json(fig)
        outliers_data[col] = {
            "count": int(outliers_combined.sum()),
            "percentage": round(outliers_combined.sum() / len(col_data) * 100, 2),
            "values": outlier_values[:10],  # Limit to 10 examples
            "methods": outlier_methods
        }
    
    # Create a summary plot showing outlier counts across columns
    if outliers_data:
        columns = list(outliers_data.keys())
        counts = [outliers_data[col]["count"] for col in columns]
        percentages = [outliers_data[col]["percentage"] for col in columns]
        
        # Create a dual-axis bar chart
        fig_summary = go.Figure()
        
        # Add bars for counts
        fig_summary.add_trace(go.Bar(
            x=columns,
            y=counts,
            name="Count",
            marker_color="indianred"
        ))
        
        # Add line for percentages
        fig_summary.add_trace(go.Scatter(
            x=columns,
            y=percentages,
            name="Percentage",
            marker_color="royalblue",
            mode="lines+markers",
            yaxis="y2"
        ))
        
        # Update layout for dual y-axes
        fig_summary.update_layout(
            title="Outliers by Column",
            xaxis_title="Column",
            yaxis_title="Count",
            yaxis2=dict(
                title="Percentage",
                overlaying="y",
                side="right",
                range=[0, max(percentages) * 1.1]
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_tickangle=-45
        )
        
        plots["summary"] = plotly_to_json(fig_summary)
    
    return {
        "plots": plots,
        "stats": outliers_data
    }

async def generate_structure_analysis(df) -> dict:
    """Analyze the structure of the dataset including column names, data types, and shape"""
    # Get basic dataset information
    if isinstance(df, dd.DataFrame):
        shape = (df.shape[0].compute(), df.shape[1])
        memory_usage = None  # Dask doesn't have a direct memory_usage method
    else:
        shape = df.shape
        memory_usage = df.memory_usage(deep=True).sum()
    
    # Get column information
    columns_info = []
    for col in df.columns:
        if isinstance(df, dd.DataFrame):
            dtype = str(df[col].dtype)
            non_null_count = df.shape[0].compute() - df[col].isnull().sum().compute()
        else:
            dtype = str(df[col].dtype)
            non_null_count = df[col].count()
        
        columns_info.append({
            "name": col,
            "dtype": dtype,
            "non_null_count": int(non_null_count),
            "null_percentage": round(100 - (non_null_count / shape[0] * 100), 2) if shape[0] > 0 else 0
        })
    
    # Create a table visualization for column info
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Column Name', 'Data Type', 'Non-Null Count', 'Null %'],
                   fill_color='paleturquoise',
                   align='left'),
        cells=dict(values=[
            [col['name'] for col in columns_info],
            [col['dtype'] for col in columns_info],
            [col['non_null_count'] for col in columns_info],
            [col['null_percentage'] for col in columns_info]
        ],
        fill_color='lavender',
        align='left'))
    ])
    
    fig.update_layout(title="Dataset Structure")
    
    return {
        "plot": plotly_to_json(fig),
        "stats": {
            "shape": shape,
            "columns": len(df.columns),
            "memory_usage": memory_usage,
            "columns_info": columns_info
        }
    }

async def generate_distribution_analysis(df) -> dict:
    """Generate distribution visualizations for numerical and categorical columns"""
    # Separate numerical and categorical columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    distributions = {}
    histograms = {}
    
    # Process numerical columns
    if num_cols:
        # Select first numerical column for visualization
        col = num_cols[0]
        if isinstance(df, dd.DataFrame):
            sample_df = df.head(10000)
            values = sample_df[col].dropna().values
        else:
            sample_size = min(10000, len(df))
            sample_df = df.sample(n=sample_size) if sample_size > 0 else df
            values = sample_df[col].dropna().values
        
        if len(values) > 0:
            # Create histogram
            fig = px.histogram(
                sample_df, x=col,
                title=f"Distribution of {col}",
                marginal="box"  # Add a box plot on the margins
            )
            histograms[col] = plotly_to_json(fig)
            
            # Calculate distribution statistics
            distributions[col] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "skewness": float(stats.skew(values)) if len(values) > 2 else None,
                "kurtosis": float(stats.kurtosis(values)) if len(values) > 2 else None
            }
    
    # Process categorical columns
    cat_distributions = {}
    bar_charts = {}
    
    if cat_cols:
        # Select first categorical column for visualization
        col = cat_cols[0]
        if isinstance(df, dd.DataFrame):
            value_counts = df[col].value_counts().compute().head(15)
        else:
            value_counts = df[col].value_counts().head(15)
        
        if not value_counts.empty:
            # Create bar chart
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution of {col}",
                labels={"x": col, "y": "Count"}
            )
            bar_charts[col] = plotly_to_json(fig)
            
            # Calculate distribution statistics
            cat_distributions[col] = {
                "unique_values": len(value_counts),
                "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
                "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "least_common": value_counts.index[-1] if len(value_counts) > 0 else None,
                "least_common_count": int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0
            }
    
    return {
        "plots": {
            "histograms": histograms,
            "bar_charts": bar_charts
        },
        "stats": {
            "numerical_distributions": distributions,
            "categorical_distributions": cat_distributions,
            "numerical_columns": num_cols,
            "categorical_columns": cat_cols
        }
    }

async def generate_correlation_analysis(df) -> dict:
    """Generate correlation analysis for numerical columns"""
    # Get numerical columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(num_cols) < 2:
        return {"error": "Need at least 2 numerical columns for correlation analysis"}
    
    # Calculate correlation matrix
    if isinstance(df, dd.DataFrame):
        sample_df = df.head(10000)
        corr_matrix = sample_df[num_cols].corr().compute()
    else:
        sample_size = min(10000, len(df))
        sample_df = df.sample(n=sample_size) if sample_size > 0 else df
        corr_matrix = sample_df[num_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )
    
    # Find strongest correlations
    corr_values = corr_matrix.unstack()
    # Remove self-correlations
    corr_values = corr_values[corr_values < 1.0]
    strongest_corr = corr_values.abs().nlargest(5)
    
    strongest_pairs = []
    for idx, corr_value in strongest_corr.items():
        col1, col2 = idx
        strongest_pairs.append({
            "column1": col1,
            "column2": col2,
            "correlation": float(corr_value)
        })
    
    # Create scatter plot for the strongest correlation pair
    scatter_plot = None
    if strongest_pairs:
        col1 = strongest_pairs[0]["column1"]
        col2 = strongest_pairs[0]["column2"]
        scatter_fig = px.scatter(
            sample_df, x=col1, y=col2,
            title=f"Scatter Plot: {col1} vs {col2}",
            trendline="ols"  # Add trend line
        )
        scatter_plot = plotly_to_json(scatter_fig)
    
    return {
        "plots": {
            "heatmap": plotly_to_json(fig),
            "scatter": scatter_plot
        },
        "stats": {
            "strongest_correlations": strongest_pairs,
            "correlation_matrix": corr_matrix.to_dict()
        }
    }

async def generate_summary_statistics(df) -> dict:
    """Generate comprehensive summary statistics for the dataset"""
    try:
        # Basic statistics
        if isinstance(df, dd.DataFrame):
            sample_df = df.head(10000)
            num_rows = df.shape[0].compute()
            num_cols = df.shape[1]
            missing_values = df.isnull().sum().sum().compute()
            duplicate_rows = df.duplicated().sum().compute()
        else:
            sample_df = df
            num_rows = len(df)
            num_cols = len(df.columns)
            missing_values = df.isnull().sum().sum()
            duplicate_rows = df.duplicated().sum()
        
        # Get numerical and categorical columns
        num_cols = sample_df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = sample_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Summary statistics for numerical columns
        num_stats = {}
        if num_cols:
            try:
                desc = sample_df[num_cols].describe().to_dict()
                num_stats = {
                    col: {
                        stat: float(value) if not pd.isna(value) else None 
                        for stat, value in stats.items()
                    } for col, stats in desc.items()
                }
            except Exception as e:
                print(f"Error calculating numerical statistics: {str(e)}")
                # Continue with empty num_stats if there's an error
        
        # Summary statistics for categorical columns
        cat_stats = {}
        for col in cat_cols:
            try:
                value_counts = sample_df[col].value_counts()
                if not value_counts.empty:
                    top_value = value_counts.index[0] if len(value_counts) > 0 else None
                    # Convert top_value to string if it's not None to ensure JSON serialization
                    top_value_str = str(top_value) if top_value is not None else None
                    
                    cat_stats[col] = {
                        "unique_count": len(value_counts),
                        "top_value": top_value_str,
                        "top_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                        "null_count": int(sample_df[col].isnull().sum())
                    }
            except Exception as e:
                print(f"Error processing categorical column {col}: {str(e)}")
                # Skip this column if there's an error
        
        # Create summary table visualization
        summary_data = [
            ["Total Rows", num_rows],
            ["Total Columns", num_cols],
            ["Numerical Columns", len(num_cols)],
            ["Categorical Columns", len(cat_cols)],
            ["Missing Values", missing_values],
            ["Missing Values (%)", round(missing_values / (num_rows * len(df.columns)) * 100, 2) if num_rows * len(df.columns) > 0 else 0],
            ["Duplicate Rows", duplicate_rows],
            ["Duplicate Rows (%)", round(duplicate_rows / num_rows * 100, 2) if num_rows > 0 else 0]
        ]
        
        # Convert summary_data to a proper dictionary
        summary_dict = {item[0]: item[1] for item in summary_data}
        
        fig = go.Figure(data=[go.Table(
            header=dict(values=["Metric", "Value"],
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=[
                    [item[0] for item in summary_data],  # Metrics
                    [item[1] for item in summary_data]   # Values
                ],
                fill_color='lavender',
                align='left')
            )
        ])
        
        fig.update_layout(title="Dataset Summary Statistics")
        
        return {
            "plot": plotly_to_json(fig),
            "stats": {
                "dataset_stats": summary_dict,
                "numerical_stats": num_stats,
                "categorical_stats": cat_stats
            }
        }
    except Exception as e:
        # Return a meaningful error message
        error_message = f"Failed to generate summary statistics: {str(e)}"
        print(error_message)
        
        # Create a simple error visualization
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Error generating summary statistics:<br>{str(e)}",
            font=dict(color="red", size=14),
            showarrow=False,
            xref="paper", yref="paper"
        )
        fig.update_layout(
            title="Summary Statistics Error",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        
        return {
            "error": error_message,
            "plot": plotly_to_json(fig),
            "stats": {
                "dataset_stats": {},
                "numerical_stats": {},
                "categorical_stats": {}
            }
        }

async def generate_categorical_consistency(df) -> dict:
    """Check categorical variables for consistency issues"""
    # Get categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not cat_cols:
        return {"error": "No categorical columns found in the dataset"}
    
    consistency_issues = {}
    visualizations = {}
    
    # Define common domain-specific validation rules
    validation_rules = {
        # Common country names and their valid forms
        "country": {
            "valid_values": [
                "USA", "United States", "US", "Canada", "France", "Germany", "UK", "United Kingdom", 
                "Spain", "Italy", "Japan", "China", "India", "Brazil", "Australia", "Morocco"
            ],
            "case_sensitive": False,
            "invalid_patterns": ["Invalid", "Unknown", "Test", "N/A"]
        },
        # Common rating scales
        "rating": {
            "valid_range": (1, 10),  # Typical rating range
            "invalid_patterns": ["invalid", "none", "unknown"]
        },
        # Common categories (A-Z)
        "category": {
            "valid_pattern": r'^[A-Z]$',  # Single uppercase letter
            "case_sensitive": True
        }
    }
    
    for col in cat_cols:
        if isinstance(df, dd.DataFrame):
            sample_df = df.head(10000)
            values = sample_df[col].compute()
        else:
            values = df[col]
        
        # Skip empty columns
        if values.dropna().empty:
            continue
        
        # Convert to string for string operations, but keep original for numeric checks
        values_str = values.astype(str)
        
        # Basic consistency checks
        lowercase_values = values_str.str.lower()
        case_inconsistencies = (values_str != lowercase_values) & (~values_str.isna())
        
        stripped_values = values_str.str.strip()
        space_inconsistencies = (values_str != stripped_values) & (~values_str.isna())
        
        # Value frequency analysis
        value_counts = values.value_counts(dropna=False)
        total_count = len(values)
        rare_threshold = 0.01  # 1% of total
        rare_values = value_counts[value_counts / total_count < rare_threshold]
        
        # Advanced validation based on column name
        invalid_values = []
        col_lower = col.lower()
        
        # Apply domain-specific validation rules
        for rule_name, rule in validation_rules.items():
            if rule_name in col_lower or any(keyword in col_lower for keyword in [rule_name, f"{rule_name}s", f"{rule_name}_id"]):
                # Check against valid values list
                if "valid_values" in rule:
                    valid_values = rule["valid_values"]
                    case_sensitive = rule.get("case_sensitive", False)
                    
                    if not case_sensitive:
                        valid_values_lower = [v.lower() if isinstance(v, str) else v for v in valid_values]
                        invalid_mask = ~lowercase_values.isin(valid_values_lower) & (~values.isna())
                    else:
                        valid_values_set = set(valid_values)
                        invalid_mask = ~values_str.isin(valid_values) & (~values.isna())
                    
                    invalid_values.extend(values[invalid_mask].unique().tolist())
                
                # Check numeric range for ratings or scores
                if "valid_range" in rule:
                    min_val, max_val = rule["valid_range"]
                    try:
                        # Convert to numeric, coercing errors to NaN
                        numeric_values = pd.to_numeric(values, errors='coerce')
                        # Find values outside the valid range
                        range_mask = (numeric_values < min_val) | (numeric_values > max_val)
                        invalid_range_values = values[range_mask & (~numeric_values.isna())]
                        invalid_values.extend(invalid_range_values.unique().tolist())
                    except:
                        pass  # Skip if conversion fails
                
                # Check for invalid patterns
                if "invalid_patterns" in rule:
                    for pattern in rule["invalid_patterns"]:
                        pattern_lower = pattern.lower()
                        pattern_mask = lowercase_values.str.contains(pattern_lower, na=False)
                        invalid_values.extend(values[pattern_mask].unique().tolist())
        
        # Collect all issues
        issues = {
            "case_inconsistencies": int(case_inconsistencies.sum()),
            "space_inconsistencies": int(space_inconsistencies.sum()),
            "rare_values_count": len(rare_values),
            "rare_values": rare_values.index.tolist()[:10],  # Limit to top 10
            "invalid_values": invalid_values[:10]  # Limit to top 10
        }
        
        # Only add to issues if there are actual problems
        if any(v > 0 for k, v in issues.items() if k.endswith('_count') or k.endswith('_inconsistencies')) or \
           any(len(v) > 0 for k, v in issues.items() if k.endswith('_values')):
            consistency_issues[col] = issues
        
        # Create visualization for columns with issues
        if col in consistency_issues:
            # Create bar chart of value counts with highlighting for potential issues
            top_values = value_counts.head(15)
            colors = ['rgba(31, 119, 180, 0.8)'] * len(top_values)
            
            # Highlight potential issue values in red
            for i, (val, _) in enumerate(top_values.items()):
                # Check if this value has any issues
                has_issues = False
                
                if pd.notna(val):
                    val_str = str(val)
                    # Check for case inconsistencies
                    if val_str != val_str.lower() and val_str.lower() in lowercase_values.values:
                        has_issues = True
                    # Check for whitespace issues
                    elif val_str != val_str.strip():
                        has_issues = True
                    # Check if it's a rare value
                    elif val in rare_values.index:
                        has_issues = True
                    # Check if it's in the invalid values list
                    elif val in invalid_values:
                        has_issues = True
                
                if has_issues:
                    colors[i] = 'rgba(214, 39, 40, 0.8)'  # Red for issues
            
            fig = go.Figure(data=[go.Bar(
                x=[str(x) for x in top_values.index],  # Convert to string to handle all types
                y=top_values.values,
                marker_color=colors
            )])
            
            fig.update_layout(
                title=f"Value Distribution for {col} (Red = Potential Issues)",
                xaxis_title=col,
                yaxis_title="Count",
                xaxis_tickangle=-45
            )
            
            visualizations[col] = plotly_to_json(fig)
    
    # Create summary table of issues
    if consistency_issues:
        summary_data = []
        for col, issues in consistency_issues.items():
            summary_data.append({
                "Column": col,
                "Case Issues": issues["case_inconsistencies"],
                "Space Issues": issues["space_inconsistencies"],
                "Rare Values": issues["rare_values_count"],
                "Invalid Values": len(issues["invalid_values"]),
                "Examples": ", ".join([str(x) for x in issues["invalid_values"][:3]]) if issues["invalid_values"] else ""
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            fig_summary = go.Figure(data=[go.Table(
                header=dict(
                    values=list(summary_df.columns),
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[summary_df[col] for col in summary_df.columns],
                    fill_color='lavender',
                    align='left'
                )
            )])
            fig_summary.update_layout(title="Categorical Consistency Issues Summary")
            visualizations["summary"] = plotly_to_json(fig_summary)
    
    return {
        "plots": visualizations,
        "stats": {
            "columns_with_issues": list(consistency_issues.keys()),
            "consistency_issues": consistency_issues
        }
    }

def plotly_to_json(fig) -> dict:
    return json.loads(fig.to_json())