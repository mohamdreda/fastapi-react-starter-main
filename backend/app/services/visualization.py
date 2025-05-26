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
from app.services.data_quality import DataQualityAnalyzer, NULL_INDICATORS

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
    """Generate visualizations and statistics for duplicate rows"""
    if df.empty:
        return {"error": "Dataset is empty"}
    
    # Use the enhanced analyzer for context-aware duplicate detection with near-duplicate detection
    analyzer = DataQualityAnalyzer(df)
    
    # Run three different duplicate analyses with different settings
    standard_duplicates = analyzer.analyze_duplicates(
        case_sensitive=True, 
        ignore_whitespace=False, 
        exclude_id_columns=False,
        similarity_threshold=1.0  # Exact matches only
    )
    
    normalized_duplicates = analyzer.analyze_duplicates(
        case_sensitive=False, 
        ignore_whitespace=True, 
        exclude_id_columns=False,
        similarity_threshold=1.0  # Exact matches with normalization
    )
    
    smart_duplicates = analyzer.analyze_duplicates(
        case_sensitive=False, 
        ignore_whitespace=True, 
        exclude_id_columns=True,
        similarity_threshold=0.9  # Include near-duplicates
    )
    
    # Create comparison visualization
    comparison_data = {
        'Analysis Type': [
            'Standard (Exact Match)', 
            'Normalized (Case-Insensitive, Ignore Whitespace)',
            'Smart (Normalized + Exclude ID + Near-Duplicates)'
        ],
        'Duplicate Count': [
            standard_duplicates['duplicate_count'],
            normalized_duplicates['duplicate_count'],
            smart_duplicates['duplicate_count']
        ],
        'Duplicate Percentage': [
            standard_duplicates['duplicate_percentage'],
            normalized_duplicates['duplicate_percentage'],
            smart_duplicates['duplicate_percentage']
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create bar chart comparing different duplicate detection methods
    fig_comparison = px.bar(
        comparison_df,
        x='Analysis Type',
        y='Duplicate Count',
        title="Duplicate Detection Comparison",
        text='Duplicate Count'
    )
    fig_comparison.update_layout(xaxis_tickangle=-45)
    
    # Create a pie chart showing the proportion of duplicates
    # Calculate percentages for pie chart
    duplicate_percentage = smart_duplicates['duplicate_percentage']
    non_duplicate_percentage = 100 - duplicate_percentage
    
    # Round to ensure they sum to exactly 100%
    duplicate_percentage_rounded = round(duplicate_percentage, 1)
    non_duplicate_percentage_rounded = round(100 - duplicate_percentage_rounded, 1)
    
    # Create pie chart
    fig_pie = go.Figure()
    fig_pie.add_trace(go.Pie(
        labels=["Non-Duplicate", "Duplicate"],
        values=[non_duplicate_percentage_rounded, duplicate_percentage_rounded],
        textinfo="percent",
        insidetextorientation="radial",
        marker=dict(
            colors=["#FF9D5C", "#4C78A8"],  # Orange for non-duplicates, blue for duplicates
        ),
        hoverinfo="label+percent",
        textfont=dict(size=14),
    ))
    
    # Update layout
    fig_pie.update_layout(
        title={
            "text": "Duplicate Rows Distribution",
            "x": 0.5,
            "font": {"size": 18}
        },
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
        annotations=[
            dict(
                text=f"Overall Duplicate Rows: {duplicate_percentage_rounded}%",
                x=0.5, y=-0.15,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=14)
            )
        ]
    )
    
    # Create a heatmap showing where duplicates are located in the dataset
    # First, get a sample of the data with duplicate flags
    if isinstance(df, dd.DataFrame):
        sample_df = df.head(1000)
    else:
        sample_size = min(1000, len(df))
        sample_df = df.sample(n=sample_size) if sample_size > 0 else df
    
    # Create a duplicate flag array
    duplicate_flags = np.zeros(len(sample_df))
    
    # Mark rows that are duplicates
    # Check if duplicate_groups exists in the dictionary
    if 'duplicate_groups' in smart_duplicates and smart_duplicates['duplicate_groups']:
        try:
            for group in smart_duplicates['duplicate_groups']:
                if 'indices' in group:
                    for idx in group['indices']:
                        if isinstance(idx, (int, np.integer)) and idx < len(duplicate_flags):
                            duplicate_flags[idx] = 1
        except Exception as e:
            print(f"Error processing duplicate groups: {e}")
    elif 'examples' in smart_duplicates and smart_duplicates['examples']:
        # Fall back to examples if duplicate_groups is not available
        try:
            for example in smart_duplicates['examples']:
                if 'original_index' in example and isinstance(example['original_index'], (int, np.integer)):
                    idx = example['original_index']
                    if idx < len(duplicate_flags):
                        duplicate_flags[idx] = 1
                if 'duplicate_index' in example and isinstance(example['duplicate_index'], (int, np.integer)):
                    idx = example['duplicate_index']
                    if idx < len(duplicate_flags):
                        duplicate_flags[idx] = 1
        except Exception as e:
            print(f"Error processing duplicate examples: {e}")
    
    # Create a heatmap showing duplicate locations
    duplicate_df = pd.DataFrame({
        'Row Index': range(len(sample_df)),
        'Is Duplicate': duplicate_flags
    })
    
    fig_heatmap = px.imshow(
        duplicate_flags.reshape(1, -1),
        color_continuous_scale=["#4682B4", "#FF7F50"],
        labels=dict(x="Row Index", y="", color="Is Duplicate"),
        title="Duplicate Rows Location Heatmap (Sample)"
    )
    
    fig_heatmap.update_layout(
        height=200,
        width=900,
        yaxis_visible=False,
        coloraxis_showscale=True,
        coloraxis_colorbar=dict(title="Is Duplicate")
    )
    
    # Create a table of duplicate examples if available
    fig_examples = None
    examples_data = []
    
    # First try to use duplicate_groups if available
    if 'duplicate_groups' in smart_duplicates and smart_duplicates['duplicate_groups']:
        try:
            # Get the first 5 duplicate groups
            for i, group in enumerate(smart_duplicates['duplicate_groups'][:5]):
                if 'indices' in group:
                    for j, row_index in enumerate(group['indices']):
                        if not isinstance(row_index, (int, np.integer)):
                            continue
                            
                        # Get row data for display
                        row_data = {}
                        
                        # Add group identifier
                        row_data['Duplicate Group'] = f"Group {i+1}"
                        row_data['Row Index'] = row_index
                        
                        # Add a few key columns for context
                        for col in list(df.columns)[:5]:  # Show first 5 columns
                            try:
                                if isinstance(df, dd.DataFrame):
                                    # For Dask, we need to compute the value
                                    row_data[col] = str(df[col].compute().iloc[row_index])
                                else:
                                    row_data[col] = str(df[col].iloc[row_index])
                            except Exception as e:
                                row_data[col] = f"Error: {str(e)}"
                        
                        examples_data.append(row_data)
        except Exception as e:
            print(f"Error creating examples table from duplicate_groups: {e}")
    
    # Fall back to examples if duplicate_groups is not available or failed
    if not examples_data and 'examples' in smart_duplicates and smart_duplicates['examples']:
        try:
            for i, example in enumerate(smart_duplicates['examples'][:5]):
                # Process original row
                if 'original_index' in example and isinstance(example['original_index'], (int, np.integer)):
                    row_data = {}
                    row_data['Duplicate Group'] = f"Pair {i+1}"
                    row_data['Row Type'] = 'Original'
                    row_data['Row Index'] = example['original_index']
                    
                    # Add sample columns
                    for col in list(df.columns)[:5]:
                        try:
                            if isinstance(df, dd.DataFrame):
                                row_data[col] = str(df[col].compute().iloc[example['original_index']])
                            else:
                                row_data[col] = str(df[col].iloc[example['original_index']])
                        except Exception as e:
                            row_data[col] = f"Error: {str(e)}"
                    
                    examples_data.append(row_data)
                
                # Process duplicate row
                if 'duplicate_index' in example and isinstance(example['duplicate_index'], (int, np.integer)):
                    row_data = {}
                    row_data['Duplicate Group'] = f"Pair {i+1}"
                    row_data['Row Type'] = 'Duplicate'
                    row_data['Row Index'] = example['duplicate_index']
                    
                    # Add sample columns
                    for col in list(df.columns)[:5]:
                        try:
                            if isinstance(df, dd.DataFrame):
                                row_data[col] = str(df[col].compute().iloc[example['duplicate_index']])
                            else:
                                row_data[col] = str(df[col].iloc[example['duplicate_index']])
                        except Exception as e:
                            row_data[col] = f"Error: {str(e)}"
                    
                    examples_data.append(row_data)
        except Exception as e:
            print(f"Error creating examples table from examples: {e}")
        
        if examples_data:
            examples_df = pd.DataFrame(examples_data)
            fig_examples = go.Figure(
                data=[go.Table(
                    header=dict(
                        values=list(examples_df.columns),
                        fill_color='paleturquoise',
                        align='left'
                    ),
                    cells=dict(
                        values=[examples_df[col] for col in examples_df.columns],
                        fill_color='lavender',
                        align='left'
                    )
                )]
            )
            fig_examples.update_layout(
                title="Examples of Detected Duplicates"
            )
    
    # Create a table showing excluded ID columns if any
    fig_id_cols = None
    if smart_duplicates['id_columns_excluded']:
        id_cols_data = {
            'ID Column': smart_duplicates['id_columns_excluded'],
            'Reason': ['Detected as ID column based on name and content' for _ in smart_duplicates['id_columns_excluded']]
        }
        id_cols_df = pd.DataFrame(id_cols_data)
        
        fig_id_cols = go.Figure(data=[go.Table(
            header=dict(
                values=list(id_cols_df.columns),
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[id_cols_df[col] for col in id_cols_df.columns],
                fill_color='lavender',
                align='left'
            )
        )])
        fig_id_cols.update_layout(title="ID Columns Excluded from Duplicate Detection")
    
    # Return in a format expected by the frontend
    # The frontend expects a 'plot' property with 'data' and 'layout'
    return {
        "plot": plotly_to_json(fig_comparison),  # Main plot is the comparison bar chart
        "additional_plots": {
            "pie_chart": plotly_to_json(fig_pie),
            "heatmap": plotly_to_json(fig_heatmap),
            "examples": plotly_to_json(fig_examples) if fig_examples else None,
            "id_columns": plotly_to_json(fig_id_cols) if fig_id_cols else None
        },
        "stats": {
            "standard": {
                "duplicate_count": standard_duplicates['duplicate_count'],
                "duplicate_percentage": standard_duplicates['duplicate_percentage']
            },
            "normalized": {
                "duplicate_count": normalized_duplicates['duplicate_count'],
                "duplicate_percentage": normalized_duplicates['duplicate_percentage'],
                "ignore_whitespace": normalized_duplicates['ignore_whitespace']
            },
            "smart": {
                "duplicate_count": smart_duplicates['duplicate_count'],
                "duplicate_percentage": smart_duplicates['duplicate_percentage'],
                "id_columns_excluded": smart_duplicates['id_columns_excluded'],
                "columns_checked": smart_duplicates['columns_checked'],
                "near_duplicate_count": smart_duplicates.get('near_duplicate_count', 0),
                "similarity_threshold": smart_duplicates.get('similarity_threshold', 0.9)
            }
        },
        "total_rows": analyzer.total_rows,
        "duplicate_rows": smart_duplicates['duplicate_count'],
        "duplicate_percentage": smart_duplicates['duplicate_percentage'],
        "additional_duplicates_found": smart_duplicates['duplicate_count'] - standard_duplicates['duplicate_count']
    }


async def generate_categorical_analysis(df) -> dict:
    """Analyze and visualize categorical columns with enhanced validation"""
    if df.empty:
        return {"error": "Dataset is empty"}
    
    # Use the enhanced analyzer for comprehensive categorical analysis
    analyzer = DataQualityAnalyzer(df)
    categorical_analysis = analyzer.analyze_categorical_columns()
    categorical_cols = categorical_analysis['categorical_columns']
    
    if not categorical_cols:
        return {"error": "No categorical columns identified"}
    
    # Analyze each categorical column
    analysis = {}
    plot_data = {}
    issue_tables = {}
    
    for col in categorical_cols:
        col_analysis = categorical_analysis['analysis'][col]
        
        # Get value counts from the analysis
        top_values = col_analysis['top_values']
        
        # Store analysis with enhanced information
        analysis[col] = {
            "unique_values": col_analysis['unique_values'],
            "top_values": top_values,
            "missing_count": col_analysis['missing_count'],
            "has_case_inconsistencies": len(col_analysis['case_inconsistencies']) > 0,
            "has_standardization_issues": len(col_analysis['standardization_issues']) > 0,
            "has_invalid_values": len(col_analysis['invalid_values']) > 0,
            "has_quality_issues": col_analysis['has_quality_issues']
        }
        
        # Create pie chart for distribution
        values_for_chart = [item for item in top_values if item['count'] > 0][:6]  # Top 6 non-zero values
        
        # If there are more values, add an "Other" category
        total_shown = sum(item['count'] for item in values_for_chart)
        total_all = sum(item['count'] for item in top_values)
        
        if total_shown < total_all:
            other_count = total_all - total_shown
            values_for_chart.append({
                'value': 'Other',
                'count': other_count,
                'percentage': (other_count / total_all) * 100,
                'is_valid': True
            })
        
        # Create the pie chart
        fig = px.pie(
            names=[item['value'] for item in values_for_chart],
            values=[item['count'] for item in values_for_chart],
            title=f"Category Distribution: {col}",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        # Add a hover template that shows validity
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        
        plot_data[col] = plotly_to_json(fig)
        
        # Create tables for issues if any exist
        issues_data = []
        
        # Add case inconsistencies
        for case_issue in col_analysis['case_inconsistencies']:
            for variation in case_issue['variations']:
                issues_data.append({
                    'Issue Type': 'Case Inconsistency',
                    'Base Value': case_issue['base_value'],
                    'Variation': variation,
                    'Count': case_issue['counts'].get(variation, 0),
                    'Recommendation': f"Standardize to '{case_issue['base_value']}'"
                })
        
        # Add standardization issues
        for std_issue in col_analysis['standardization_issues']:
            for variation in std_issue['found_variations']:
                issues_data.append({
                    'Issue Type': 'Standardization',
                    'Base Value': '/'.join(std_issue['standard_options']),
                    'Variation': variation,
                    'Count': std_issue['counts'].get(variation, 0),
                    'Recommendation': f"Standardize to one of {std_issue['standard_options']}"
                })
        
        # Add invalid values
        for invalid in col_analysis['invalid_values']:
            issues_data.append({
                'Issue Type': 'Invalid Value',
                'Base Value': 'N/A',
                'Variation': invalid,
                'Count': sum(1 for item in top_values if item['value'] == invalid),
                'Recommendation': 'Replace with valid value'
            })
        
        # Create a table for the issues if any exist
        if issues_data:
            issues_df = pd.DataFrame(issues_data)
            fig_issues = go.Figure(data=[go.Table(
                header=dict(
                    values=list(issues_df.columns),
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[issues_df[col] for col in issues_df.columns],
                    fill_color='lavender',
                    align='left'
                )
            )])
            fig_issues.update_layout(title=f"Quality Issues in Column: {col}")
            issue_tables[col] = plotly_to_json(fig_issues)
    
    # Select the first column with issues as the main plot, or the first column if none have issues
    main_col = next((col for col in categorical_cols if analysis[col]['has_quality_issues']), categorical_cols[0])
    
    return {
        "plot": plot_data[main_col],  # Main plot
        "all_plots": plot_data,
        "issue_tables": issue_tables,
        "analysis": analysis
    }
async def generate_outlier_analysis(df) -> dict:
    """Detect and visualize outliers in numerical columns"""
    if df.empty:
        return {"error": "Dataset is empty"}
        

    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Select only numeric columns, excluding ID-like columns
    exclude_cols = [col for col in df_copy.columns if col.lower() in ['id', 'index', 'key', 'uuid']]
    num_cols = df_copy.select_dtypes(include=['number']).columns.tolist()
    num_cols = [col for col in num_cols if col not in exclude_cols]
    
    if not num_cols:
        return {"error": "No numerical columns found in the dataset"}
    
# Use the DataQualityAnalyzer to detect outliers
analyzer = DataQualityAnalyzer(df)

outliers_data = {}
plots = {}

# Make a copy to filter extreme outliers for better visualization
df_filtered = df_copy.copy()

for col in num_cols:
    # Calculate Q1, Q3 and IQR for the column
    q1 = df_copy[col].quantile(0.25)
    q3 = df_copy[col].quantile(0.75)
    iqr = q3 - q1
    
    # Define outlier boundaries with a conservative 3*IQR range
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    
    # Identify outliers outside the boundaries
    outliers = df_copy[(df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)]
    
    if not outliers.empty:
        # Compute percentage of outliers relative to dataset
        outlier_percentage = len(outliers) / len(df_copy) * 100
        
        # Save outlier info in dictionary
        outliers_data[col] = {
            'count': len(outliers),
            'percentage': round(outlier_percentage, 2),
            'min_value': float(df_copy[col].min()),
            'max_value': float(df_copy[col].max()),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'examples': outliers[col].head(5).tolist()
        }
        
        # Identify extreme outliers beyond 5*IQR for visualization adjustment
        extreme_outliers = df_copy[(df_copy[col] < q1 - 5 * iqr) | (df_copy[col] > q3 + 5 * iqr)]
        if not extreme_outliers.empty:
            # Replace extreme outliers values in filtered df with boundary values
            extreme_low_mask = df_filtered[col] < q1 - 5 * iqr
            extreme_high_mask = df_filtered[col] > q3 + 5 * iqr
            
            if extreme_low_mask.any():
                df_filtered.loc[extreme_low_mask, col] = q1 - 5 * iqr
            if extreme_high_mask.any():
                df_filtered.loc[extreme_high_mask, col] = q3 + 5 * iqr

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
    
# Convert numpy integer types to native Python int for JSON serialization
if isinstance(shape[0], np.integer):
    shape = (int(shape[0]), int(shape[1]))

if memory_usage is not None and isinstance(memory_usage, np.integer):
    memory_usage = int(memory_usage)

    return {
        "plot": plotly_to_json(fig),
        "stats": {
            "shape": shape,
"columns": int(len(df.columns)),

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
bar_charts = {}
pie_charts = {}

# Process numerical columns - create histograms for first 5 numerical columns
for col in num_cols[:5]:
    # Ici, on pourrait générer un histogramme ou autre type de graphique pour chaque col
    bar_charts[col] = create_histogram(df[col])  # Exemple de fonction à définir
    
# Ensuite, si tu veux sélectionner la première colonne numérique pour une analyse plus détaillée :
if num_cols:
    col = num_cols[0]
    # Code d’analyse plus détaillée ou visualisation pour cette colonne

        if isinstance(df, dd.DataFrame):
            sample_df = df.head(10000)
            values = sample_df[col].dropna().values
        else:
            sample_size = min(10000, len(df))
            sample_df = df.sample(n=sample_size) if sample_size > 0 else df
            values = sample_df[col].dropna().values
        
        if len(values) > 0:
// Create enhanced histogram with multiple visualization options
fig = px.histogram(
    sample_df, 
    x=col,
    title=`Distribution of ${col}`,
    marginal="box",               // Add a box plot on the margins
    histnorm="probability density", // Normalize to show probability density
    color_discrete_sequence=['#4682B4']
);

try {
    // Compute KDE using scipy.stats.gaussian_kde
    kde = stats.gaussian_kde(values);
    x_range = np.linspace(Math.min(...values), Math.max(...values), 1000);
    y_kde = kde(x_range);
    fig.add_trace(go.Scatter({
        x: x_range,
        y: y_kde,
        mode: 'lines',
        name: 'KDE',
        line: {color: '#FF7F50', width: 2}
    }));
} catch (e) {
    // Skip KDE if error (e.g. identical values)
}

const mean_val = np.mean(values);
const median_val = np.median(values);

fig.add_vline({
    x: mean_val,
    line_dash: "dash",
    line_color: "red",
    annotation_text: `Mean: ${mean_val.toFixed(2)}`,
    annotation_position: "top right"
});
fig.add_vline({
    x: median_val,
    line_dash: "dash",
    line_color: "green",
    annotation_text: `Median: ${median_val.toFixed(2)}`,
    annotation_position: "top left"
});

fig.update_layout({
    height: 500,
    width: 800,
    xaxis_title: col,
    yaxis_title: "Density",
    legend: {
        orientation: "h",
        yanchor: "bottom",
        y: 1.02,
        xanchor: "right",
        x: 1
    }
});

histograms[col] = plotly_to_json(fig);

            distributions[col] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
# Process numerical columns - extended statistics
num_distributions = {}
histograms = {}

for col in num_cols[:5]:  # Limit to first 5 columns for performance
    if isinstance(df, dd.DataFrame):
        sample_df = df.head(10000)
        values = sample_df[col].dropna().to_numpy()
    else:
        sample_size = min(10000, len(df))
        sample_df = df.sample(n=sample_size) if sample_size > 0 else df
        values = sample_df[col].dropna().to_numpy()
    
    if len(values) > 0:
        # Create enhanced histogram with KDE, mean, median lines (code as given before)
        # ... [histogram creation code here, as in previous snippet] ...
        
        # Calculate comprehensive distribution statistics
        num_distributions[col] = {
            "count": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std_dev": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "q1": float(np.percentile(values, 25)),
            "q3": float(np.percentile(values, 75)),
            "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "skewness": float(stats.skew(values)) if len(values) > 2 else None,
            "kurtosis": float(stats.kurtosis(values)) if len(values) > 2 else None,
            "normality": float(stats.normaltest(values)[0]) if len(values) > 8 else None
        }

# Process categorical columns - bar + pie charts + statistics
cat_distributions = {}
bar_charts = {}
pie_charts = {}

for col in cat_cols[:5]:  # Limit to first 5 columns for performance
    if isinstance(df, dd.DataFrame):
        sample_df = df.head(10000)
        value_counts = sample_df[col].value_counts().compute().head(20)
    else:
        sample_size = min(10000, len(df))
        sample_df = df.sample(n=sample_size) if sample_size > 0 else df
        value_counts = sample_df[col].value_counts().head(20)

    if not value_counts.empty:
        total = value_counts.sum()
        percentages = [(count / total * 100) for count in value_counts.values]

        # Bar chart with counts and percentages
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=value_counts.index,
            y=value_counts.values,
            name="Count",
            marker_color='#4682B4'
        ))
        fig_bar.add_trace(go.Scatter(
            x=value_counts.index,
            y=percentages,
            name="Percentage",
            yaxis="y2",
            mode="lines+markers",
            marker=dict(color='#FF7F50', size=8),
            line=dict(color='#FF7F50', width=2)
        ))
        fig_bar.update_layout(
            title=f"Distribution of {col}",
            xaxis_title=col,
            yaxis_title="Count",
            yaxis2=dict(
                title="Percentage (%)",
                overlaying="y",
                side="right",
                range=[0, max(percentages) * 1.1]
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
            width=800,
            xaxis_tickangle=-45
        )
        bar_charts[col] = plotly_to_json(fig_bar)

        # Pie chart for category proportions
        fig_pie = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f"Proportion of Categories in {col}",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(
            height=500,
            width=500,
            showlegend=False if len(value_counts) > 10 else True
        )
        pie_charts[col] = plotly_to_json(fig_pie)

        # Calculate comprehensive distribution statistics
        cat_distributions[col] = {
            "unique_values": int(len(value_counts)),
            "most_common": str(value_counts.index[0]) if len(value_counts) > 0 else None,
            "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            "most_common_percentage": float(percentages[0]) if len(percentages) > 0 else 0,
            "entropy": float(stats.entropy(value_counts.values)) if len(value_counts) > 1 else 0,
            "top_5_categories": [str(x) for x in value_counts.index[:5].tolist()],
            "top_5_counts": [int(x) for x in value_counts.values[:5].tolist()]
        }

    
    return {
        "plots": {
            "histograms": histograms,

            "bar_charts": bar_charts,
            "pie_charts": pie_charts

        },
        "stats": {
            "numerical_distributions": distributions,
            "categorical_distributions": cat_distributions,
            "numerical_columns": num_cols,
            "categorical_columns": cat_cols
        }
    }

async def generate_correlation_analysis(df) -> dict:

    """Generate correlation matrix for numerical columns with outlier handling"""
    if df.empty:
        return {"error": "Dataset is empty"}
    
    # Create a sample for visualization
    if isinstance(df, dd.DataFrame):
        sample_df = df.head(1000)
    else:
        # Handle case where dataset has fewer than 1000 rows
        sample_size = min(1000, len(df))
        sample_df = df.sample(n=sample_size) if sample_size > 0 else df
    
    # Select only numeric columns
    num_df = df.select_dtypes(include=['number'])
    
    if num_df.empty or num_df.shape[1] < 2:
        return {"error": "Not enough numerical columns for correlation analysis"}
    
    # Create a copy for outlier handling
    df_filtered = num_df.copy()
    
    # Handle outliers for better correlation calculation
    for col in df_filtered.columns:
        # Calculate IQR and outlier boundaries
        q1 = df_filtered[col].quantile(0.25)
        q3 = df_filtered[col].quantile(0.75)
        iqr = q3 - q1
        
        if iqr > 0:  # Avoid division by zero or very small IQR
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            # Replace outliers with boundary values for correlation calculation
            df_filtered[col] = df_filtered[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Calculate correlation matrix on filtered data
    if isinstance(df, dd.DataFrame):
        # For Dask DataFrames, compute first
        df_filtered_computed = df_filtered.compute()
        corr_matrix = df_filtered_computed.corr()
    else:
        corr_matrix = df_filtered.corr()
    
    # Also calculate correlation on original data for comparison
    if isinstance(df, dd.DataFrame):
        orig_corr_matrix = num_df.corr().compute()
    else:
        orig_corr_matrix = num_df.corr()
    
    # Convert to JSON-serializable format
    corr_data = {}
    for col1 in corr_matrix.columns:
        corr_data[col1] = {}
        for col2 in corr_matrix.columns:
            # Store both the filtered and original correlation values
            filtered_corr = float(corr_matrix.loc[col1, col2])
            orig_corr = float(orig_corr_matrix.loc[col1, col2])
            
            corr_data[col1][col2] = {
                'filtered': filtered_corr,
                'original': orig_corr,
                'difference': abs(filtered_corr - orig_corr)
            }
    
    # Create heatmap using the outlier-filtered correlation
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',  # Format to 2 decimal places
        color_continuous_scale='Viridis',
        title="Correlation Matrix (Outlier-Adjusted)"
    )
    
    # Add annotations to highlight significant differences
    annotations = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i != j:  # Skip diagonal
                filtered_corr = corr_data[col1][col2]['filtered']
                orig_corr = corr_data[col1][col2]['original']
                diff = abs(filtered_corr - orig_corr)
                
                # If there's a significant difference, add a marker
                if diff > 0.2:  # Threshold for significant difference
                    annotations.append(dict(
                        x=j,
                        y=i,
                        text="*",
                        showarrow=False,
                        font=dict(color="white", size=16)
                    ))
    
    fig.update_layout(
        height=600,
        width=700,
        margin=dict(l=60, r=50, t=50, b=50),
        annotations=annotations
    )
    
    # Create a comparison heatmap showing the difference
    diff_matrix = abs(corr_matrix - orig_corr_matrix)
    fig_diff = px.imshow(
        diff_matrix,
        text_auto='.2f',
        color_continuous_scale='Reds',
        title="Correlation Difference (Original vs. Outlier-Adjusted)"
    )
    
    fig_diff.update_layout(
        height=600,
        width=700,
        margin=dict(l=60, r=50, t=50, b=50)
    )
    
    # Find strongest correlations (using filtered data)

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
if isinstance(df, dd.DataFrame):
    num_rows = int(df.shape[0].compute())
    num_cols = int(df.shape[1])
    missing_values = int(df.isnull().sum().sum().compute())
    duplicate_rows = int(df.duplicated().sum().compute())
else:
    sample_df = df
    num_rows = int(len(df))
    num_cols = int(len(df.columns))
    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

        
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