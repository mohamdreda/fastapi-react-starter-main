import pandas as pd
import json
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Dataset
from app.utils.logger import setup_logger
from typing import Dict, Any
from datetime import datetime
from app.services.data_quality import DataQualityAnalyzer

logger = setup_logger(__name__)

async def process_uploaded_file(db: AsyncSession, file_path: str, user_id: int, file_type: str) -> int:
    """Process an uploaded file and save its metadata to the database"""
    try:
        file_path = Path(file_path)
        filename = file_path.name
        
        # Initial dataset entry
        dataset = Dataset(
            filename=filename,
            file_type=file_type,
            file_path=str(file_path),
            format="auto-detect",
            user_id=user_id,
            updated_at=datetime.utcnow()
        )
        
        # Read the file based on its type
        df = read_dataframe(file_path, file_type)
        
        # Use the new DataQualityAnalyzer for comprehensive analysis
        analyzer = DataQualityAnalyzer(df)
        analysis_results = analyzer.get_full_analysis()
        
        # Save analysis results to the dataset
        dataset.missing_values = json.dumps(analysis_results['missing_values'])
        # Sum exact and near duplicates for total duplicate count
        total_duplicates = analysis_results['duplicates']['exact_duplicates'] + analysis_results['duplicates'].get('near_duplicates', 0)
        dataset.duplicates = total_duplicates
        dataset.data_types = json.dumps(analysis_results['data_types']['inferred_types'])
        dataset.categorical_issues = json.dumps(analysis_results['categorical_consistency'])
        dataset.summary_stats = json.dumps(get_summary_stats(df))
        
        # Save additional metadata
        dataset.analysis_metadata = json.dumps({
            'id_columns': analysis_results['id_columns'],
            'duplicate_details': {
                'exact_duplicates': analysis_results['duplicates']['exact_duplicates'],
                'near_duplicates': analysis_results['duplicates'].get('near_duplicates', 0),
                'duplicate_pairs': analysis_results['duplicates'].get('duplicate_pairs', []),
                'duplicate_percentage': analysis_results['duplicates'].get('duplicate_percentage', 0)
            },
            'type_issues': analysis_results['data_types']['type_issues']
        })
        
        # Save to database
        db.add(dataset)
        await db.flush()
        
        logger.info(f"Successfully processed file: {filename}")
        return dataset.id
        
    except Exception as e:
        logger.exception("Error processing file")
        raise

def read_dataframe(file_path: Path, file_type: str) -> pd.DataFrame:
    """Read a file into a pandas DataFrame based on its type"""
    try:
        if file_type == 'csv':
            return pd.read_csv(file_path)
        elif file_type in ['xlsx', 'xls']:
            return pd.read_excel(file_path)
        elif file_type == 'json':
            return pd.read_json(file_path)
        elif file_type == 'parquet':
            return pd.read_parquet(file_path)
        elif file_type == 'feather':
            return pd.read_feather(file_path)
        elif file_type == 'tsv':
            return pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        raise

def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Legacy function - use DataQualityAnalyzer instead"""
    analyzer = DataQualityAnalyzer(df)
    return analyzer.analyze_missing_values()

def get_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """Legacy function - use DataQualityAnalyzer instead"""
    analyzer = DataQualityAnalyzer(df)
    return analyzer.analyze_data_types()['inferred_types']

def find_categorical_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """Legacy function - use DataQualityAnalyzer instead"""
    analyzer = DataQualityAnalyzer(df)
    return analyzer.analyze_categorical_consistency()

def get_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary statistics for numerical columns"""
    # Get numeric columns only
    numeric_df = df.select_dtypes(include=['number'])
    
    # If there are no numeric columns, return an empty dict
    if numeric_df.empty or len(numeric_df.columns) == 0:
        logger.warning("No numeric columns found in dataset for summary statistics")
        return {}
    
    numeric_stats = numeric_df.describe().to_dict()
    
    # Convert numpy types to Python native types for JSON serialization
    return {
        col: {k: float(v) for k, v in stats.items()}
        for col, stats in numeric_stats.items()
    }