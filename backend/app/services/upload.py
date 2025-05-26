import pandas as pd
import json
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Dataset
from app.utils.logger import setup_logger
from typing import Dict, Any
from datetime import datetime

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
        
        # Calculate statistics
        dataset.missing_values = json.dumps(analyze_missing_values(df))
        dataset.duplicates = df.duplicated().sum()
        dataset.data_types = json.dumps(get_data_types(df))
        dataset.categorical_issues = json.dumps(find_categorical_issues(df))
        dataset.summary_stats = json.dumps(get_summary_stats(df))
        
        # Save to database
        db.add(dataset)
        await db.flush()
        
        logger.info(f"Successfully processed file: {filename}")
        return dataset.id
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
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
    """Analyze missing values in the dataset"""
    total_missing = df.isna().sum().sum()
    total_cells = df.size
    missing_percentage = (total_missing / total_cells) * 100 if total_cells > 0 else 0
    
    per_column = {
        col: {
            'count': int(missing),
            'percentage': float(missing / len(df) * 100)
        }
        for col, missing in df.isna().sum().items()
        if missing > 0
    }
    
    return {
        'total_missing': int(total_missing),
        'missing_percentage': float(missing_percentage),
        'per_column': per_column
    }

def get_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """Get data types for each column"""
    return {col: str(dtype) for col, dtype in df.dtypes.items()}

def find_categorical_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """Find potential issues in categorical columns"""
    issues = {}
    
    for col in df.select_dtypes(include=['object', 'category']):
        # Check for inconsistent capitalization or spacing
        values = df[col].dropna().astype(str)
        cleaned_values = values.str.strip().str.lower()
        if not (values == cleaned_values).all():
            issues[col] = {
                'inconsistent_format': list(set(values[values != cleaned_values]))
            }
            
        # Check for possible typos (very rare values)
        value_counts = df[col].value_counts()
        rare_values = value_counts[value_counts == 1].index.tolist()
        if rare_values:
            if col not in issues:
                issues[col] = {}
            issues[col]['rare_values'] = rare_values[:10]  # Limit to top 10
            
    return issues

def get_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary statistics for numerical columns"""
    numeric_stats = df.describe().to_dict()
    
    # Convert numpy types to Python native types for JSON serialization
    return {
        col: {k: float(v) for k, v in stats.items()}
        for col, stats in numeric_stats.items()
    }
