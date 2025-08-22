"""
Utility functions for file operations.
"""
import os
import uuid
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from datetime import datetime


def ensure_directory_exists(directory: str) -> None:
    """Ensure that the specified directory exists, create it if it doesn't."""
    os.makedirs(directory, exist_ok=True)


def generate_unique_filename(original_filename: str) -> str:
    """
    Generate a unique filename with a timestamp and UUID.
    
    Args:
        original_filename: The original filename (used to preserve extension)
        
    Returns:
        A unique filename string
    """
    # Extract the file extension
    name_parts = os.path.splitext(original_filename)
    base_name = name_parts[0]
    extension = name_parts[1] if len(name_parts) > 1 else ''
    
    # Generate a unique filename with timestamp and UUID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
    
    return f"{base_name}_transformed_{timestamp}_{unique_id}{extension}"


def save_dataframe(
    df: pd.DataFrame, 
    output_dir: str, 
    filename: str,
    file_type: str = 'csv'
) -> Tuple[str, str]:
    """
    Save a pandas DataFrame to a file.
    
    Args:
        df: The DataFrame to save
        output_dir: Directory to save the file in
        filename: Name of the output file (without path)
        file_type: Type of file to save ('csv', 'parquet', 'feather', 'json')
        
    Returns:
        Tuple of (file_path, filename)
    """
    ensure_directory_exists(output_dir)
    
    # Ensure filename has the correct extension
    if not filename.lower().endswith(f'.{file_type}'):
        filename = f"{filename}.{file_type}"
    
    file_path = os.path.join(output_dir, filename)
    
    # Save based on file type
    if file_type == 'csv':
        df.to_csv(file_path, index=False)
    elif file_type == 'parquet':
        df.to_parquet(file_path, index=False)
    elif file_type == 'feather':
        df.to_feather(file_path)
    elif file_type == 'json':
        df.to_json(file_path, orient='records', lines=True)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    return file_path, filename


def get_file_extension(filename: str) -> str:
    """
    Get the file extension from a filename.
    
    Args:
        filename: The filename to get the extension from
        
    Returns:
        The file extension in lowercase (without the dot)
    """
    _, ext = os.path.splitext(filename)
    return ext.lstrip('.').lower()


def detect_file_type(filename: str) -> str:
    """
    Detect the file type based on the filename extension.
    
    Args:
        filename: The filename to check
        
    Returns:
        The detected file type ('csv', 'parquet', 'feather', 'json')
    """
    ext = get_file_extension(filename)
    
    if ext == 'csv':
        return 'csv'
    elif ext == 'parquet':
        return 'parquet'
    elif ext == 'feather':
        return 'feather'
    elif ext == 'json':
        return 'json'
    else:
        # Default to CSV
        return 'csv'


def validate_file_extension(filename: str, valid_extensions: list[str]) -> bool:
    """
    Validate that a filename has an allowed extension.
    
    Args:
        filename: The filename to check
        valid_extensions: List of valid extensions (with dot, e.g. ['.csv', '.xlsx'])
        
    Returns:
        True if the file extension is valid, False otherwise
    """
    if not filename:
        return False
        
    # Get the file extension (with dot)
    _, ext = os.path.splitext(filename.lower())
    
    # Check if the extension is in the list of valid extensions
    return ext in valid_extensions
