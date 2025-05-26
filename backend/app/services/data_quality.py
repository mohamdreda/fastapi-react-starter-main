import pandas as pd
import dask.dataframe as dd
import numpy as np
import re
from typing import Dict, List, Any, Tuple, Optional, Set, Union
import logging
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Comprehensive list of null indicators across data types
NULL_INDICATORS = [
    # Python/Pandas null types
    None, np.nan, pd.NA, 
    
    # Common string representations of null
    'nan', 'NaN', 'NA', 'N/A', 'n/a', 'na',
    'null', 'NULL', 'Null', 'none', 'NONE', 'None',
    '', ' ', '-', '--', '?', 
    'unknown', 'UNKNOWN', 'Unknown', 'UNK',
    'undefined', 'UNDEFINED', 'Undefined', 
    'missing', 'MISSING', 'Missing', 'MISS',
    
    # Excel/CSV null indicators
    '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
    '1.#IND', '1.#QNAN', '<NA>', 
    
    # Database null indicators
    'NULL', 'nil', 'NIL',
    
    # Additional indicators
    '(null)', '[null]', '{null}', '<null>',
    '(empty)', '[empty]', '{empty}', '<empty>',
    'not available', 'Not Available', 'NOT AVAILABLE',
    'not applicable', 'Not Applicable', 'NOT APPLICABLE',
    'to be determined', 'To Be Determined', 'TBD', 'tbd',
    'not specified', 'Not Specified', 'NOT SPECIFIED',
    'not provided', 'Not Provided', 'NOT PROVIDED',
    
    # SQL-specific nulls
    'NULL', 'null', 
    
    # Additional variations
    r'\N', '\n', '\r', '\t',
    'void', 'VOID', 'Void',
    'blank', 'BLANK', 'Blank',
    'empty', 'EMPTY', 'Empty',
    '0', '0.0', '0.00',  # Sometimes used as null indicators in certain contexts
    'false', 'FALSE', 'False',  # Sometimes used as null indicators
    'no value', 'NO VALUE', 'No Value',
    'no data', 'NO DATA', 'No Data'
]

# Regex patterns for identifying ID-like columns
ID_COLUMN_PATTERNS = [
    r'(?i)^id$', r'(?i).*_id$', r'(?i)^.*id$', r'(?i)^.*_key$',
    r'(?i)^uuid$', r'(?i)^guid$', r'(?i)^identifier$',
    r'(?i)^primary_?key$', r'(?i)^unique.*$'
]

class DataQualityAnalyzer:
    """Enhanced data quality analysis service with comprehensive detection capabilities"""
    
    def __init__(self, df: Union[pd.DataFrame, dd.DataFrame], sample_size: int = 10000):
        """Initialize with dataframe and optional sample size for large datasets"""
        self.df = df
        self.is_dask = isinstance(df, dd.DataFrame)
        self.sample_size = sample_size
        self.total_rows = self._get_row_count()
        self.id_columns = self._identify_id_columns()
        
    def _get_row_count(self) -> int:
        """Get row count with Dask/Pandas compatibility"""
        if self.is_dask:
            return self.df.shape[0].compute()
        return len(self.df)
    
    def _get_sample(self) -> pd.DataFrame:
        """Get a representative sample for large datasets"""
        if self.is_dask:
            sample_frac = min(self.sample_size / self.total_rows, 1.0) if self.total_rows > 0 else 1.0
            return self.df.sample(frac=sample_frac).compute()
        else:
            if self.total_rows > self.sample_size:
                return self.df.sample(n=self.sample_size)
            return self.df
    
    def _identify_id_columns(self) -> List[str]:
        """Identify columns that appear to be IDs based on name and content"""
        id_columns = []
        
        # First pass: check column names against patterns
        for col in self.df.columns:
            if any(re.match(pattern, col) for pattern in ID_COLUMN_PATTERNS):
                id_columns.append(col)
                continue
                
            # For non-obvious names, check content in a sample
            if not self.is_dask:  # Skip content check for Dask DataFrames
                sample = self.df[col].dropna().head(100)
                if len(sample) > 0:
                    # Check if values look like IDs (unique, incremental, or UUID-like)
                    uniqueness = len(sample.unique()) / len(sample) if len(sample) > 0 else 0
                    has_uuid_pattern = sample.astype(str).str.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$').any()
                    
                    if uniqueness > 0.9 or has_uuid_pattern:
                        id_columns.append(col)
        
        return id_columns
    
    def analyze_missing_values(self) -> Dict[str, Any]:
        """Detect missing values in the dataset using comprehensive null indicators"""
        df_sample = self._get_sample()
        
        # Create a copy for analysis
        df_copy = df_sample.copy()
        
        # Track which null indicators were found in the data
        found_indicators = set()
        
        # First pass: replace all null indicators with NaN for consistent detection
        for indicator in NULL_INDICATORS:
            if indicator is not None:  # Skip None as it's handled differently
                # Check if this indicator exists in the dataset
                indicator_found = False
                
                # For string indicators, check string columns
                if isinstance(indicator, str):
                    for col in df_copy.select_dtypes(include=['object']).columns:
                        # Use more precise detection with exact matching
                        if isinstance(indicator, str) and len(indicator) > 0:
                            # Case-insensitive exact match
                            mask = df_copy[col].astype(str).str.lower() == indicator.lower()
                            if mask.any():
                                indicator_found = True
                                found_indicators.add(str(indicator))
                                # Replace matched values with NaN
                                df_copy.loc[mask, col] = np.nan
                else:  # For non-string indicators like np.nan, pd.NA
                    if df_copy.isna().any().any():
                        indicator_found = True
                        found_indicators.add(str(indicator))
                
                # Also do a global replace for all columns
                df_copy = df_copy.replace(indicator, np.nan)
        
        # Additional check for empty strings and whitespace-only strings
        for col in df_copy.select_dtypes(include=['object']).columns:
            # Replace empty strings and whitespace-only strings with NaN
            mask = df_copy[col].astype(str).str.strip() == ''
            if mask.any():
                found_indicators.add('empty string')
                df_copy.loc[mask, col] = np.nan
        
        # Count missing values per column
        missing_counts = df_copy.isna().sum()
        missing_percentages = (missing_counts / len(df_copy) * 100).round(2)
        
        # Get total missing values
        total_missing = missing_counts.sum()
        total_cells = len(df_copy) * len(df_copy.columns)
        missing_percentage = (total_missing / total_cells * 100).round(2) if total_cells > 0 else 0.0
        
        # Detailed analysis per column
        per_column = {}
        for col in df_copy.columns:
            missing_count = missing_counts[col]
            
            # Always include column in results, even if no missing values
            # This helps frontend know all columns were checked
            examples = []
            
            if missing_count > 0:
                # Get examples of rows with missing values in this column
                missing_mask = df_copy[col].isna()
                missing_indices = df_copy[missing_mask].index.tolist()[:5]  # Get up to 5 examples
                
                for idx in missing_indices:
                    # Get the original value before replacement
                    orig_value = df_sample.loc[idx, col]
                    
                    # Determine which null indicator matched
                    detected_as = 'Missing'
                    if orig_value is None:
                        detected_as = 'None'
                    elif isinstance(orig_value, float) and np.isnan(orig_value):
                        detected_as = 'NaN'
                    elif isinstance(orig_value, str):
                        if orig_value.strip() == '':
                            detected_as = 'Empty string'
                        elif orig_value in NULL_INDICATORS:
                            detected_as = f'Null indicator: {orig_value}'
                    
                    examples.append({
                        'row_index': int(idx),
                        'original_value': str(orig_value) if orig_value is not None else 'None',
                        'detected_as': detected_as
                    })
            
            per_column[col] = {
                'missing_count': int(missing_count),
                'missing_percentage': float(missing_percentages[col]),
                'examples': examples
            }
        
        # For large datasets, extrapolate the missing values count
        if self.is_dask or len(df_sample) < self.total_rows:
            # Scale up the missing percentage to the full dataset
            estimated_total_missing = int((missing_percentage / 100) * (self.total_rows * len(df_copy.columns)))
            
            # Update per-column estimates
            for col in per_column:
                per_column[col]['estimated_missing_count'] = int((per_column[col]['missing_percentage'] / 100) * self.total_rows)
        else:
            estimated_total_missing = total_missing
        
        return {
            'total_missing': int(total_missing),
            'estimated_total_missing': int(estimated_total_missing),
            'missing_percentage': float(missing_percentage),
            'per_column': per_column,
            'null_indicators_used': list(found_indicators),
            'null_indicators_checked': [str(ind) for ind in NULL_INDICATORS if ind is not None]
        }
    
    def _get_missing_examples(self, df: pd.DataFrame, column: str, max_examples: int = 5) -> List[str]:
        """Get examples of values that were detected as missing but not originally NaN"""
        # Find values that match our null indicators but aren't NaN
        examples = []
        for indicator in NULL_INDICATORS:
            if isinstance(indicator, (str, int, float)):
                matches = df[~df[column].isna() & (df[column] == indicator)][column].unique()
                examples.extend([str(m) for m in matches])
                if len(examples) >= max_examples:
                    break
        
        return examples[:max_examples]
    
    def analyze_duplicates(self, case_sensitive: bool = False, 
                          ignore_whitespace: bool = True,
                          exclude_id_columns: bool = True,
                          similarity_threshold: float = 0.9) -> Dict[str, Any]:
        """Context-aware duplicate detection with configurable parameters
        
        Parameters:
        - case_sensitive: Whether to consider case when comparing string values
        - ignore_whitespace: Whether to ignore whitespace in string comparisons
        - exclude_id_columns: Whether to exclude ID-like columns from comparison
        - similarity_threshold: Threshold for fuzzy matching (0.0-1.0)
        """
        df_sample = self._get_sample()
        
        # Determine columns to check
        columns_to_check = list(df_sample.columns)
        id_columns_excluded = []
        
        if exclude_id_columns:
            # Use the identified ID columns and also exclude any columns with ID-like patterns in the name
            id_columns_excluded = self.id_columns.copy()
            
            # More comprehensive ID column detection
            id_patterns = ['id', 'key', 'uuid', 'guid', 'index', 'code', 'num', 'no', 'number']
            additional_id_cols = []
            
            for col in columns_to_check:
                # Check if column name contains any ID patterns
                if any(pattern in col.lower() for pattern in id_patterns):
                    additional_id_cols.append(col)
                # Check if column has mostly unique values (>95% unique)
                elif len(df_sample) > 10:  # Only check if we have enough rows
                    uniqueness = df_sample[col].nunique() / len(df_sample)
                    if uniqueness > 0.95:
                        additional_id_cols.append(col)
            
            # Add the additional ID columns that weren't already in id_columns_excluded
            id_columns_excluded.extend([col for col in additional_id_cols if col not in id_columns_excluded])
            
            # Remove ID columns from the columns to check
            columns_to_check = [col for col in columns_to_check if col not in id_columns_excluded]
        
        # Create a copy for normalization
        df_normalized = df_sample[columns_to_check].copy()
        
        # Apply normalization based on parameters
        for col in df_normalized.columns:
            if df_normalized[col].dtype == 'object':
                # Convert to string to handle non-string objects
                df_normalized[col] = df_normalized[col].astype(str)
                
                # Replace null indicators with a consistent value
                for indicator in ["nan", "null", "none", "na", "n/a", ""]:
                    df_normalized[col] = df_normalized[col].str.replace(f"^{indicator}$", "NULL", case=False, regex=True)
                
                # Apply case normalization if specified (default is case-insensitive)
                if not case_sensitive:
                    df_normalized[col] = df_normalized[col].str.lower()
                
                # Remove whitespace if specified (default is true)
                if ignore_whitespace:
                    # Strip leading/trailing whitespace
                    df_normalized[col] = df_normalized[col].str.strip()
                    # Replace multiple spaces with a single space
                    df_normalized[col] = df_normalized[col].str.replace(r'\s+', ' ', regex=True)
                    # Remove special characters that might cause false negatives
                    df_normalized[col] = df_normalized[col].str.replace(r'[,\.\'"\-_\(\)\[\]\{\}]', '', regex=True)
        
        # Find exact duplicates
        duplicates = df_normalized.duplicated(keep='first')
        duplicate_count = duplicates.sum()
        duplicate_indices = duplicates[duplicates].index.tolist()
        
        # Find near-duplicates using fuzzy matching for small to medium datasets
        near_duplicate_pairs = []
        if len(df_normalized) <= 10000 and similarity_threshold < 1.0:  # Only for reasonably sized datasets
            try:
                from rapidfuzz import fuzz
                
                # Convert dataframe to list of tuples for faster comparison
                records = df_normalized.to_dict('records')
                record_strings = [str(r) for r in records]
                
                # Compare each record with others (excluding already identified exact duplicates)
                non_duplicate_indices = [i for i in range(len(records)) if i not in duplicate_indices]
                
                for i in range(len(non_duplicate_indices)):
                    idx1 = non_duplicate_indices[i]
                    for j in range(i+1, len(non_duplicate_indices)):
                        idx2 = non_duplicate_indices[j]
                        
                        # Calculate similarity ratio
                        similarity = fuzz.ratio(record_strings[idx1], record_strings[idx2]) / 100.0
                        
                        if similarity >= similarity_threshold:
                            near_duplicate_pairs.append((idx1, idx2, similarity))
                            duplicate_count += 1  # Count near-duplicates in the total
            except ImportError:
                # If rapidfuzz is not available, skip near-duplicate detection
                pass
        
        duplicate_percentage = (duplicate_count / len(df_normalized) * 100) if len(df_normalized) > 0 else 0
        
        # Get examples of duplicates for explanation
        examples = []
        
        # First add exact duplicates
        if duplicate_indices:
            for dup_idx in duplicate_indices[:min(5, len(duplicate_indices))]:
                # Find the original row this is a duplicate of
                dup_values = df_normalized.loc[dup_idx].to_dict()
                
                # Find all rows with these values
                matches = df_normalized.loc[(df_normalized == dup_values).all(axis=1)].index.tolist()
                original_idx = min(matches)  # Assume the first occurrence is the original
                
                if original_idx != dup_idx:  # Ensure we're not comparing a row to itself
                    examples.append({
                        'original_index': int(original_idx),
                        'duplicate_index': int(dup_idx),
                        'original_row': df_sample.loc[original_idx].to_dict(),
                        'duplicate_row': df_sample.loc[dup_idx].to_dict(),
                        'type': 'exact',
                        'similarity': 1.0
                    })
        
        # Then add near-duplicates
        for idx1, idx2, similarity in near_duplicate_pairs[:min(5, len(near_duplicate_pairs))]:
            examples.append({
                'original_index': int(idx1),
                'duplicate_index': int(idx2),
                'original_row': df_sample.loc[idx1].to_dict(),
                'duplicate_row': df_sample.loc[idx2].to_dict(),
                'type': 'near',
                'similarity': float(similarity)
            })
        
        # Extrapolate to full dataset
        if self.is_dask:
            # For Dask, we can't easily compute duplicates on the full dataset
            # So we extrapolate based on the sample
            estimated_duplicate_count = int(duplicate_percentage / 100 * self.total_rows)
            duplicate_count = estimated_duplicate_count
        else:
            # For Pandas, we can compute the exact count if the dataset isn't too large
            if self.total_rows <= 100000:  # Only compute for reasonably sized datasets
                # Apply the same normalization to the full dataset
                full_df_normalized = self.df[columns_to_check].copy()
                
                for col in full_df_normalized.columns:
                    if full_df_normalized[col].dtype == 'object':
                        full_df_normalized[col] = full_df_normalized[col].astype(str)
                        
                        if not case_sensitive:
                            full_df_normalized[col] = full_df_normalized[col].str.lower()
                        
                        if ignore_whitespace:
                            full_df_normalized[col] = full_df_normalized[col].str.strip()
                            full_df_normalized[col] = full_df_normalized[col].str.replace(r'\s+', ' ', regex=True)
                
                exact_duplicate_count = full_df_normalized.duplicated(keep='first').sum()
                
                # Adjust for near-duplicates in the full dataset
                if near_duplicate_pairs:
                    near_duplicate_ratio = len(near_duplicate_pairs) / len(df_normalized)
                    estimated_near_duplicates = int(near_duplicate_ratio * self.total_rows)
                    duplicate_count = int(exact_duplicate_count) + estimated_near_duplicates
                else:
                    duplicate_count = int(exact_duplicate_count)
                
                duplicate_percentage = (duplicate_count / self.total_rows * 100) if self.total_rows > 0 else 0
            else:
                # For large datasets, still use the sample estimate
                estimated_duplicate_count = int(duplicate_percentage / 100 * self.total_rows)
                duplicate_count = estimated_duplicate_count
        
        return {
            'duplicate_count': int(duplicate_count),
            'duplicate_percentage': float(duplicate_percentage),
            'columns_checked': columns_to_check,
            'id_columns_excluded': id_columns_excluded,
            'case_sensitive': case_sensitive,
            'ignore_whitespace': ignore_whitespace,
            'similarity_threshold': similarity_threshold,
            'near_duplicate_count': len(near_duplicate_pairs),
            'examples': examples
        }
        
    def _detect_case_inconsistencies(self, series: pd.Series) -> List[Dict[str, Any]]:
        """Detect case inconsistencies in string values"""
        if series.dtype != 'object':
            return []
            
        # Convert to strings and filter out missing values
        values = series.dropna().astype(str)
        if len(values) == 0:
            return []
            
        # Group values that are the same when lowercased
        case_groups = {}
        for val in values.unique():
            lower_val = val.lower()
            if lower_val in case_groups:
                case_groups[lower_val].append(val)
            else:
                case_groups[lower_val] = [val]
                
        # Find groups with multiple case variations
        inconsistencies = []
        for lower_val, variations in case_groups.items():
            if len(variations) > 1:
                counts = {var: int(series[series == var].count()) for var in variations}
                inconsistencies.append({
                    'base_value': lower_val,
                    'variations': variations,
                    'counts': counts,
                    'total_occurrences': sum(counts.values())
                })
                
        return inconsistencies
        
    def _detect_standardization_issues(self, series: pd.Series) -> List[Dict[str, Any]]:
        """Detect values that might be different representations of the same entity"""
        if series.dtype != 'object':
            return []
            
        # Convert to strings and filter out missing values
        values = series.dropna().astype(str)
        if len(values) == 0:
            return []
            
        # Common standardization patterns
        patterns = [
            # Country codes and names
            {'us', 'usa', 'united states', 'united states of america'},
            {'uk', 'united kingdom', 'great britain'},
            {'ae', 'uae', 'united arab emirates'},
            # Yes/No variations
            {'y', 'yes', 'true', 't', '1'},
            {'n', 'no', 'false', 'f', '0'},
            # Common abbreviations
            {'st', 'street'},
            {'rd', 'road'},
            {'ave', 'avenue'},
            {'apt', 'apartment'},
            # Titles
            {'mr', 'mister'},
            {'mrs', 'missus'},
            {'dr', 'doctor'},
            {'prof', 'professor'}
        ]
        
        # Check for matches against patterns
        standardization_issues = []
        unique_values = values.unique()
        
        for pattern_set in patterns:
            matches = []
            for val in unique_values:
                if val.lower() in pattern_set or any(p in val.lower() for p in pattern_set):
                    matches.append(val)
                    
            if len(matches) > 1:
                counts = {match: int(series[series == match].count()) for match in matches}
                standardization_issues.append({
                    'standard_options': list(pattern_set),
                    'found_variations': matches,
                    'counts': counts,
                    'total_occurrences': sum(counts.values())
                })
                
        return standardization_issues
        
    def _detect_invalid_domain_values(self, column_name: str, series: pd.Series) -> List[str]:
        """Detect values that are invalid for a specific domain based on column name"""
        if series.dtype != 'object':
            return []
            
        # Convert to strings and filter out missing values
        values = series.dropna().astype(str).unique()
        if len(values) == 0:
            return []
            
        invalid_values = []
        column_lower = column_name.lower()
        
        # Country validation
        if any(term in column_lower for term in ['country', 'nation', 'location', 'region', 'territory']):
            # Define country code mappings for standardization
            country_code_mappings = {
                # USA variations
                'us': 'united states', 'usa': 'united states', 'u.s.': 'united states', 'u.s.a.': 'united states',
                'united states of america': 'united states', 'america': 'united states', 'the united states': 'united states',
                'united states': 'united states',
                
                # UK variations
                'uk': 'united kingdom', 'u.k.': 'united kingdom', 'great britain': 'united kingdom',
                'britain': 'united kingdom', 'england': 'united kingdom', 'scotland': 'united kingdom',
                'wales': 'united kingdom', 'northern ireland': 'united kingdom',
                
                # Common abbreviations
                'uae': 'united arab emirates', 'u.a.e.': 'united arab emirates',
                'aus': 'australia', 'ca': 'canada', 'can': 'canada',
                'de': 'germany', 'ger': 'germany', 'fr': 'france', 'fra': 'france',
                'jp': 'japan', 'jpn': 'japan', 'cn': 'china', 'chn': 'china',
                'ru': 'russia', 'rus': 'russia', 'br': 'brazil', 'bra': 'brazil',
                'in': 'india', 'ind': 'india', 'mx': 'mexico', 'mex': 'mexico',
                'it': 'italy', 'ita': 'italy', 'es': 'spain', 'esp': 'spain',
                'kr': 'south korea', 'kor': 'south korea', 'za': 'south africa'
            }
            
            # ISO 3166-1 country names (comprehensive list)
            valid_countries = set([
                'afghanistan', 'albania', 'algeria', 'andorra', 'angola', 'antigua and barbuda',
                'argentina', 'armenia', 'australia', 'austria', 'azerbaijan', 'bahamas', 'bahrain',
                'bangladesh', 'barbados', 'belarus', 'belgium', 'belize', 'benin', 'bhutan',
                'bolivia', 'bosnia and herzegovina', 'botswana', 'brazil', 'brunei', 'bulgaria',
                'burkina faso', 'burundi', 'cabo verde', 'cambodia', 'cameroon', 'canada',
                'central african republic', 'chad', 'chile', 'china', 'colombia', 'comoros',
                'congo', 'costa rica', 'croatia', 'cuba', 'cyprus', 'czechia', 'czech republic',
                'denmark', 'djibouti', 'dominica', 'dominican republic', 'ecuador', 'egypt',
                'el salvador', 'equatorial guinea', 'eritrea', 'estonia', 'eswatini', 'ethiopia',
                'fiji', 'finland', 'france', 'gabon', 'gambia', 'georgia', 'germany', 'ghana',
                'greece', 'grenada', 'guatemala', 'guinea', 'guinea-bissau', 'guyana', 'haiti',
                'honduras', 'hungary', 'iceland', 'india', 'indonesia', 'iran', 'iraq', 'ireland',
                'israel', 'italy', 'jamaica', 'japan', 'jordan', 'kazakhstan', 'kenya', 'kiribati',
                'korea', 'north korea', 'south korea', 'kosovo', 'kuwait', 'kyrgyzstan', 'laos',
                'latvia', 'lebanon', 'lesotho', 'liberia', 'libya', 'liechtenstein', 'lithuania',
                'luxembourg', 'madagascar', 'malawi', 'malaysia', 'maldives', 'mali', 'malta',
                'marshall islands', 'mauritania', 'mauritius', 'mexico', 'micronesia', 'moldova',
                'monaco', 'mongolia', 'montenegro', 'morocco', 'mozambique', 'myanmar', 'namibia',
                'nauru', 'nepal', 'netherlands', 'new zealand', 'nicaragua', 'niger', 'nigeria',
                'north macedonia', 'norway', 'oman', 'pakistan', 'palau', 'palestine', 'panama',
                'papua new guinea', 'paraguay', 'peru', 'philippines', 'poland', 'portugal',
                'qatar', 'romania', 'russia', 'rwanda', 'saint kitts and nevis', 'saint lucia',
                'saint vincent and the grenadines', 'samoa', 'san marino', 'sao tome and principe',
                'saudi arabia', 'senegal', 'serbia', 'seychelles', 'sierra leone', 'singapore',
                'slovakia', 'slovenia', 'solomon islands', 'somalia', 'south africa', 'south sudan',
                'spain', 'sri lanka', 'sudan', 'suriname', 'sweden', 'switzerland', 'syria',
                'taiwan', 'tajikistan', 'tanzania', 'thailand', 'timor-leste', 'togo', 'tonga',
                'trinidad and tobago', 'tunisia', 'turkey', 'turkmenistan', 'tuvalu', 'uganda',
                'ukraine', 'united arab emirates', 'united kingdom', 'united states', 'uruguay',
                'uzbekistan', 'vanuatu', 'vatican city', 'venezuela', 'vietnam', 'yemen', 'zambia',
                'zimbabwe'
            ])
            
            # Add all country codes and mappings to valid countries
            valid_countries.update(country_code_mappings.keys())
            
            # Create a standardization mapping for the values
            standardized_values = {}
            obviously_invalid = []
            
            for val in values:
                val_lower = val.lower().strip()
                
                # Check if it's a known country code or name
                if val_lower in country_code_mappings:
                    standardized_values[val] = country_code_mappings[val_lower]
                elif val_lower in valid_countries:
                    standardized_values[val] = val_lower
                else:
                    # Check for obviously invalid values (non-geographic entities)
                    invalid_entities = ['moon', 'mars', 'jupiter', 'saturn', 'space', 'universe', 
                                       'international', 'global', 'world', 'internet', 'online',
                                       'test', 'invalid', 'unknown', 'none', 'other', 'n/a']
                    
                    if val_lower in invalid_entities or len(val_lower) < 2:
                        obviously_invalid.append(val)
                    else:
                        # Check for partial matches (e.g., "New York" instead of a country)
                        is_valid = False
                        for country in valid_countries:
                            # Check if country is a substring of val or val is a substring of country
                            if country in val_lower or val_lower in country:
                                similarity_ratio = len(country) / len(val_lower) if len(val_lower) > 0 else 0
                                if 0.5 <= similarity_ratio <= 2.0:  # Within reasonable similarity range
                                    standardized_values[val] = country
                                    is_valid = True
                                    break
                        
                        if not is_valid:
                            invalid_values.append(val)
        
        # Email validation
        elif any(term in column_lower for term in ['email', 'e-mail', 'mail']):
            email_pattern = r'^[\w\.-]+@([\w\-]+\.)+[A-Za-z]{2,}$'
            for val in values:
                if not re.match(email_pattern, val):
                    invalid_values.append(val)
        
        # Phone validation
        elif any(term in column_lower for term in ['phone', 'telephone', 'mobile', 'cell']):
            # Basic phone pattern - should contain mostly digits
            for val in values:
                digit_count = sum(c.isdigit() for c in val)
                if digit_count < 7 or digit_count / len(val) < 0.7:
                    invalid_values.append(val)
        
        # Gender validation
        elif any(term in column_lower for term in ['gender', 'sex']):
            valid_genders = {'male', 'female', 'm', 'f', 'other', 'non-binary', 'prefer not to say'}
            for val in values:
                if val.lower() not in valid_genders:
                    invalid_values.append(val)
        
        # Rating validation
        elif any(term in column_lower for term in ['rating', 'score', 'rank']):
            # Check if values can be converted to numbers
            for val in values:
                try:
                    num_val = float(val)
                    # Check if the value is in a reasonable range (0-10 or 0-100)
                    if not (0 <= num_val <= 100):
                        invalid_values.append(val)
                except ValueError:
                    # Non-numeric rating
                    invalid_values.append(val)
        
        return invalid_values
    
    def analyze_data_types(self) -> Dict[str, Any]:
        """Enhanced data type analysis with mixed type detection and validation"""
        df_sample = self._get_sample()
        
        type_issues = {}
        inferred_types = {}
        invalid_values = {}
        
        for col in df_sample.columns:
            current_type = str(df_sample[col].dtype)
            col_lower = col.lower()
            
            # Check for mixed types in object columns
            if current_type == 'object':
                # Try to infer better type
                numeric_success = False
                datetime_success = False
                
                # Check if column could be numeric
                try:
                    numeric_col = pd.to_numeric(df_sample[col], errors='coerce')
                    # If most values converted successfully, it's likely numeric
                    non_null_count = numeric_col.notna().sum()
                    original_non_null = df_sample[col].notna().sum()
                    
                    if original_non_null > 0 and non_null_count / original_non_null > 0.8:
                        numeric_success = True
                        inferred_types[col] = 'numeric'
                        
                        # Check for mixed numeric types (int/float)
                        if numeric_col.dropna().apply(lambda x: x.is_integer()).all():
                            inferred_types[col] = 'integer'
                except Exception:
                    pass
                
                # Special handling for date-related columns
                is_date_column = any(date_term in col_lower for date_term in 
                                    ['date', 'day', 'month', 'year', 'time', 'dob', 'birth'])
                
                # Check if column could be datetime
                if not numeric_success or is_date_column:
                    try:
                        # First try standard datetime parsing
                        datetime_col = pd.to_datetime(df_sample[col], errors='coerce')
                        non_null_count = datetime_col.notna().sum()
                        original_non_null = df_sample[col].notna().sum()
                        
                        # If we have date-like column name but poor conversion, try common formats
                        if is_date_column and (original_non_null == 0 or non_null_count / original_non_null < 0.8):
                            # Try common date formats
                            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', 
                                        '%d-%m-%Y', '%m-%d-%Y', '%d.%m.%Y', '%Y.%m.%d']:
                                try:
                                    datetime_col = pd.to_datetime(df_sample[col], format=fmt, errors='coerce')
                                    new_non_null = datetime_col.notna().sum()
                                    if new_non_null > non_null_count:
                                        non_null_count = new_non_null
                                except Exception:
                                    continue
                        
                        # Check for invalid dates
                        if original_non_null > 0 and non_null_count / original_non_null > 0.5:
                            datetime_success = True
                            inferred_types[col] = 'datetime'
                            
                            # Find invalid dates
                            if non_null_count < original_non_null:
                                invalid_mask = df_sample[col].notna() & datetime_col.isna()
                                if invalid_mask.any():
                                    invalid_dates = df_sample.loc[invalid_mask, col].unique().tolist()
                                    invalid_values[col] = {
                                        'type': 'invalid_dates',
                                        'values': [str(val) for val in invalid_dates[:10]],  # Show up to 10 examples
                                        'count': int(invalid_mask.sum())
                                    }
                    except Exception as e:
                        # If there's an error in datetime parsing, it might indicate invalid formats
                        pass
                
                # If we found a better type, record the issue
                if numeric_success or datetime_success:
                    # Get examples of values
                    sample_values = df_sample[col].dropna().sample(min(5, len(df_sample[col].dropna()))).tolist()
                    
                    type_issues[col] = {
                        'current_type': current_type,
                        'inferred_type': inferred_types.get(col, 'unknown'),
                        'example_values': [str(val) for val in sample_values]
                    }
            
            # Check for integer-like floats (e.g., 1.0, 2.0)
            elif current_type.startswith('float'):
                # Check if all non-NaN values are effectively integers
                non_null_values = df_sample[col].dropna()
                if len(non_null_values) > 0:
                    integer_like = all(val.is_integer() for val in non_null_values)
                    if integer_like:
                        inferred_types[col] = 'integer'
                        type_issues[col] = {
                            'current_type': current_type,
                            'inferred_type': 'integer',
                            'issue': 'Integer values stored as floats'
                        }
        
        # Check for mixed types in numeric columns (e.g., integers stored as floats)
        for col in df_sample.select_dtypes(include=['float']).columns:
            # Check if all non-NaN values are effectively integers
            non_null_values = df_sample[col].dropna()
            if len(non_null_values) > 0:
                try:
                    integer_like = all(float(val).is_integer() for val in non_null_values)
                    if integer_like:
                        inferred_types[col] = 'integer'
                        type_issues[col] = {
                            'current_type': current_type,
                            'inferred_type': 'integer',
                            'issue': 'Integer values stored as floats'
                        }
                except Exception:
                    pass
        
        return {
            'column_types': {col: str(df_sample[col].dtype) for col in df_sample.columns},
            'inferred_types': inferred_types,
            'type_issues': type_issues,
            'invalid_values': invalid_values
        }
    
    def analyze_categorical_columns(self) -> Dict[str, Any]:
        """Identify and analyze categorical columns with enhanced validation"""
        df_sample = self._get_sample()
        
        # Identify potential categorical columns
        categorical_cols = []
        for col in df_sample.columns:
            # Skip ID columns
            if col in self.id_columns:
                continue
                
            # Check if column is already categorical
            if df_sample[col].dtype.name == 'category':
                categorical_cols.append(col)
                continue
                
            # Check if object type with low cardinality
            if df_sample[col].dtype == 'object':
                unique_values = df_sample[col].nunique()
                if unique_values <= min(50, len(df_sample) * 0.2):  # More permissive heuristic
                    categorical_cols.append(col)
                    continue
                    
            # Check if numeric with low cardinality
            if pd.api.types.is_numeric_dtype(df_sample[col]):
                unique_values = df_sample[col].nunique()
                if unique_values <= min(15, len(df_sample) * 0.1):  # Less strict for numeric
                    categorical_cols.append(col)
        
        # Analyze each categorical column
        categorical_analysis = {}
        for col in categorical_cols:
            # Get value counts including missing values
            value_counts = df_sample[col].value_counts(dropna=False).head(30)
            
            # Check for potential case inconsistencies
            case_inconsistencies = self._detect_case_inconsistencies(df_sample[col])
            
            # Check for potential standardization issues (e.g., US, USA, United States)
            standardization_issues = self._detect_standardization_issues(df_sample[col])
            
            # Check for invalid values if column name suggests a specific domain
            invalid_values = self._detect_invalid_domain_values(col, df_sample[col])
            
            # Convert to serializable format
            values = []
            for val, count in value_counts.items():
                # Handle NaN and None values
                if pd.isna(val):
                    val_str = 'Missing'
                    is_valid = False
                else:
                    val_str = str(val)
                    is_valid = val_str not in invalid_values
                    
                values.append({
                    'value': val_str,
                    'count': int(count),
                    'percentage': float(count / len(df_sample) * 100),
                    'is_valid': is_valid
                })
            
            categorical_analysis[col] = {
                'unique_values': int(df_sample[col].nunique()),
                'top_values': values,
                'missing_count': int(df_sample[col].isna().sum()),
                'case_inconsistencies': case_inconsistencies,
                'standardization_issues': standardization_issues,
                'invalid_values': invalid_values,
                'has_quality_issues': len(case_inconsistencies) > 0 or len(standardization_issues) > 0 or len(invalid_values) > 0
            }
        
        return {
            'categorical_columns': categorical_cols,
            'analysis': categorical_analysis
        }
    
    def analyze_categorical_consistency(self) -> Dict[str, Any]:
        """Analyze categorical columns for consistency issues"""
        df_sample = self._get_sample()
        
        consistency_issues = {}
        
        # Only analyze object and category columns
        for col in df_sample.select_dtypes(include=['object', 'category']).columns:
            values = df_sample[col].dropna()
            if len(values) == 0:
                continue
                
            # Convert to strings for consistency analysis
            values = values.astype(str)
            
            # Check for case inconsistencies
            lowercase_values = values.str.lower()
            case_inconsistencies = (values != lowercase_values)
            
            # Check for whitespace inconsistencies
            stripped_values = values.str.strip()
            space_inconsistencies = (values != stripped_values)
            
            # Check for rare values (potential typos)
            value_counts = values.value_counts()
            total_count = len(values)
            rare_threshold = max(1, total_count * 0.01)  # Values appearing in less than 1% of rows
            rare_values = value_counts[value_counts < rare_threshold]
            
            # Detect potentially invalid values using regex patterns
            # Example: detecting values that don't match expected patterns
            invalid_values = []
            
            # Only add to issues if there are actual problems
            if case_inconsistencies.sum() > 0 or space_inconsistencies.sum() > 0 or len(rare_values) > 0:
                consistency_issues[col] = {
                    'case_inconsistencies': int(case_inconsistencies.sum()),
                    'space_inconsistencies': int(space_inconsistencies.sum()),
                    'rare_values_count': len(rare_values),
                    'rare_values': rare_values.index.tolist()[:10],  # Limit to top 10
                    'invalid_values': invalid_values,
                    'examples': {
                        'case_variations': self._get_case_variation_examples(values[case_inconsistencies]),
                        'whitespace_variations': self._get_whitespace_variation_examples(values[space_inconsistencies])
                    }
                }
                
        return {
            'columns_with_issues': list(consistency_issues.keys()),
            'consistency_issues': consistency_issues
        }
    
    def _get_case_variation_examples(self, values, max_examples: int = 5) -> Dict[str, List[str]]:
        """Get examples of case variations for the same value"""
        if len(values) == 0:
            return {}
            
        examples = {}
        for val in values.head(max_examples):
            lower_val = val.lower()
            if lower_val not in examples:
                examples[lower_val] = []
            
            # Find all variations of this value in the dataset
            variations = values[values.str.lower() == lower_val].unique().tolist()
            examples[lower_val] = variations[:max_examples]
            
        return examples
    
    def _get_whitespace_variation_examples(self, values, max_examples: int = 5) -> Dict[str, List[str]]:
        """Get examples of whitespace variations for the same value"""
        if len(values) == 0:
            return {}
            
        examples = {}
        for val in values.head(max_examples):
            stripped_val = val.strip()
            if stripped_val not in examples:
                examples[stripped_val] = []
            
            # Find all variations of this value in the dataset
            variations = values[values.str.strip() == stripped_val].unique().tolist()
            examples[stripped_val] = variations[:max_examples]
            
        return examples
    
    def get_full_analysis(self) -> Dict[str, Any]:
        """Run all analyses and return comprehensive results"""
        return {
            'missing_values': self.analyze_missing_values(),
            'duplicates': self.analyze_duplicates(),
            'data_types': self.analyze_data_types(),
            'categorical_consistency': self.analyze_categorical_consistency(),
            'row_count': self.total_rows,
            'column_count': len(self.df.columns),
            'id_columns': self.id_columns
        }
