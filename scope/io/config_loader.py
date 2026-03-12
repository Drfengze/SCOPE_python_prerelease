"""Configuration file loading for SCOPE.

This module handles loading and parsing of SCOPE configuration files:
- setoptions.csv: Simulation options
- filenames.csv: Input/output file paths
- input_data_*.csv: Parameter values
"""

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from ..types import Options


# Missing value indicators
MISSING_VALUES = {"-9999", "-9999.0", ".", "NA", "NaN", "nan", ""}


def parse_value(value: Any) -> Union[float, int, str, None]:
    """Parse a value from CSV, handling missing values.
    
    Args:
        value: Raw value from CSV file
        
    Returns:
        Parsed value (float, int, str) or None for missing values
    """
    if pd.isna(value):
        return None
    
    str_val = str(value).strip()
    
    if str_val in MISSING_VALUES:
        return None
    
    # Try to parse as number
    try:
        # Check if it's an integer
        float_val = float(str_val)
        if float_val == int(float_val):
            return int(float_val)
        return float_val
    except ValueError:
        return str_val


def load_options(filepath: Union[str, Path]) -> Options:
    """Load simulation options from setoptions.csv.
    
    Args:
        filepath: Path to setoptions.csv
        
    Returns:
        Options dataclass with parsed values
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required options are missing
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Options file not found: {filepath}")
    
    # Read CSV with flexible column handling
    df = pd.read_csv(filepath, skipinitialspace=True)
    
    # Find the name and value columns
    name_col = None
    value_col = None
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ["name", "option", "parameter"]:
            name_col = col
        elif col_lower in ["value", "val"]:
            value_col = col
    
    if name_col is None or value_col is None:
        # Try positional columns
        if len(df.columns) >= 2:
            name_col = df.columns[0]
            value_col = df.columns[1]
        else:
            raise ValueError("Cannot find name and value columns in options file")
    
    # Parse options
    options_dict = {}
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        value = parse_value(row[value_col])
        
        if value is not None:
            # Convert to boolean for boolean options
            if isinstance(value, (int, float)):
                value = bool(int(value))
            options_dict[name] = value
    
    # Create Options with parsed values
    option_fields = {f.name for f in fields(Options)}
    filtered_dict = {k: v for k, v in options_dict.items() if k in option_fields}
    
    return Options(**filtered_dict)


def load_filenames(filepath: Union[str, Path]) -> Dict[str, str]:
    """Load filename mappings from filenames.csv.
    
    Args:
        filepath: Path to filenames.csv
        
    Returns:
        Dictionary mapping file IDs to file paths
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Filenames file not found: {filepath}")
    
    df = pd.read_csv(filepath, skipinitialspace=True)
    
    # Find columns
    id_col = None
    path_col = None
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ["id", "fileid", "file_id"]:
            id_col = col
        elif col_lower in ["path", "filename", "file"]:
            path_col = col
    
    if id_col is None or path_col is None:
        if len(df.columns) >= 2:
            id_col = df.columns[0]
            path_col = df.columns[1]
        else:
            raise ValueError("Cannot find id and path columns in filenames file")
    
    filenames = {}
    for _, row in df.iterrows():
        file_id = str(row[id_col]).strip()
        file_path = str(row[path_col]).strip()
        if file_path and file_path not in MISSING_VALUES:
            filenames[file_id] = file_path
    
    return filenames


def load_input_data(
    filepath: Union[str, Path],
    row_index: Optional[int] = None
) -> Dict[str, Any]:
    """Load parameter values from input_data CSV file.
    
    Args:
        filepath: Path to input_data_*.csv
        row_index: Optional row to load (0-indexed). If None, loads first row.
        
    Returns:
        Dictionary mapping parameter names to values
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Input data file not found: {filepath}")
    
    df = pd.read_csv(filepath, skipinitialspace=True)
    
    # Select row
    if row_index is None:
        row_index = 0
    
    if row_index >= len(df):
        raise ValueError(f"Row index {row_index} out of range (max: {len(df) - 1})")
    
    row = df.iloc[row_index]
    
    # Parse all values
    data = {}
    for col in df.columns:
        value = parse_value(row[col])
        if value is not None:
            data[col.strip()] = value
    
    return data


def load_spectral_data(
    filepath: Union[str, Path],
    wavelength_col: str = "wl"
) -> Dict[str, np.ndarray]:
    """Load spectral data from CSV file.
    
    Args:
        filepath: Path to spectral data CSV
        wavelength_col: Name of wavelength column
        
    Returns:
        Dictionary with wavelength array and data arrays
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Spectral data file not found: {filepath}")
    
    df = pd.read_csv(filepath, skipinitialspace=True)
    
    result = {}
    for col in df.columns:
        col_clean = col.strip()
        result[col_clean] = df[col].values.astype(np.float64)
    
    return result


def load_verification_data(
    filepath: Union[str, Path]
) -> pd.DataFrame:
    """Load verification data CSV file.
    
    Args:
        filepath: Path to verification data CSV
        
    Returns:
        DataFrame with verification data
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Verification file not found: {filepath}")
    
    return pd.read_csv(filepath)
