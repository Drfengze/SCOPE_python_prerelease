"""Time series data loading for SCOPE.

This module handles loading meteorological and vegetation time series data.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..constants import CONSTANTS
from ..supporting.physics import satvap
from .config_loader import MISSING_VALUES, parse_value


@dataclass
class TimeSeriesData:
    """Container for time series data.
    
    Attributes:
        timestamps: Array of datetime objects
        doy: Day of year array
        t: Time array (decimal hours or similar)
        data: Dictionary mapping variable names to arrays
    """
    timestamps: np.ndarray
    doy: np.ndarray
    t: np.ndarray
    data: Dict[str, np.ndarray]


def parse_timestamp(value: Any) -> Tuple[Optional[datetime], Optional[float]]:
    """Parse timestamp from various formats.
    
    Args:
        value: Timestamp value (DOY, datetime string, etc.)
        
    Returns:
        Tuple of (datetime object or None, DOY or None)
    """
    if pd.isna(value):
        return None, None
    
    str_val = str(value).strip()
    
    # Try parsing as DOY (decimal)
    try:
        doy = float(str_val)
        # Convert DOY to datetime (assuming current year)
        year = datetime.now().year
        dt = datetime(year, 1, 1) + pd.Timedelta(days=doy - 1)
        return dt, doy
    except ValueError:
        pass
    
    # Try parsing as datetime string
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(str_val, fmt)
            # Calculate DOY
            doy = dt.timetuple().tm_yday + dt.hour / 24.0 + dt.minute / 1440.0
            return dt, doy
        except ValueError:
            continue
    
    return None, None


def calc_zenith_angle(
    doy: float,
    hour: float,
    lat: float,
    lon: float,
    timezone: float = 0.0
) -> float:
    """Calculate solar zenith angle.
    
    Args:
        doy: Day of year (1-365/366)
        hour: Hour of day (0-24)
        lat: Latitude in degrees
        lon: Longitude in degrees
        timezone: Timezone offset from UTC
        
    Returns:
        Solar zenith angle in degrees
    """
    # Convert to radians
    lat_rad = np.radians(lat)
    
    # Solar declination
    gamma = 2 * np.pi * (doy - 1) / 365.0
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    
    # Equation of time (in minutes)
    eqtime = (229.18 * (0.000075 + 0.001868 * np.cos(gamma) 
              - 0.032077 * np.sin(gamma) - 0.014615 * np.cos(2 * gamma)
              - 0.040849 * np.sin(2 * gamma)))
    
    # Solar time
    solar_time = hour + (4 * lon - 60 * timezone + eqtime) / 60.0
    
    # Hour angle
    omega = np.radians(15 * (solar_time - 12))
    
    # Zenith angle
    cos_zenith = (np.sin(lat_rad) * np.sin(decl) 
                  + np.cos(lat_rad) * np.cos(decl) * np.cos(omega))
    
    # Clamp to valid range
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    
    return np.degrees(np.arccos(cos_zenith))


def load_timeseries(
    filepath: Union[str, Path],
    timestamp_col: str = "t",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> TimeSeriesData:
    """Load time series data from CSV file.
    
    Handles:
    - Parsing timestamps (DOY or datetime formats)
    - Filtering by date range
    - Computing derived variables (zenith angle, vapor pressure)
    - Unit conversions
    
    Args:
        filepath: Path to time series CSV
        timestamp_col: Name of timestamp column
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        lat: Latitude for zenith angle calculation
        lon: Longitude for zenith angle calculation
        
    Returns:
        TimeSeriesData with parsed data
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Time series file not found: {filepath}")
    
    # Read CSV, replacing missing values with NaN
    df = pd.read_csv(filepath, skipinitialspace=True, na_values=list(MISSING_VALUES))
    
    # Find timestamp column
    if timestamp_col not in df.columns:
        # Try common alternatives
        for col in ["t", "time", "timestamp", "TIMESTAMP", "DOY", "doy"]:
            if col in df.columns:
                timestamp_col = col
                break
    
    # Parse timestamps
    timestamps = []
    doys = []
    
    for val in df[timestamp_col]:
        dt, doy = parse_timestamp(val)
        timestamps.append(dt)
        doys.append(doy)
    
    timestamps = np.array(timestamps)
    doys = np.array(doys, dtype=np.float64)
    
    # Filter by date range
    if start_date is not None or end_date is not None:
        mask = np.ones(len(timestamps), dtype=bool)
        
        if start_date is not None:
            mask &= np.array([t >= start_date if t else False for t in timestamps])
        if end_date is not None:
            mask &= np.array([t <= end_date if t else False for t in timestamps])
        
        df = df[mask].reset_index(drop=True)
        timestamps = timestamps[mask]
        doys = doys[mask]
    
    # Parse data columns
    data = {}
    for col in df.columns:
        if col == timestamp_col:
            continue
        
        col_clean = col.strip()
        values = df[col].values.astype(np.float64)
        
        # Unit conversions
        if col_clean == "p" and np.nanmean(values) < 200:
            # Pressure in kPa, convert to hPa
            values = values * 10
        elif col_clean == "SMC" and np.nanmax(values) > 1:
            # Soil moisture 0-100%, convert to 0-1
            values = values / 100
        elif col_clean == "RH" and np.nanmax(values) > 1:
            # Relative humidity 0-100%, convert to 0-1
            values = values / 100
        
        data[col_clean] = values
    
    # Compute derived variables
    
    # Solar zenith angle (if not provided)
    if "tts" not in data and lat is not None and lon is not None:
        tts = []
        for i, doy in enumerate(doys):
            if not np.isnan(doy):
                hour = (doy % 1) * 24
                tts.append(calc_zenith_angle(doy, hour, lat, lon))
            else:
                tts.append(np.nan)
        data["tts"] = np.array(tts)
    
    # Vapor pressure (if not provided but can be computed)
    if "ea" not in data:
        if "Ta" in data and "RH" in data:
            # ea = RH * es(Ta)
            es = satvap(data["Ta"] + CONSTANTS.C2K)  # Convert to Kelvin
            data["ea"] = data["RH"] * es
        elif "Ta" in data and "VPD" in data:
            # ea = es(Ta) - VPD
            es = satvap(data["Ta"] + CONSTANTS.C2K)
            data["ea"] = es - data["VPD"]
    
    # Time array (decimal hours)
    t = (doys % 1) * 24
    
    return TimeSeriesData(
        timestamps=timestamps,
        doy=doys,
        t=t,
        data=data
    )


@dataclass
class AtmosphericData:
    """Container for atmospheric/irradiance data.
    
    Attributes:
        wl: Wavelength array [nm]
        Esun: Direct solar irradiance [W m-2 nm-1]
        Esky: Diffuse sky irradiance [W m-2 nm-1]
    """
    wl: np.ndarray
    Esun: np.ndarray
    Esky: np.ndarray


def load_atmo(
    filepath: Union[str, Path],
    wl_scope: Optional[np.ndarray] = None
) -> AtmosphericData:
    """Load atmospheric irradiance data.
    
    Args:
        filepath: Path to atmospheric file (.csv or .mat)
        wl_scope: Target wavelength grid for interpolation
        
    Returns:
        AtmosphericData with irradiance spectra
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Atmospheric file not found: {filepath}")
    
    suffix = filepath.suffix.lower()
    
    if suffix == ".csv":
        df = pd.read_csv(filepath)
        
        # Find columns
        wl = None
        Esun = None
        Esky = None
        
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ["wl", "wavelength", "lambda"]:
                wl = df[col].values.astype(np.float64)
            elif col_lower in ["esun", "esun_", "direct"]:
                Esun = df[col].values.astype(np.float64)
            elif col_lower in ["esky", "esky_", "diffuse"]:
                Esky = df[col].values.astype(np.float64)
        
        if wl is None:
            # Assume first column is wavelength
            wl = df.iloc[:, 0].values.astype(np.float64)
        if Esun is None and len(df.columns) >= 2:
            Esun = df.iloc[:, 1].values.astype(np.float64)
        if Esky is None and len(df.columns) >= 3:
            Esky = df.iloc[:, 2].values.astype(np.float64)
        elif Esky is None:
            Esky = np.zeros_like(Esun)
    
    elif suffix == ".mat":
        try:
            from scipy.io import loadmat
            mat = loadmat(str(filepath))
            
            # Try to find arrays
            for key, val in mat.items():
                if key.startswith("_"):
                    continue
                if isinstance(val, np.ndarray):
                    if val.shape[1] >= 2:
                        Esun = val[:, 0].flatten()
                        Esky = val[:, 1].flatten()
                        # Assume wavelength is evenly spaced
                        wl = np.arange(len(Esun)) + 400  # Placeholder
                        break
        except ImportError:
            raise ImportError("scipy required for loading .mat files")
    
    else:
        raise ValueError(f"Unsupported atmospheric file format: {suffix}")
    
    # Interpolate to target wavelength grid if provided
    if wl_scope is not None and wl is not None:
        Esun = np.interp(wl_scope, wl, Esun)
        Esky = np.interp(wl_scope, wl, Esky)
        wl = wl_scope
    
    return AtmosphericData(wl=wl, Esun=Esun, Esky=Esky)
