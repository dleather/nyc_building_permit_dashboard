# Standard data manipulation libraries
import geopandas as gpd
import pandas as pd
import json
import numpy as np
from pathlib import Path

# Project-specific imports
from src.config import PROCESSED_DATA_PATH

# Visualization libraries
import plotly.express as px
import plotly.graph_objects as go

# System utilities
import logging
import os
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Commented out debug logging statements
##logger.info("=== Starting data_utils.py ===")
##logger.info(f"Current working directory: {os.getcwd()}")
##logger.info(f"PROCESSED_DATA_PATH: {PROCESSED_DATA_PATH}")
##logger.info(f"Does data path exist? {os.path.exists(PROCESSED_DATA_PATH)}")

# Define input file paths
hex_file = f"{PROCESSED_DATA_PATH}/nyc_hexes.geojson"
permits_file = f"{PROCESSED_DATA_PATH}/permits_wide.csv"

#logger.info(f"Does {hex_file} exist? {os.path.exists(hex_file)}")
#logger.info(f"Does {permits_file} exist? {os.path.exists(permits_file)}")

# Load and process hex grid data
hex_gdf = gpd.read_file(f"{PROCESSED_DATA_PATH}/nyc_hexes.geojson")
hex_gdf['h3_index'] = hex_gdf['h3_index'].astype(str)  # Convert h3 indices to strings
hex_geojson = json.loads(hex_gdf.to_json())

#logger.info(f"Loaded hex_gdf with shape: {hex_gdf.shape}")
#logger.info(f"Sample h3_index values: {hex_gdf['h3_index'].head().tolist()}")

# Load building permit data
permit_counts_path = Path(f"{PROCESSED_DATA_PATH}/permits_wide.csv")
permit_counts_wide = pd.read_csv(permit_counts_path)

#logger.info(f"Loaded permit_counts_wide with shape: {permit_counts_wide.shape}")
#logger.info(f"Available columns: {permit_counts_wide.columns.tolist()}")
#logger.info(f"Sample periods: {permit_counts_wide['period'].unique()[:5].tolist()}")

# Create lookup for time periods
quarters = sorted(permit_counts_wide['period'].unique())
quarter_to_index = {q: i for i, q in enumerate(quarters)}

# Define permit types and their display labels
permit_options = [
    {"label": "New Building (NB)", "value": "NB"},
    {"label": "Demolition (DM)", "value": "DM"},
    {"label": "Type I - Major Alteration (A1)", "value": "A1"},
    {"label": "Type II - Minor Alteration (A2)", "value": "A2"},
    {"label": "Type III - Minor Alteration (A3)", "value": "A3"},
    {"label": "All Permits", "value": "total_permit_count"}
]
permit_type_list = [opt["value"] for opt in permit_options]

# Calculate global color scales for consistent visualization
global_color_scales = {}
for pt in permit_type_list:
    val_series = permit_counts_wide[pt]
    if val_series.empty:
        global_min = 0
        global_max = 0
    else:
        global_min = val_series.min()
        global_max = np.percentile(val_series, 99)  # Use 99th percentile to handle outliers
    global_color_scales[pt] = (global_min, global_max)
    
# Calculate 99th percentile values for single quarters
global_quarterly_99 = {}
for pt in permit_type_list:
    val_series = permit_counts_wide[pt].fillna(0)
    if len(val_series) == 0:
        global_quarterly_99[pt] = 0
    else:
        global_quarterly_99[pt] = np.percentile(val_series, 99)

# Calculate 99th percentile values for aggregated data
global_agg_99 = {}
for pt in permit_type_list:
    sub = permit_counts_wide.groupby("h3_index", as_index=False)[pt].sum()
    val_series = sub[pt].fillna(0)
    if len(val_series) == 0:
        global_agg_99[pt] = 0
    else:
        global_agg_99[pt] = np.percentile(val_series, 99)
        
# Store list of all hex indices for data completeness checks
all_hexes = hex_gdf["h3_index"].unique().tolist()

def ensure_all_hexes(df, permit_type):
    """
    Helper function to make sure we have data for all hex cells.
    Fills in missing hexes with zero permit counts.
    """
    df = df.set_index("h3_index")
    df = df.reindex(all_hexes)
    df[permit_type] = df[permit_type].fillna(0).astype(int)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "h3_index"}, inplace=True)
    return df
    
def get_permit_label(permit_value):
    """Get the human-readable label for a permit type code"""
    from src.data_utils import permit_options
    for opt in permit_options:
        if opt["value"] == permit_value:
            return opt["label"]
    return permit_value

def get_subrange_singlequarter_99(permit_type: str, start_label: str, end_label: str) -> float:
    """
    Get the 99th percentile value for a date range and permit type.
    Useful for consistent color scaling across time periods.
    """
    sub = permit_counts_wide[
        (permit_counts_wide["period"] >= start_label) &
        (permit_counts_wide["period"] <= end_label)
    ]
    if sub.empty:
        return 0.0
    return np.percentile(sub[permit_type].fillna(0), 99)

def build_log_ticks(new_cmax):
    """
    Create nicely formatted tick marks for log-scale visualizations.
    Handles both small and large numbers appropriately.
    """
    if new_cmax <= 0:
        return [0], ["0"]
    
    max_power = math.ceil(math.log10(new_cmax))
    tick_vals = [0] + [10**i for i in range(max_power)]
    
    if new_cmax not in tick_vals:
        tick_vals.append(new_cmax)
    
    tick_text = ["0"] + [f"{int(v):,}" if v < 1000 else f"{v:.0e}" for v in tick_vals[1:]]
    
    return tick_vals, tick_text

def should_use_log_scale(values, threshold=50):
    """
    Smart detection of when to use log scale based on data distribution.
    Returns True for highly skewed data spanning multiple orders of magnitude.
    """
    non_zero = values[values > 0]
    if len(non_zero) < 2:
        return False
    
    p99 = np.percentile(non_zero, 99)
    p50 = np.percentile(non_zero, 50)
    
    return bool(p99 > threshold and (p99 / p50) > 5)

def get_colorscale_params(values, use_log=False):
    """
    Generate appropriate colorscale parameters based on data distribution.
    Handles both linear and log scales with custom color gradients.
    """
    if len(values) == 0:
        return 0, 1, px.colors.sequential.Blues, ""

    zmax = np.percentile(values, 99)
    
    # Custom colorscale for better visual distinction
    custom_colorscale = [
        [0, 'rgba(240,240,240,0.8)'],    # Very light gray for zeros
        [0.1, 'rgba(215,225,240,0.85)'],  # Light blue-gray
        [0.25, 'rgba(190,210,235,0.9)'],  # Slightly darker blue
        [0.4, 'rgba(220,180,180,0.9)'],   # Light rose
        [0.6, 'rgba(230,150,150,0.92)'],  # Medium rose-red
        [0.8, 'rgba(240,100,100,0.95)'],  # Strong red
        [0.9, 'rgba(250,50,50,0.97)'],    # Very strong red
        [1, 'rgba(255,0,0,1)']            # Pure red
    ]
    
    if use_log:
        zmin = max(0.1, np.percentile(values, 1))
        return zmin, zmax, custom_colorscale, " (log scale)"
    else:
        zmin = values.min()
        
        # Adjust scale if data has many zeros
        if (values == 0).sum() / len(values) > 0.5:
            non_zero = values[values > 0]
            if len(non_zero) > 0:
                zmin = non_zero.min() * 0.9
        
        return zmin, zmax, custom_colorscale, ""

def get_global_max_for_permit_type(permit_type, start_idx=None, end_idx=None):
    """
    Get the maximum value for a permit type, optionally within a time range.
    Used for consistent color scaling across visualizations.
    """
    if start_idx is None or end_idx is None:
        return permit_counts_wide[permit_type].max()
        
    start_label = quarters[start_idx]
    end_label = quarters[end_idx]
    
    return permit_counts_wide.loc[
        (permit_counts_wide["period"] >= start_label) &
        (permit_counts_wide["period"] <= end_label),
        permit_type
    ].max()

def create_time_series_figure(df, permit_type, selected_range=None):
    """
    Create an interactive time series plot with dark theme and highlighting.
    Optimized for readability and user interaction.
    """
    fig = go.Figure()
    
    # Main data line
    fig.add_trace(go.Scatter(
        x=df['period'],
        y=df[permit_type],
        mode='lines+markers',
        name='Permits',
        line=dict(
            color='rgba(255, 99, 132, 0.8)',
            width=2
        ),
        marker=dict(
            size=6,
            color='rgba(255, 99, 132, 0.9)',
            line=dict(
                color='rgba(255, 255, 255, 0.5)',
                width=1
            )
        )
    ))

    # Dark theme styling
    fig.update_layout(
        plot_bgcolor='rgb(12, 12, 12)',
        paper_bgcolor='rgb(12, 12, 12)',
        font=dict(
            color='rgba(255, 255, 255, 0.7)'
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.08)',
            tickfont=dict(
                color='rgba(255, 255, 255, 0.7)',
                size=14
            ),
            tickangle=45,
            dtick=4,
            zeroline=False,
            gridwidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.08)',
            tickfont=dict(
                color='rgba(255, 255, 255, 0.7)',
                size=14
            ),
            zeroline=False,
            gridwidth=1
        ),
        margin=dict(l=40, r=40, t=40, b=80),
        hovermode='x unified',
        showlegend=False
    )

    # Add selection highlight if range is specified
    if selected_range:
        start_period = df.iloc[selected_range[0]]['period']
        end_period = df.iloc[selected_range[1]]['period']
        fig.add_vrect(
            x0=start_period,
            x1=end_period,
            fillcolor='rgba(255, 255, 255, 0.05)',
            layer='below',
            line_width=0
        )

    return fig

# Export key variables for use in other modules
__all__ = [
    "hex_geojson", "permit_counts_wide", "quarters", "quarter_to_index",
    "permit_options", "permit_type_list", "global_color_scales", "global_quarterly_99",
    "global_agg_99", "ensure_all_hexes"
]

# Default scale settings for different permit types
DEFAULT_SCALES = {
    "NB": {"aggregated": "4th-root", "quarterly": "6th-root"},
    "DM": {"aggregated": "cube-root", "quarterly": "6th-root"},
    "A1": {"aggregated": "4th-root", "quarterly": "6th-root"},
    "A2": {"aggregated": "4th-root", "quarterly": "6th-root"},
    "A3": {"aggregated": "4th-root", "quarterly": "5th-root"},
    "total_permit_count": {"aggregated": "5th-root", "quarterly": "6th-root"}
}
