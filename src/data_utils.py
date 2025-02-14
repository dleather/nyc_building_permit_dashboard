import geopandas as gpd
import pandas as pd
import json
import numpy as np
from pathlib import Path
from src.config import PROCESSED_DATA_PATH
import plotly.express as px
import logging
import os
import numpy as np
import plotly.express as px

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("=== Starting data_utils.py ===")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"PROCESSED_DATA_PATH: {PROCESSED_DATA_PATH}")
logger.info(f"Does data path exist? {os.path.exists(PROCESSED_DATA_PATH)}")

# Check for specific files
hex_file = f"{PROCESSED_DATA_PATH}/nyc_hexes.geojson"
permits_file = f"{PROCESSED_DATA_PATH}/permits_wide.csv"

logger.info(f"Does {hex_file} exist? {os.path.exists(hex_file)}")
logger.info(f"Does {permits_file} exist? {os.path.exists(permits_file)}")

# Load hex data and do necessary type conversions:
hex_gdf = gpd.read_file(f"{PROCESSED_DATA_PATH}/nyc_hexes.geojson")
hex_gdf['h3_index'] = hex_gdf['h3_index'].astype(str)
hex_geojson = json.loads(hex_gdf.to_json())

logger.info(f"Loaded hex_gdf with shape: {hex_gdf.shape}")
logger.info(f"Sample h3_index values: {hex_gdf['h3_index'].head().tolist()}")

# Load permits data:
permit_counts_path = Path(f"{PROCESSED_DATA_PATH}/permits_wide.csv")
permit_counts_wide = pd.read_csv(permit_counts_path)

logger.info(f"Loaded permit_counts_wide with shape: {permit_counts_wide.shape}")
logger.info(f"Available columns: {permit_counts_wide.columns.tolist()}")
logger.info(f"Sample periods: {permit_counts_wide['period'].unique()[:5].tolist()}")

# Compute quarters & mapping:
quarters = sorted(permit_counts_wide['period'].unique())
quarter_to_index = {q: i for i, q in enumerate(quarters)}

# Permit options and their list:
permit_options = [
    {"label": "New Building", "value": "NB"},
    {"label": "Demolition", "value": "DM"},
    {"label": "Type I - Major Alteration (A1)", "value": "A1"},
    {"label": "Type II - Minor Alteration (A2)", "value": "A2"},
    {"label": "Type III - Minor Alteration (A3)", "value": "A3"},
    {"label": "All Permits", "value": "total_permit_count"}
]
permit_type_list = [opt["value"] for opt in permit_options]

# Precompute global color scales for each permit type:
global_color_scales = {}
for pt in permit_type_list:
    val_series = permit_counts_wide[pt]
    if val_series.empty:
        global_min = 0
        global_max = 0
    else:
        global_min = val_series.min()
        global_max = np.percentile(val_series, 99)
    global_color_scales[pt] = (global_min, global_max)

def create_map_for_single_quarter(quarter_label: str, 
                                  start_quarter: str, 
                                  end_quarter: str, 
                                  permit_type: str):
    """
    Show the distribution for one particular quarter (quarter_label),
    but color scale is normalized to the *subrange* [start_quarter..end_quarter].
    """
    # (1) The data for just this one quarter
    sub_quarter = permit_counts_wide.loc[
        permit_counts_wide["period"] == quarter_label,
        ["h3_index", "period", permit_type]
    ]
    if sub_quarter.empty:
        fig = px.choropleth_mapbox()
        fig.update_layout(title_text="No data for selected quarter.")
        return fig
    
    # (2) Find the max single-quarter value *across the entire subrange*
    # so that all quarters in that subrange share the same scale
    subrange_max = get_subrange_singlequarter_max(permit_type, start_quarter, end_quarter)
    # e.g. subrange_max = sub_quarter[permit_type].max() 
    # if you only want to scale to *this* quarter’s max. But we want subrange scale.
    
    # (3) Decide on log vs. linear
    USE_LOG = (subrange_max > 20)
    
    if USE_LOG:
        sub_quarter["log_count"] = np.log10(sub_quarter[permit_type] + 1.0)
        cmin = 0
        cmax = np.log10(subrange_max + 1.0)
        color_col = "log_count"
    else:
        cmin = 0
        cmax = subrange_max
        color_col = permit_type
    
    # (4) Build the figure
    fig = px.choropleth_mapbox(
        sub_quarter,
        geojson=hex_geojson,
        locations="h3_index",
        featureidkey="properties.h3_index",
        color=color_col,
        color_continuous_scale="Reds",
        range_color=(cmin, cmax),
        zoom=9,
        center={"lat": 40.7, "lon": -73.9},
        opacity=0.6,
        mapbox_style="carto-positron"
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    if USE_LOG:
        # Possibly define custom ticks in log space
        possible_ticks = np.arange(0, 7)
        tick_vals = [v for v in possible_ticks if v <= cmax]
        tick_text = [f"{10**v:.0f}" for v in tick_vals]
        
        fig.update_layout(
            coloraxis_colorbar=dict(
                tickmode="array",
                tickvals=tick_vals,
                ticktext=tick_text,
                title=f"{permit_type}",
            )
        )
    else:
        fig.update_layout(
            coloraxis_colorbar=dict(
                title=f"{permit_type}"
            )
        )
    
    return fig



def create_map_for_aggregated(start_quarter: str, end_quarter: str, permit_type: str):
    """
    Build a choropleth that sums the chosen permit type from start_quarter..end_quarter.
    The color scale is normalized *only* for that subrange, so it re-scales each time
    the user changes start/end. We optionally apply a log scale if the subrange max is big.
    """
    
    sub = permit_counts_wide.loc[
        (permit_counts_wide["period"] >= start_quarter) &
        (permit_counts_wide["period"] <= end_quarter),
        ["h3_index", "period", permit_type]
    ]
    if sub.empty:
        fig = px.choropleth_mapbox()
        fig.update_layout(title_text="No data for selected time range.")
        return fig
    
    # (1) Sum across the subrange
    grouped = sub.groupby("h3_index", as_index=False)[permit_type].sum()

    # (2) Find the subrange-wide aggregated max
    agg_max = grouped[permit_type].max()
    
    # (3) Decide on log vs. linear
    USE_LOG = (agg_max > 20)  # or pick your own threshold
    if USE_LOG:
        grouped["log_count"] = np.log10(grouped[permit_type] + 1.0)
        cmin = 0
        cmax = np.log10(agg_max + 1.0)
        color_col = "log_count"
    else:
        cmin = 0
        cmax = agg_max
        color_col = permit_type
    
    # (4) Build the figure
    fig = px.choropleth_mapbox(
        grouped,
        geojson=hex_geojson,
        locations="h3_index",
        featureidkey="properties.h3_index",
        color=color_col,
        color_continuous_scale="Reds",
        range_color=(cmin, cmax),
        zoom=9,
        center={"lat": 40.7, "lon": -73.9},
        opacity=0.6,
        mapbox_style="carto-positron",
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    # (5) Customize the colorbar
    if USE_LOG:
        # Example: custom log ticks at powers of 10
        # E.g. 0 => 10^0=1, 1 => 10^1=10, etc.
        # Filter to only show ticks up to cmax
        possible_ticks = np.arange(0, 7)  # 0..6
        tick_vals = [v for v in possible_ticks if v <= cmax]
        tick_text = [f"{10**v:.0f}" for v in tick_vals]
        
        fig.update_layout(
            coloraxis_colorbar=dict(
                tickmode="array",
                tickvals=tick_vals,
                ticktext=tick_text,
                title=f"{permit_type}",
            )
        )
    else:
        fig.update_layout(
            coloraxis_colorbar=dict(
                title=f"{permit_type}"
            )
        )
    
    return fig

def get_subrange_singlequarter_max(permit_type: str, start_label: str, end_label: str) -> float:
    """
    Return the maximum single-quarter count of `permit_type` over all
    quarters in [start_label, end_label].
    
    For each row in that date range (each (quarter, h3_index)), 
    we just look at the raw value (not summed across quarters).
    """
    sub = permit_counts_wide[
        (permit_counts_wide["period"] >= start_label) &
        (permit_counts_wide["period"] <= end_label)
    ]
    if sub.empty:
        return 0  # or np.nan
    
    return sub[permit_type].max(skipna=True)


def get_subrange_aggregated_max(permit_type: str, start_label: str, end_label: str) -> float:
    """
    Return the maximum aggregated sum of `permit_type` across the subrange [start_label, end_label].
    
    - We sum the permit counts from start_label..end_label for each hex.
    - Then find the max over all hexes.
    """
    sub = permit_counts_wide[
        (permit_counts_wide["period"] >= start_label) &
        (permit_counts_wide["period"] <= end_label)
    ]
    if sub.empty:
        return 0  # or np.nan

    grouped = sub.groupby("h3_index")[permit_type].sum()
    return grouped.max(skipna=True)


# Expose these variables for use in callbacks and layout:
__all__ = [
    "hex_geojson", "permit_counts_wide", "quarters", "quarter_to_index",
    "permit_options", "permit_type_list", "global_color_scales"
]
