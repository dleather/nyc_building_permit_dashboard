import geopandas as gpd
import pandas as pd
import json
import numpy as np
from pathlib import Path
from src.config import PROCESSED_DATA_PATH
import plotly.express as px
import logging
import os

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
    
def create_map_for_single_quarter(quarter: str, permit_type: str):
    """
    Build a choropleth map for a single quarter using a specified permit type.
    
    :param quarter: The quarter string (e.g., "2019Q1") to filter on.
    :param permit_type: The column name to use for coloring (e.g., "NB", "DM", "total_permit_count").
    :return: A Plotly figure object (choropleth map).
    """
    
    # Filter rows for the given quarter
    mask = (permit_counts_wide["period"] == quarter)
    data_sub = permit_counts_wide.loc[mask, ["h3_index", permit_type]]
    
    if data_sub.empty:
        # Handle the edge case of no data
        # (maybe return an empty figure or a figure with a note)
        fig = px.choropleth_map()
        fig.update_layout(title_text="No data for selected quarter.")
        return fig

    # Get the color scale range for this permit type
    cmin, cmax = global_color_scales[permit_type]

    fig = px.choropleth_map(
        data_sub,
        geojson=hex_geojson,
        locations="h3_index",
        featureidkey="properties.h3_index",
        color=permit_type,
        color_continuous_scale="Reds",
        range_color=(cmin, cmax),
        map_style="basic",
        zoom=9,
        center={"lat": 40.7, "lon": -73.9},  # Adjust center for NYC
        opacity=0.6,
        labels={permit_type: permit_type}
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig

def create_map_for_aggregated(start_quarter: str, end_quarter: str, permit_type: str):
    """
    Build a choropleth map aggregating the specified permit type 
    from start_quarter through end_quarter (inclusive).
    
    :param start_quarter: The starting quarter (e.g., "2019Q1").
    :param end_quarter: The ending quarter (e.g., "2021Q2").
    :param permit_type: The permit type column to sum (e.g., "NB", "total_permit_count").
    :return: A Plotly figure with the aggregated data.
    """
    
    # 1) Filter rows to the chosen time range
    mask = (
        (permit_counts_wide["period"] >= start_quarter)
        & (permit_counts_wide["period"] <= end_quarter)
    )
    data_range = permit_counts_wide.loc[mask, ["h3_index", permit_type]]
    
    # 2) If no data for this range -> return an empty figure
    if data_range.empty:
        fig = px.choropleth_map()
        fig.update_layout(title_text="No data for selected time range.")
        return fig
    
    # 3) Group by h3_index, summing the chosen permit column
    grouped = data_range.groupby("h3_index", as_index=False)[permit_type].sum()
    
    # 4) Dynamically compute the color scale min/max from the data
    cmin = grouped[permit_type].min()
    cmax = grouped[permit_type].max()
    
    # If cmin == cmax, give a small range so we don’t get a zero-width color scale
    if cmin == cmax:
        cmax = cmin + 1

    # 5) Build the choropleth (adjust to your actual function name if needed)
    fig = px.choropleth_map(
        grouped,
        geojson=hex_geojson,
        locations="h3_index",
        featureidkey="properties.h3_index",
        color=permit_type,
        color_continuous_scale="Reds",
        range_color=(cmin, cmax),  # Renormalize each time based on this subset
        map_style="basic",
        zoom=9,
        center={"lat": 40.7, "lon": -73.9},
        opacity=0.6,
        labels={permit_type: permit_type}
    )
    
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


# Expose these variables for use in callbacks and layout:
__all__ = [
    "hex_geojson", "permit_counts_wide", "quarters", "quarter_to_index",
    "permit_options", "permit_type_list", "global_color_scales"
]
