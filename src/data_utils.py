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
import plotly.graph_objects as go
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

##logger.info("=== Starting data_utils.py ===")
##logger.info(f"Current working directory: {os.getcwd()}")
##logger.info(f"PROCESSED_DATA_PATH: {PROCESSED_DATA_PATH}")
##logger.info(f"Does data path exist? {os.path.exists(PROCESSED_DATA_PATH)}")

# Check for specific files
hex_file = f"{PROCESSED_DATA_PATH}/nyc_hexes.geojson"
permits_file = f"{PROCESSED_DATA_PATH}/permits_wide.csv"

#logger.info(f"Does {hex_file} exist? {os.path.exists(hex_file)}")
#logger.info(f"Does {permits_file} exist? {os.path.exists(permits_file)}")

# Load hex data and do necessary type conversions:
hex_gdf = gpd.read_file(f"{PROCESSED_DATA_PATH}/nyc_hexes.geojson")
hex_gdf['h3_index'] = hex_gdf['h3_index'].astype(str)
hex_geojson = json.loads(hex_gdf.to_json())

#logger.info(f"Loaded hex_gdf with shape: {hex_gdf.shape}")
#logger.info(f"Sample h3_index values: {hex_gdf['h3_index'].head().tolist()}")

# Load permits data:
permit_counts_path = Path(f"{PROCESSED_DATA_PATH}/permits_wide.csv")
permit_counts_wide = pd.read_csv(permit_counts_path)

#logger.info(f"Loaded permit_counts_wide with shape: {permit_counts_wide.shape}")
#logger.info(f"Available columns: {permit_counts_wide.columns.tolist()}")
#logger.info(f"Sample periods: {permit_counts_wide['period'].unique()[:5].tolist()}")

# Compute quarters & mapping:
quarters = sorted(permit_counts_wide['period'].unique())
quarter_to_index = {q: i for i, q in enumerate(quarters)}



# Permit options and their list:
permit_options = [
    {"label": "New Building (NB)", "value": "NB"},
    {"label": "Demolition (DM)", "value": "DM"},
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
    
# ----------------------------------------------------------------
# Precompute the global 99th percentile for single-quarter values:
# (We look at each row's permit_type value, ignoring sums across quarters)
global_quarterly_99 = {}
for pt in permit_type_list:
    # All single-quarter values for that permit type:
    val_series = permit_counts_wide[pt].fillna(0)
    if len(val_series) == 0:
        global_quarterly_99[pt] = 0
    else:
        global_quarterly_99[pt] = np.percentile(val_series, 99)

# ----------------------------------------------------------------
# Precompute the global 99th percentile for aggregated values:
# (Sum across all quarters for each hex, for each permit type)
global_agg_99 = {}
for pt in permit_type_list:
    # Group by hex, summing across *all* quarters in the entire dataset:
    sub = permit_counts_wide.groupby("h3_index", as_index=False)[pt].sum()
    val_series = sub[pt].fillna(0)
    if len(val_series) == 0:
        global_agg_99[pt] = 0
    else:
        global_agg_99[pt] = np.percentile(val_series, 99)
        
all_hexes = hex_gdf["h3_index"].unique().tolist()

def ensure_all_hexes(df, permit_type):
    """
    Given a DF with columns ["h3_index", permit_type],
    return a DF that has *all* hexes, filling missing counts with 0.
    """
    # Make sure 'h3_index' is the index, then reindex
    df = df.set_index("h3_index")
    # Reindex to *all* hexes
    df = df.reindex(all_hexes)
    # Fill missing count with 0, converting values to int
    df[permit_type] = df[permit_type].fillna(0).astype(int)
    # Move h3_index back to a column
    df.reset_index(inplace=True)
    df.rename(columns={"index": "h3_index"}, inplace=True)
    return df
    
def get_permit_label(permit_value):
    from src.data_utils import permit_options
    # find the matching option
    for opt in permit_options:
        if opt["value"] == permit_value:
            return opt["label"]
    return permit_value  # fallback

def get_subrange_singlequarter_99(permit_type: str, start_label: str, end_label: str) -> float:
    """
    Returns the 99th percentile of single-quarter counts (for permit_type)
    over all quarters in [start_label, end_label].
    """
    sub = permit_counts_wide[
        (permit_counts_wide["period"] >= start_label) &
        (permit_counts_wide["period"] <= end_label)
    ]
    if sub.empty:
        return 0.0  # or np.nan, if you prefer
    return np.percentile(sub[permit_type].fillna(0), 99)

def build_log_ticks(new_cmax):
    """
    Build a list of ticks in log-space with improved spacing.
    """
    if new_cmax <= 0:
        return [0], ["0"]
    
    # Create breaks at 1, 10, 100, etc. up to the max
    max_power = math.ceil(math.log10(new_cmax))
    tick_vals = [0] + [10**i for i in range(max_power)]
    
    # Add the actual maximum if it's not already included
    if new_cmax not in tick_vals:
        tick_vals.append(new_cmax)
    
    # Format tick labels
    tick_text = ["0"] + [f"{int(v):,}" if v < 1000 else f"{v:.0e}" for v in tick_vals[1:]]
    
    return tick_vals, tick_text

def should_use_log_scale(values, threshold=50):
    """
    Determine if a log scale would be appropriate based on data distribution.
    Returns True if data is highly skewed and spans multiple orders of magnitude.
    """
    non_zero = values[values > 0]
    if len(non_zero) < 2:
        return False
    
    p99 = np.percentile(non_zero, 99)
    p50 = np.percentile(non_zero, 50)
    
    # Use log scale if:
    # 1. 99th percentile is > threshold AND
    # 2. Ratio between p99 and median is large (indicates skew)
    return bool(p99 > threshold and (p99 / p50) > 5)  # Lowered ratio threshold from 10 to 5

def get_colorscale_params(values, use_log=False):
    """
    Get appropriate colorscale parameters based on the data distribution.
    Returns tuple of (zmin, zmax, colorscale, colorbar_title_suffix)
    """
    # Avoid calling np.percentile on an empty list/array
    if len(values) == 0:
        return 0, 1, px.colors.sequential.Blues, ""
    # Otherwise, continue with normal processing
    zmax = np.percentile(values, 99)  # Use 99th percentile to avoid outliers
    
    # Create custom colorscale with better separation in lower ranges
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
        # Use 1st percentile as minimum to avoid extreme small values
        zmin = max(0.1, np.percentile(values, 1))
        return zmin, zmax, custom_colorscale, " (log scale)"
    else:
        # For linear scale, use robust min/max
        zmin = values.min()
        
        # If we have a lot of zeros, adjust the scale to give more resolution to non-zero values
        if (values == 0).sum() / len(values) > 0.5:  # If more than 50% zeros
            non_zero = values[values > 0]
            if len(non_zero) > 0:
                zmin = non_zero.min() * 0.9  # Slight buffer below minimum non-zero value
        
        return zmin, zmax, custom_colorscale, ""

# def build_two_trace_mapbox(
#     df_base,
#     df_top,
#     permit_type,
#     cmin_base,
#     cmax_base,
#     cmin_top,
#     cmax_top,
#     selecting=False,
#     show_all_if_empty=True,
#     map_title="",
#     use_log_base=None,  # Changed to None to allow auto-detection
#     use_log_top=None    # Changed to None to allow auto-detection
# ):
#     import plotly.graph_objects as go
    
#     # Get values and determine if log scale should be used
#     base_values = df_base[permit_type]
#     top_values = df_top[permit_type]
    
#     # Auto-detect log scale if not explicitly provided
#     if use_log_base is None:
#         use_log_base = should_use_log_scale(base_values)
#     if use_log_top is None:
#         use_log_top = should_use_log_scale(top_values)
    
#     # Determine if we should use the same color scale for both layers
#     use_same_scale = df_base.equals(df_top)
    
#     # Get colorscale parameters for base layer
#     if use_log_base:
#         zmin_base, zmax_base, cs_base, suffix_base = get_colorscale_params(base_values, True)
#         tick_vals, tick_text = build_log_ticks(math.log10(zmax_base))
#     else:
#         # Use robust scaling even with provided bounds
#         zmin_base = min(cmin_base, base_values.min())
#         zmax_base = min(cmax_base, np.percentile(base_values, 99))
#         _, _, cs_base, suffix_base = get_colorscale_params(base_values, False)
    
#     # Base trace with potential log scale
#     base_trace = go.Choroplethmapbox(
#         geojson=hex_geojson,
#         featureidkey="properties.h3_index",
#         locations=df_base["h3_index"],
#         z=df_base[permit_type],
#         zmin=zmin_base,
#         zmax=zmax_base,
#         colorscale=cs_base,
#         marker_opacity=0.75,
#         marker_line_width=1.5,
#         marker_line_color='rgba(255, 255, 255, 0.3)',
#         showscale=False,
#         hoverinfo="skip",
#         name="Base"
#     )
    
#     if use_log_base:
#         base_trace.update(
#             colorbar=dict(
#                 title=f"Count{suffix_base}",
#                 ticktext=tick_text,
#                 tickvals=tick_vals,
#                 tickformat=".1e"
#             )
#         )
        
#     # Get colorscale parameters for top layer
#     if use_same_scale:
#         # Use the same parameters as the base layer
#         zmin_top, zmax_top = zmin_base, zmax_base
#         cs_top = cs_base
#         suffix_top = suffix_base
#         tick_vals_top, tick_text_top = tick_vals, tick_text if use_log_base else (None, None)
#     elif use_log_top:
#         zmin_top, zmax_top, cs_top, suffix_top = get_colorscale_params(top_values, True)
#         tick_vals_top, tick_text_top = build_log_ticks(math.log10(zmax_top))
#     else:
#         # Use robust scaling even with provided bounds
#         zmin_top = min(cmin_top, top_values.min())
#         zmax_top = min(cmax_top, np.percentile(top_values, 99))
#         _, _, cs_top, suffix_top = get_colorscale_params(top_values, False)
#         tick_vals_top, tick_text_top = None, None
    
#     # Top trace with potential log scale
#     top_trace = go.Choroplethmapbox(
#         geojson=hex_geojson,
#         featureidkey="properties.h3_index",
#         locations=df_top["h3_index"],
#         z=df_top[permit_type],
#         zmin=zmin_top,
#         zmax=zmax_top,
#         colorscale=cs_top,
#         marker_opacity=1,
#         marker_line_width=2,
#         marker_line_color='rgba(255, 255, 255, 0.8)',
#         showscale=True,
#         name="Selected",
#         hoverinfo="location+z",
#         colorbar_title=f"Count{suffix_top}"
#     )
    
#     if use_log_top:
#         top_trace.update(
#             colorbar=dict(
#                 title=f"Count{suffix_top}",
#                 ticktext=tick_text_top,
#                 tickvals=tick_vals_top,
#                 tickformat=".1e"
#             )
#         )

#     fig = go.Figure([base_trace, top_trace])
#     fig.update_layout(
#         title=dict(
#             text=map_title, 
#             x=0.5,
#             font=dict(color='rgba(255, 255, 255, 0.9)')
#         ),
#         mapbox=dict(
#             style="carto-positron",
#             center={"lat": 40.7, "lon": -73.9},
#             zoom=9
#         ),
#         margin=dict(r=0, t=30, l=0, b=0),
#         dragmode="select",
#         uirevision="constant",
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)'
#     )

#     # Update colorbar styling for both traces
#     for trace in [base_trace, top_trace]:
#         trace.update(
#             colorbar=dict(
#                 title=dict(
#                     font=dict(color='rgba(255, 255, 255, 0.9)')
#                 ),
#                 tickfont=dict(color='rgba(255, 255, 255, 0.9)'),
#                 bgcolor='rgba(0,0,0,0)'
#             )
#         )

#     return fig

def get_global_max_for_permit_type(permit_type, start_idx=None, end_idx=None):
    """
    Returns the maximum value for a permit type across hex_index × period combinations
    within the specified time range.
    
    Args:
        permit_type: The type of permit to get max value for
        start_idx: Starting quarter index (optional)
        end_idx: Ending quarter index (optional)
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
    fig = go.Figure()
    
    # Add the main line trace
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

    # Update the layout with darker theme and larger ticks
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
                size=14  # Increased font size
            ),
            tickangle=45,  # Rotate labels 45 degrees
            dtick=4,  # Show every 4th tick (changed from 2)
            zeroline=False,
            gridwidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.08)',
            tickfont=dict(
                color='rgba(255, 255, 255, 0.7)',
                size=14  # Increased font size
            ),
            zeroline=False,
            gridwidth=1
        ),
        margin=dict(l=40, r=40, t=40, b=80),  # Increased bottom margin for rotated labels
        hovermode='x unified',
        showlegend=False
    )

    # If there's a selected range, add highlighting
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

# Expose these variables for use in callbacks and layout:
__all__ = [
    "hex_geojson", "permit_counts_wide", "quarters", "quarter_to_index",
    "permit_options", "permit_type_list", "global_color_scales", "global_quarterly_99",
    "global_agg_99", "ensure_all_hexes"
]

# Add this to data_utils.py
DEFAULT_SCALES = {
    "NB": {"aggregated": "4th-root", "quarterly": "6th-root"},
    "DM": {"aggregated": "cube-root", "quarterly": "6th-root"},
    "A1": {"aggregated": "4th-root", "quarterly": "6th-root"},
    "A2": {"aggregated": "4th-root", "quarterly": "6th-root"},
    "A3": {"aggregated": "4th-root", "quarterly": "5th-root"},
    "total_permit_count": {"aggregated": "5th-root", "quarterly": "6th-root"}
}
