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
    # Fill missing count with 0
    df[permit_type] = df[permit_type].fillna(0)
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
    Build a list of ticks (in log-space) from 0 up to new_cmax,
    including integer steps and always including the min (0) and max value.

    Returns (tick_vals, tick_text) for coloraxis ticks.
    Example:
       new_cmax = 3.5 -> tick_vals=[0,1,2,3,3.5], tick_text=['1','10','100','1000','3162']
    """
    if new_cmax <= 0:
        return [0], ["1"]

    # integer part
    floor_val = int(math.floor(new_cmax))  # e.g. 3 if cmax=3.5
    tick_vals = list(range(floor_val + 1)) # [0,1,2,3]

    # Always include the max value if it's not already included
    if new_cmax > floor_val:
        max_val = round(new_cmax, 2)
        if max_val not in tick_vals:
            tick_vals.append(max_val)

    # Generate tick text for all values
    tick_text = [f"{int(10**v) if v.is_integer() else 10**v:.0f}" for v in tick_vals]
    return tick_vals, tick_text

def should_use_log_scale(values, threshold=100):
    """
    Determine if a log scale would be appropriate based on data distribution.
    Returns True if the 99th percentile of the data is greater than 100.
    """
    non_zero = values[values > 0]
    if len(non_zero) < 2:  # Need at least 2 points for meaningful comparison
        return False
        
    # Calculate 99th percentile
    p99 = np.percentile(non_zero, 99)
    return p99 > 100

def get_colorscale_params(values, use_log=False):
    """
    Get appropriate colorscale parameters based on the data and scale type.
    Returns tuple of (zmin, zmax, colorscale_name, colorbar_title_suffix)
    """
    # Create custom colorscale that goes from light blue to red
    custom_colorscale = [
        [0, 'rgba(50,50,70,0.6)'],      # Dark blue-gray for bottom 5%
        [0.05, 'rgba(100,100,150,0.7)'], # Slightly lighter blue for 5%
        [0.2, 'rgba(200,150,150,0.7)'],  # Muted rose color
        [0.4, 'rgba(255,150,150,0.8)'],  # Light red with more opacity
        [0.6, 'rgba(255,100,100,0.9)'],  # Medium red
        [0.8, 'rgba(255,50,50,0.95)'],   # Brighter red
        [1, 'rgba(255,0,0,1)']           # Pure red for highest values
    ]
    
    if use_log:
        non_zero = values[values > 0]
        if len(non_zero) == 0:
            return 0, 1, custom_colorscale, " (log scale)"
            
        zmin = max(0.1, non_zero.min())  # Avoid log(0)
        zmax = non_zero.max()
        return zmin, zmax, custom_colorscale, " (log scale)"
    else:
        return values.min(), values.max(), custom_colorscale, ""

def build_two_trace_mapbox(
    df_base,
    df_top,
    permit_type,
    cmin_base,
    cmax_base,
    cmin_top,
    cmax_top,
    selecting=False,
    show_all_if_empty=True,
    map_title="",
    use_log_base=False,
    use_log_top=False
):
    import plotly.graph_objects as go
    
    # Use provided log scale parameters
    base_values = df_base[permit_type]
    top_values = df_top[permit_type]
    #logger.info(f"base_values min, max: {base_values.min()}, {base_values.max()}")
    #logger.info(f"top_values min, max: {top_values.min()}, {top_values.max()}")
    
    #logger.info(f"Using log base: {use_log_base} (provided)")
    #logger.info(f"Using log top: {use_log_top} (provided)")
    
    # Determine if we should use the same color scale for both layers
    use_same_scale = df_base.equals(df_top)
    
    # Get colorscale parameters for base layer
    if use_log_base:
        zmin_base, zmax_base, cs_base, suffix_base = get_colorscale_params(base_values, True)
    else:
        # Use provided linear scale bounds
        zmin_base, zmax_base = cmin_base, cmax_base
        cs_base = get_colorscale_params(base_values, False)[2]  # Just get colorscale
        suffix_base = ""
    
    # Base trace with potential log scale
    base_trace = go.Choroplethmapbox(
        geojson=hex_geojson,
        featureidkey="properties.h3_index",
        locations=df_base["h3_index"],
        z=df_base[permit_type],
        zmin=zmin_base,
        zmax=zmax_base,
        colorscale=cs_base,
        marker_opacity=0.75,
        marker_line_width=1.5,
        marker_line_color='rgba(255, 255, 255, 0.3)',
        showscale=False,
        hoverinfo="skip",
        name="Base"
    )
    
    if use_log_base:
        base_trace.update(
            colorbar_title=f"Count{suffix_base}",
            colorbar_tickformat=".1e"
        )
        
    # Get colorscale parameters for top layer
    if use_same_scale:
        # Use the same parameters as the base layer
        zmin_top, zmax_top = zmin_base, zmax_base
        cs_top = cs_base
        suffix_top = suffix_base
    elif use_log_top:
        zmin_top, zmax_top, cs_top, suffix_top = get_colorscale_params(top_values, True)
    else:
        # Use provided linear scale bounds
        zmin_top, zmax_top = cmin_top, cmax_top
        cs_top = get_colorscale_params(top_values, False)[2]  # Just get colorscale
        suffix_top = ""
    
    # Top trace with potential log scale
    top_trace = go.Choroplethmapbox(
        geojson=hex_geojson,
        featureidkey="properties.h3_index",
        locations=df_top["h3_index"],
        z=df_top[permit_type],
        zmin=zmin_top,
        zmax=zmax_top,
        colorscale=cs_top,
        marker_opacity=1,
        marker_line_width=2,
        marker_line_color='rgba(255, 255, 255, 0.8)',
        showscale=True,
        name="Selected",
        hoverinfo="location+z",
        colorbar_title=f"Count{suffix_top}"
    )
    
    if use_log_top:
        top_trace.update(colorbar_tickformat=".1e")

    fig = go.Figure([base_trace, top_trace])
    fig.update_layout(
        title=dict(
            text=map_title, 
            x=0.5,
            font=dict(color='rgba(255, 255, 255, 0.9)'),
            # Remove any background color by not setting it
        ),
        mapbox=dict(
            style="carto-positron",
            center={"lat": 40.7, "lon": -73.9},
            zoom=9
        ),
        margin=dict(r=0, t=30, l=0, b=0),
        dragmode="select",
        uirevision="constant",
        # Add paper and plot background colors
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)'    # Transparent background
    )

    # Update colorbar styling for both traces
    base_trace.update(
        colorbar=dict(
            title=dict(
                text=f"Count{suffix_base}",
                font=dict(color='rgba(255, 255, 255, 0.9)')
            ),
            tickfont=dict(color='rgba(255, 255, 255, 0.9)'),  # This makes the numbers white
            bgcolor='rgba(0,0,0,0)'  # Transparent background
        )
    )

    top_trace.update(
        colorbar=dict(
            title=dict(
                text=f"Count{suffix_top}",
                font=dict(color='rgba(255, 255, 255, 0.9)')
            ),
            tickfont=dict(color='rgba(255, 255, 255, 0.9)'),  # This makes the numbers white
            bgcolor='rgba(0,0,0,0)'  # Transparent background
        )
    )

    return fig

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

    # Update the layout with darker theme to match carto-darkmatter
    fig.update_layout(
        plot_bgcolor='rgb(12, 12, 12)',        # Much darker background
        paper_bgcolor='rgb(12, 12, 12)',       # Matching paper color
        font=dict(
            color='rgba(255, 255, 255, 0.7)'
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.08)',  # More subtle grid
            tickfont=dict(color='rgba(255, 255, 255, 0.7)'),
            zeroline=False,
            gridwidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.08)',  # More subtle grid
            tickfont=dict(color='rgba(255, 255, 255, 0.7)'),
            zeroline=False,
            gridwidth=1
        ),
        margin=dict(l=40, r=40, t=40, b=40),
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
            fillcolor='rgba(255, 255, 255, 0.05)',  # More subtle highlight
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
