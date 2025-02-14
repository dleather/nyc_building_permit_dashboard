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
    # if you only want to scale to *this* quarter's max. But we want subrange scale.
    
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

import plotly.express as px
import numpy as np
import plotly.graph_objects as go

def build_quarterly_figure_faded_px(df, selected_hex, permit_type, hex_geojson, cmin, cmax, start_idx=None, end_idx=None):
    """
    Builds a quarterly choropleth map where unselected hexes appear faint, and
    selected hexes use the normal color scale.

    Parameters:
        df (pd.DataFrame): Dataframe containing hex-level permit counts for a given quarter.
        selected_hex (list of str): List of selected hexes.
        permit_type (str): The permit type being visualized.
        hex_geojson (dict): The geojson containing hex boundaries.
        cmin (float): Minimum color scale value.
        cmax (float): Maximum color scale value.

    Returns:
        plotly.graph_objects.Figure: The choropleth map.
    """

    if not selected_hex:
        # If no hex is selected, treat them *all* as selected
        df["display_value"] = df[permit_type]
    else:
        df["is_selected"] = df["h3_index"].isin(selected_hex)
        df["display_value"] = np.where(df["is_selected"], df[permit_type], 0.1)

    # Generate the figure
    fig = px.choropleth_map(
        df,
        geojson=hex_geojson,
        locations="h3_index",
        featureidkey="properties.h3_index",
        color="display_value",
        color_continuous_scale=["#FFFFFF", "#FF0000"],  # white to red
        range_color=(cmin, cmax),
        map_style="carto-positron",
        zoom=9,
        center={"lat": 40.7, "lon": -73.9},
        opacity=0.8,
        hover_data=["h3_index", permit_type],
    )
    
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    fig.update_layout(
        coloraxis_colorbar=dict(
            title=f"{permit_type}",
        )
    )

    return fig

def build_quarterly_figure_two_traces(
    df,
    selected_hex,
    permit_type,
    hex_geojson,
    cmin_base,
    cmax_base,
    cmin_selected=None,
    cmax_selected=None,
    start_idx=None,
    end_idx=None,
    current_idx=None
):
    """
    Build a 2-trace choropleth mapbox. The base layer uses (cmin_base..cmax_base),
    the selected layer uses (cmin_selected..cmax_selected) if provided.
    Otherwise it defaults to the base range.

    We label the colorbar from the selected layer only.

    If cmax_* > 20, we do log-scaling for that layer.

    Parameters:
        start_idx: Starting quarter index for title display
        end_idx: Ending quarter index for title display
    """
    import numpy as np
    import plotly.graph_objects as go

    # Defensive checks
    if permit_type not in df.columns:
        return go.Figure()

    df = df.copy()
    df[permit_type] = df[permit_type].fillna(0)

    # For safety, if not provided, the selected range = base range
    if cmin_selected is None:
        cmin_selected = cmin_base
    if cmax_selected is None:
        cmax_selected = cmax_base

    # Split DF
    selected_df = df.loc[df["h3_index"].isin(selected_hex)].copy()
    base_df     = df.copy()  # all hexes

    # -- Decide log-scaling for base layer
    use_log_base = (cmax_base > 20)
    if use_log_base:
        base_df["display_value_base"] = np.log10(base_df[permit_type] + 1.0)
        new_cmin_base = 0
        new_cmax_base = np.log10(cmax_base + 1.0)
    else:
        base_df["display_value_base"] = base_df[permit_type]
        new_cmin_base = cmin_base
        new_cmax_base = cmax_base

    # -- Decide log-scaling for selected layer
    use_log_sel = (cmax_selected > 20)
    if use_log_sel:
        selected_df["display_value_sel"] = np.log10(selected_df[permit_type] + 1.0)
        new_cmin_sel = 0
        new_cmax_sel = np.log10(cmax_selected + 1.0)
    else:
        selected_df["display_value_sel"] = selected_df[permit_type]
        new_cmin_sel = cmin_selected
        new_cmax_sel = cmax_selected

    # Build figure
    fig = go.Figure()

    # --- Trace 1: base (all hexes), faint, no colorbar
    fig.add_trace(go.Choroplethmapbox(
        geojson=hex_geojson,
        featureidkey="properties.h3_index",
        locations=base_df["h3_index"],
        z=base_df["display_value_base"],
        colorscale="Reds",
        zmin=new_cmin_base,
        zmax=new_cmax_base,
        marker_line_width=0.5,
        marker_line_color="gray",
        showscale=False,
        hoverinfo="skip",
        marker=dict(opacity=0.5),
        name="Base Layer",
    ))

    # --- Trace 2: selected hexes or a dummy
    if selected_df.empty:
        # Dummy trace that is invisible, but forces a colorbar
        # scaled the same as the base
        disp_col = "display_value_base"
        final_zmin = new_cmin_base
        final_zmax = new_cmax_base
        use_log = use_log_base
        top_layer_name = "All Hexes"  # or "Dummy"
        layer_df = base_df
    else:
        disp_col = "display_value_sel"
        final_zmin = new_cmin_sel
        final_zmax = new_cmax_sel
        use_log = use_log_sel
        top_layer_name = "Selected"
        layer_df = selected_df

    # Build colorbar
    if use_log:
        tick_vals, tick_text = build_log_ticks(final_zmax)
        colorbar_props = dict(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            title=f"{permit_type}"
        )
    else:
        colorbar_props = dict(title=str(permit_type))

    fig.add_trace(go.Choroplethmapbox(
        geojson=hex_geojson,
        featureidkey="properties.h3_index",
        locations=layer_df["h3_index"],
        z=layer_df[disp_col],
        colorscale="Reds",
        zmin=final_zmin,
        zmax=final_zmax,
        marker_line_width=1 if not selected_df.empty else 0,
        marker_line_color="black" if not selected_df.empty else "gray",
        showscale=True,
        colorbar=colorbar_props,
        hoverinfo="location+z" if not selected_df.empty else "none",
        marker=dict(opacity=0.9 if not selected_df.empty else 0),
        name=top_layer_name
    ))

    # Get period labels for title if indices provided
    permit_label = get_permit_label(permit_type)
    if current_idx is not None:
        from src.data_utils import quarters
        quarter_label = quarters[current_idx]
        title_text = f"{quarter_label}"
    else:
        title_text = f"{permit_label}"

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center={"lat": 40.7, "lon": -73.9},
            zoom=9,
        ),
        title={
            "text": title_text,
            "x": 0.5,            # centers the title horizontally
            "xanchor": "center",
            "yanchor": "top"
        },
        margin=dict(r=0, t=0, l=0, b=0)
    )

    return fig


def build_log_ticks(new_cmax):
    """
    Build a list of ticks (in log-space) from 0 up to new_cmax,
    including integer steps and (optionally) the fractional top if needed.

    Returns (tick_vals, tick_text) for coloraxis ticks.
    Example:
       new_cmax = 3.5 -> tick_vals=[0,1,2,3,3.5], tick_text=['1','10','100','1000','3162']
    """
    if new_cmax <= 0:
        return [0], ["1"]

    # integer part
    floor_val = int(math.floor(new_cmax))  # e.g. 3 if cmax=3.5
    tick_vals = list(range(floor_val + 1)) # [0,1,2,3]

    # If there's a fractional part > 0, append it
    if new_cmax > floor_val:
        tick_vals.append(round(new_cmax, 2))  # e.g. 3.5

    tick_text = [f"{10**v:.0f}" for v in tick_vals]
    return tick_vals, tick_text

def build_two_trace_mapbox(
    df_base,
    df_top,
    permit_type,
    cmin_base,
    cmax_base,
    cmin_top,
    cmax_top,
    current_idx=None,
    map_title=None
):
    import plotly.graph_objects as go
    import numpy as np

    # Make sure to fillna(0) or ensure data is present
    df_base = df_base.copy()
    df_base[permit_type] = df_base[permit_type].fillna(0)

    df_top = df_top.copy()
    df_top[permit_type] = df_top[permit_type].fillna(0)

    # Decide log-scaling for base layer
    use_log_base = (cmax_base > 20)
    if use_log_base:
        df_base["display_value"] = np.log10(df_base[permit_type] + 1.0)
        new_cmin_base = 0
        new_cmax_base = np.log10(cmax_base + 1.0)
    else:
        df_base["display_value"] = df_base[permit_type]
        new_cmin_base = cmin_base
        new_cmax_base = cmax_base

    # Decide log-scaling for top layer
    use_log_top = (cmax_top > 20)
    if use_log_top:
        df_top["display_value"] = np.log10(df_top[permit_type] + 1.0)
        new_cmin_top = 0
        new_cmax_top = np.log10(cmax_top + 1.0)
    else:
        df_top["display_value"] = df_top[permit_type]
        new_cmin_top = cmin_top
        new_cmax_top = cmax_top

    # Base trace
    base_trace = go.Choroplethmapbox(
        geojson=hex_geojson,
        featureidkey="properties.h3_index",
        locations=df_base["h3_index"],
        z=df_base["display_value"],
        zmin=new_cmin_base,
        zmax=new_cmax_base,
        colorscale="Reds",
        marker_line_width=0.3,
        marker_line_color="#999",
        marker_opacity=0.4,    # faint
        showscale=False,       # hide base colorbar
        hoverinfo="skip",
        name="Base (faint)"
    )

    # Build colorbar for top trace
    if use_log_top:
        tick_vals, tick_text = build_log_ticks(new_cmax_top)
        colorbar_props = dict(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            title=f"{permit_type}"
        )
    else:
        colorbar_props = dict(title=f"{permit_type}")

    # Top trace (only subset)
    top_trace = go.Choroplethmapbox(
        geojson=hex_geojson,
        featureidkey="properties.h3_index",
        locations=df_top["h3_index"],
        z=df_top["display_value"],
        zmin=new_cmin_top,
        zmax=new_cmax_top,
        colorscale="Reds",
        marker_line_width=1.0,
        marker_line_color="#333",
        marker_opacity=0.9,     # highlight
        showscale=True,         # show top colorbar only
        colorbar=colorbar_props,
        hoverinfo="location+z",
        name="Selected"
    )

    fig = go.Figure([base_trace, top_trace])
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center={"lat": 40.7, "lon": -73.9},
            zoom=9
        ),
        margin=dict(r=0, t=30, l=0, b=0),
        title=dict(
            text=map_title if map_title else "",
            x=0.5
        ),
        dragmode="select"
    )
    return fig


# Expose these variables for use in callbacks and layout:
__all__ = [
    "hex_geojson", "permit_counts_wide", "quarters", "quarter_to_index",
    "permit_options", "permit_type_list", "global_color_scales", "global_quarterly_99",
    "global_agg_99", "ensure_all_hexes"
]
