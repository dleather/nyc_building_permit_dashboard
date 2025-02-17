# src/callbacks.py

from dash.dependencies import Input, Output, State
import dash
from dash import no_update, html
from src.app_instance import app
import logging
from dotenv import load_dotenv
import os
from src.data_utils import create_time_series_figure
import datetime
import pandas as pd
from src.data_utils import permit_counts_wide, quarters
import math
from src.data_utils import should_use_log_scale, get_colorscale_params, build_log_ticks

# Load environment variables from .env file
load_dotenv()

# Access the Mapbox token
mapbox_token = os.getenv('MAPBOX_ACCESS_TOKEN')

# Plotly for the time-series figure:
import plotly.express as px
import plotly.graph_objs as go
from src.config import MAPBOX_STYLE

# Data and helper functions
from src.data_utils import (
    quarters,
    quarter_to_index,
    hex_geojson,
    permit_counts_wide,
    permit_options,
    get_permit_label,
    ensure_all_hexes,
    global_agg_99,
    global_quarterly_99,
    get_subrange_singlequarter_99,
    get_global_max_for_permit_type,
    DEFAULT_SCALES
)

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.info("=== Initial Data Check ===")
logger.info(f"permit_counts_wide shape: {permit_counts_wide.shape}")
logger.info(f"permit_counts_wide columns: {permit_counts_wide.columns.tolist()}")
logger.info(f"Sample of NB values: {permit_counts_wide['NB'].head()}")
logger.info(f"NB column stats: min={permit_counts_wide['NB'].min()}, max={permit_counts_wide['NB'].max()}, mean={permit_counts_wide['NB'].mean():.2f}")

# Add this to your layout somewhere
debug_div = html.Div([
    html.Pre(id='debug-output', style={'whiteSpace': 'pre-wrap'}),
], style={'display': 'none'})  # Set to 'block' to see debug output

def log_callback(func):
    def wrapper(*args, **kwargs):
        ctx = dash.callback_context  # Get the context to see what triggered this callback
        logger.info(
            f"[{datetime.datetime.now()}] Callback '{func.__name__}' triggered with: {ctx.triggered}"
        )
        return func(*args, **kwargs)
    return wrapper

@app.callback(
    Output("dummy-output", "children"),
    [Input("map-quarterly", "selectedData"), Input("map-aggregated", "selectedData")],
    prevent_initial_call=True
)
def debug_map_selections(qtr_sel, agg_sel):
    logger.info("=== debug_map_selections fired ===")
    logger.info("Quarterly selection: %s", qtr_sel)
    logger.info("Aggregated selection: %s", agg_sel)
    return ""
# ------------------------------------------------------------------------------
# 1) Aggragtor callback
# ------------------------------------------------------------------------------
@app.callback(
    [Output("global_filter", "data"),
     Output("scale-type-aggregated", "value"),
     Output("scale-type-quarterly", "value")],
    [Input("period-range-slider", "value"),
     Input("permit-type", "value"),
     Input("play-button", "n_clicks"),
     Input("pause-button", "n_clicks"),
     Input("speed-radio", "value"),
     Input("animation-interval", "n_intervals"),
     Input("map-quarterly", "selectedData"),
     Input("map-aggregated", "selectedData"),
     Input("clear-hexes", "n_clicks"),
     Input("clear-time-range", "n_clicks")],
    [State("global_filter", "data"),
     State("scale-type-aggregated", "value"),
     State("scale-type-quarterly", "value")],
    prevent_initial_call=True
)
def aggregator_callback(
    slider_value,
    permit_value,
    play_clicks,
    pause_clicks,
    speed_value,
    n_intervals,
    qtr_sel,
    agg_sel,
    clear_hexes_clicks,
    clear_time_clicks,
    global_filter,
    current_agg_scale,
    current_qtr_scale
):
    """
    Single aggregator callback that updates global_filter's fields:
      - Time range slider
      - Permit type
      - Play/Pause
      - Speed
      - Advance current quarter
      - Selected hexes
      - Clear hexes
      - Clear time range
    """
    import dash
    from src.data_utils import quarters, permit_counts_wide

    ctx = dash.callback_context  # to see what triggered
    triggered_ids = [t["prop_id"].split(".")[0] for t in ctx.triggered]

    # Copy so we don't mutate global_filter in-place
    new_filter = dict(global_filter)
    new_agg_scale = current_agg_scale
    new_qtr_scale = current_qtr_scale

    # 1) If the slider changed...
    if "period-range-slider" in triggered_ids and slider_value is not None:
        if len(slider_value) == 2:
            start_idx, end_idx = slider_value
            start_idx = max(0, min(start_idx, len(quarters) - 1))
            end_idx   = max(0, min(end_idx, len(quarters) - 1))
            new_filter["startQuarterIndex"] = start_idx
            new_filter["endQuarterIndex"]   = end_idx
            # Clamp currentQuarterIndex into [start_idx, end_idx]
            curr_idx = new_filter.get("currentQuarterIndex", 0)
            if curr_idx < start_idx or curr_idx > end_idx:
                new_filter["currentQuarterIndex"] = start_idx

    # 2) If the permit changed...
    if "permit-type" in triggered_ids and permit_value is not None:
        new_filter["permitType"] = permit_value
        # Get default scales for this permit type
        defaults = DEFAULT_SCALES.get(permit_value, {"aggregated": "linear", "quarterly": "linear"})
        new_agg_scale = defaults["aggregated"]
        new_qtr_scale = defaults["quarterly"]

    # 3) If play/pause changed...
    if "play-button" in triggered_ids:
        new_filter["play"] = True
    if "pause-button" in triggered_ids:
        new_filter["play"] = False

    # 4) If speed changed...
    if "speed-radio" in triggered_ids and speed_value is not None:
        new_filter["speed"] = speed_value

    # 5) If interval ticked => advance quarter
    if "animation-interval" in triggered_ids:
        current_idx = new_filter.get("currentQuarterIndex", 0)
        start_idx   = new_filter.get("startQuarterIndex", 0)
        end_idx     = new_filter.get("endQuarterIndex", len(quarters) - 1)
        new_idx = current_idx + 1
        # Example: wrap around or clamp
        if new_idx > end_idx:
            new_idx = start_idx
        new_filter["currentQuarterIndex"] = new_idx

    # 6) If a map selection changed => update selected hexes
    if any(t in triggered_ids for t in ["map-quarterly", "map-aggregated"]):
        qtr_trigger = ("map-quarterly" in triggered_ids)
        agg_trigger = ("map-aggregated" in triggered_ids)

        # If the user triggered the quarterly map:
        if qtr_trigger and qtr_sel is not None and "points" in qtr_sel:
            idx_list = [pt["pointIndex"] for pt in qtr_sel["points"]]
            cur_q_idx = new_filter.get("currentQuarterIndex", 0)
            quarter_label = quarters[cur_q_idx]
            df_all = permit_counts_wide.loc[permit_counts_wide["period"] == quarter_label].copy()
            df_plot = df_all.sort_values("h3_index").reset_index(drop=True)
            if not idx_list:
                # Show all hexes as selected if empty
                idx_list = df_plot.index.tolist()
            else:
                selected_hexes = df_plot.loc[idx_list, "h3_index"].tolist()
                new_filter["selectedHexes"] = selected_hexes

        # If the user triggered the aggregated map:
        if agg_trigger and agg_sel is not None and "points" in agg_sel:
            idx_list = [pt["pointIndex"] for pt in agg_sel["points"]]
            start_idx = new_filter.get("startQuarterIndex", 0)
            end_idx = new_filter.get("endQuarterIndex", len(quarters)-1)
            s_label = quarters[start_idx]
            e_label = quarters[end_idx]
            df_sub = permit_counts_wide[
                (permit_counts_wide["period"] >= s_label) &
                (permit_counts_wide["period"] <= e_label)
            ].copy()
            df_agg = df_sub.groupby("h3_index", as_index=False)[new_filter["permitType"]].sum()
            df_agg = df_agg.sort_values("h3_index").reset_index(drop=True)
            if not idx_list:
                # Show all hexes as selected if empty
                idx_list = df_agg.index.tolist()
            else:
                selected_hexes = df_agg.loc[idx_list, "h3_index"].tolist()
                new_filter["selectedHexes"] = selected_hexes

    # 7) If user clicked "Clear Hexes"
    if "clear-hexes" in triggered_ids:
        new_filter["selectedHexes"] = []

    # 8) If user clicked "Clear Time Range"
    if "clear-time-range" in triggered_ids:
        new_filter["startQuarterIndex"] = 0
        new_filter["endQuarterIndex"]   = len(quarters) - 1
        # Also clamp currentQuarterIndex
        new_filter["currentQuarterIndex"] = 0

    return new_filter, new_agg_scale, new_qtr_scale


# ------------------------------------------------------------------------------
# 2) CONTROL THE dcc.Interval (id="animation-interval") BASED ON "play" & "speed"
# ------------------------------------------------------------------------------
@app.callback(
    [Output("animation-interval", "disabled"),
     Output("animation-interval", "interval")],
    Input("global_filter", "data")
)
def control_animation_interval(global_filter):
    # If play is False => disable the interval
    # If play is True  => enable it and use the 'speed' from global_filter
    is_playing = global_filter.get("play", False)
    speed      = global_filter.get("speed", 1000)
    return (not is_playing, speed)




# ------------------------------------------------------------------------------
# 3) UPDATE THE QUARTERLY MAP (id="map-quarterly") FROM GLOBAL_FILTER
# ------------------------------------------------------------------------------
from src.data_utils import ensure_all_hexes, global_quarterly_99, all_hexes

@app.callback(
    Output("map-quarterly", "figure"),
    [
        Input("global_filter", "data"),
        Input("map_view_store", "data"),
        Input("scale-type-quarterly", "value")
    ]
)
def update_quarterly_map(global_filter, map_view, scale_type):
    logger.info("=== Updating Quarterly Map ===")
    permit_type = global_filter.get("permitType", "NB")
    current_idx = global_filter.get("currentQuarterIndex", 0)
    selected_hexes = global_filter.get("selectedHexes", [])
    
    # Determine the root value based on permit type as a default, if needed.
    default_root = {
        "NB": 3,
        "DM": 2,
        "A1": 3,
        "A2": 7,
        "A3": 5,
        "ALL": 6
    }.get(permit_type, 3)
    
    # Safeguard against out-of-bounds index
    current_idx = min(current_idx, len(quarters) - 1)
    quarter_label = quarters[current_idx]
    
    logger.info(f"Permit type: {permit_type}")
    logger.info(f"Current quarter: {quarter_label}")
    
    # Get data for the current quarter
    df_all = permit_counts_wide.loc[
        permit_counts_wide["period"] == quarter_label,
        ["h3_index", permit_type]  # Only select needed columns
    ].copy()
    
    df_plot = df_all.sort_values("h3_index").reset_index(drop=True)
    logger.info(f"Data shape for quarter: {df_plot.shape}")
    
    # Calculate indices for selected hexes
    if not selected_hexes:
        # Show all hexes as selected if empty
        selected_idx_list = df_plot.index.tolist()
    else:
        selected_idx_list = df_plot[df_plot["h3_index"].isin(selected_hexes)].index.tolist()
    
    # Compute max value for color scale based on current time range
    slider_start = global_filter.get("startQuarterIndex", 0)
    slider_end = global_filter.get("endQuarterIndex", len(quarters) - 1)
    zmax_quarterly = get_global_max_for_permit_type(permit_type, slider_start, slider_end)
    logger.info(f"Computed zmax_quarterly: {zmax_quarterly}")
    
    # Map the dropdown scale option into transformation parameters.
    if scale_type in {"sqrt", "cube-root", "4th-root", "5th-root", "6th-root"}:
        root_mapping = {"sqrt": 2, "cube-root": 3, "4th-root": 4, "5th-root": 5, "6th-root": 6}
        force_scale = "root"
        root_n = root_mapping[scale_type]
    elif scale_type == "ln":
        force_scale = "ln"
        root_n = None
    elif scale_type == "log":
        force_scale = "log"
        root_n = None
    else:
        force_scale = "linear"
        root_n = None

    fig = build_single_choropleth_map(
        df_plot, 
        permit_type, 
        "Quarterly View", 
        zmax_override=zmax_quarterly, 
        force_scale=force_scale, 
        root_n=root_n if root_n is not None else default_root
    )
    fig.update_traces(selectedpoints=selected_idx_list)
    
    if map_view:
        fig.update_layout(
            mapbox=dict(
                center=map_view.get("center"),
                zoom=map_view.get("zoom"),
                bearing=map_view.get("bearing"),
                pitch=map_view.get("pitch")
            ),
            uirevision="synced-maps",
            mapbox_style=MAPBOX_STYLE
        )
    return fig


# ------------------------------------------------------------------------------
# 4) UPDATE THE AGGREGATED MAP (id="map-aggregated") FROM GLOBAL_FILTER
# ------------------------------------------------------------------------------
@app.callback(
    Output("map-aggregated", "figure"),
    [
        Input("global_filter", "data"),
        Input("map_view_store", "data"),
        Input("scale-type-aggregated", "value")
    ]
)
def update_aggregated_map(global_filter, map_view, scale_type):
    logger.info("=== Updating Aggregated Map ===")
    permit_type = global_filter.get("permitType", "NB")
    start_idx = global_filter.get("startQuarterIndex", 0)
    raw_end_idx = global_filter.get("endQuarterIndex", len(quarters) - 1)
    end_idx = min(raw_end_idx, len(quarters) - 1)
    start_label = quarters[start_idx]
    end_label = quarters[end_idx]
    
    logger.info(f"Permit type: {permit_type}")
    logger.info(f"Date range: {start_label} to {end_label}")
    
    df_sub = permit_counts_wide.loc[
        (permit_counts_wide["period"] >= start_label) &
        (permit_counts_wide["period"] <= end_label)
    ].copy()
    
    logger.info(f"Raw data shape before aggregation: {df_sub.shape}")
    logger.info(f"Raw data {permit_type} stats: min={df_sub[permit_type].min()}, max={df_sub[permit_type].max()}, mean={df_sub[permit_type].mean():.2f}")
    
    if df_sub.empty:
        return px.choropleth_mapbox()
    
    df_agg = df_sub.groupby("h3_index", as_index=False)[permit_type].sum()
    logger.info(f"Aggregated data shape: {df_agg.shape}")
    logger.info(f"Aggregated {permit_type} stats: min={df_agg[permit_type].min()}, max={df_agg[permit_type].max()}, mean={df_agg[permit_type].mean():.2f}")
    
    df_plot = df_agg.sort_values("h3_index").reset_index(drop=True)
    aggregated_zmax = df_agg[permit_type].max()
    logger.info(f"Final zmax value: {aggregated_zmax}")
    
    selected_hexes = global_filter.get("selectedHexes", [])
    if not selected_hexes:
        # Show all hexes as selected if empty
        selected_idx_list = df_plot.index.tolist()
    else:
        selected_idx_list = df_plot[df_plot["h3_index"].isin(selected_hexes)].index.tolist()
    
    # Map the dropdown scale option into transformation parameters.
    if scale_type in {"sqrt", "cube-root", "4th-root", "5th-root", "6th-root"}:
        root_mapping = {"sqrt": 2, "cube-root": 3, "4th-root": 4, "5th-root": 5, "6th-root": 6}
        force_scale = "root"
        root_n = root_mapping[scale_type]
    elif scale_type == "ln":
        force_scale = "ln"
        root_n = None
    elif scale_type == "log":
        force_scale = "log"
        root_n = None
    else:
        force_scale = "linear"
        root_n = None

    fig = build_single_choropleth_map(
        df_plot, 
        permit_type, 
        "Aggregated View", 
        zmax_override=aggregated_zmax,
        force_scale=force_scale,
        root_n=root_n if root_n is not None else 3.5
    )
    fig.update_traces(selectedpoints=selected_idx_list)
    
    if map_view:
        fig.update_layout(
            mapbox=dict(
                center=map_view.get("center"),
                zoom=map_view.get("zoom"),
                bearing=map_view.get("bearing"),
                pitch=map_view.get("pitch")
            ),
            uirevision="synced-maps",
            mapbox_style=MAPBOX_STYLE
        )
    return fig


# ------------------------------------------------------------------------------
# 5) OPTIONAL: UPDATE THE TIME-SERIES ITSELF
# ------------------------------------------------------------------------------
@app.callback(
    Output("time-series", "figure"),
    Input("global_filter", "data")
)
def update_time_series(global_filter):
    permit_type = global_filter.get("permitType", "NB")
    start_idx = global_filter.get("startQuarterIndex", 0)
    end_idx = global_filter.get("endQuarterIndex", len(quarters) - 1)
    current_idx = global_filter.get("currentQuarterIndex", 0)
    selected_hexes = global_filter.get("selectedHexes", [])

    # 1) Filter data based on selected hexes if any
    if selected_hexes:
        df_filtered = permit_counts_wide[permit_counts_wide["h3_index"].isin(selected_hexes)]
    else:
        df_filtered = permit_counts_wide

    # 2) Aggregate by quarter
    agg_ts = df_filtered.groupby("period")[permit_type].sum().reset_index()
    agg_ts["quarter_idx"] = agg_ts["period"].map(quarter_to_index)

    # 3) Create the time series figure with dark theme
    selected_range = [start_idx, end_idx] if (start_idx != 0 or end_idx != len(quarters) - 1) else None
    fig = create_time_series_figure(agg_ts, permit_type, selected_range)

    # 4) Add a vertical dashed line for the current quarter
    if 0 <= current_idx < len(quarters):
        current_quarter_label = quarters[current_idx]
        # If that quarter actually exists in agg_ts['period']:
        if current_quarter_label in agg_ts["period"].values:
            # We can find the y-max
            ymax = agg_ts[permit_type].max() if not agg_ts.empty else 1
            
            # Add white outline effect by creating two lines
            # First, add a slightly thicker white line behind
            fig.add_shape(
                type="line",
                x0=current_quarter_label,
                x1=current_quarter_label,
                y0=0,
                y1=ymax,
                line=dict(
                    color="rgba(255, 255, 255, 0.5)",  # semi-transparent white
                    width=4,  # slightly thicker than the blue line
                    dash="dash"
                ),
                xref="x",
                yref="y"
            )
            
            # Then add the blue line on top
            fig.add_shape(
                type="line",
                x0=current_quarter_label,
                x1=current_quarter_label,
                y0=0,
                y1=ymax,
                line=dict(
                    color="rgba(66, 135, 245, 0.8)",  # semi-transparent blue
                    width=2,
                    dash="dash"
                ),
                xref="x",
                yref="y"
            )

    return fig



@app.callback(
    [Output("map-quarterly-title", "children"),
     Output("map-aggregated-title", "children"),
     Output("time-series-title", "children")],
    Input("global_filter", "data")
)
def update_titles(global_filter):
    permit_type = global_filter.get("permitType", "NB")
    permit_label = get_permit_label(permit_type)
    
    start_idx = global_filter.get("startQuarterIndex", 0)
    end_idx = global_filter.get("endQuarterIndex", len(quarters) - 1)
    start_label = quarters[start_idx]
    end_label = quarters[end_idx]
    
    # Compute the titles for each section
    quarterly_title = f"{permit_label} Permits Issued Across Space and Time"
    aggregated_title = f"{permit_label} Permits Issued from {start_label} - {end_label}"
    time_series_title = f"Time-Series of {permit_label}"
    
    return quarterly_title, aggregated_title, time_series_title

@app.callback(
    Output("period-range-slider", "value"),
    Input("clear-time-range", "n_clicks"),
    prevent_initial_call=True
)
def reset_time_range(n_clicks):
    """
    When the user clicks the Clear Time Range button,
    reset the slider to [0, len(quarters) - 1].
    """
    if n_clicks:
        return [0, len(quarters) - 1]
    # If for some reason it's triggered without clicks, do nothing
    return dash.no_update

@app.callback(
    Output("map_view_store", "data"),
    Input("map-aggregated", "relayoutData"),
    Input("map-quarterly", "relayoutData"),
    State("map_view_store", "data"),
    prevent_initial_call=True
)
def update_map_view(agg_relayout, qtr_relayout, current_view):
    """
    Whenever the user pans or zooms on either map,
    capture the new center/zoom/bearing/pitch from relayoutData
    and store them in map_view_store.data so that the other map
    can match it.
    """
    ctx = dash.callback_context

    if not ctx.triggered:
        return current_view  # no change

    # Figure out which map triggered the callback
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Utility to parse the new center/zoom from relayoutData
    def parse_mapbox_view(rld):
        # Typically rld might have keys like:
        # {
        #   'mapbox.center': {'lon': -73.91, 'lat': 40.77},
        #   'mapbox.zoom': 10.2,
        #   'mapbox.pitch': 0,
        #   'mapbox.bearing': 0,
        #   ...
        # }
        updated = {}
        if not rld:
            return {}
        if 'map.center' in rld:
            updated['center'] = rld['map.center']
        if 'map.zoom' in rld:
            updated['zoom'] = rld['map.zoom']
        if 'map.bearing' in rld:
            updated['bearing'] = rld['map.bearing']
        if 'map.pitch' in rld:
            updated['pitch'] = rld['map.pitch']
        return updated

    # Grab whichever relayoutData is available from the triggered map
    new_view = {}
    if trigger_id == "map-aggregated" and agg_relayout:
        new_view = parse_mapbox_view(agg_relayout)
    elif trigger_id == "map-quarterly" and qtr_relayout:
        new_view = parse_mapbox_view(qtr_relayout)

    # Merge the new view parameters into the existing store
    updated_view = {} if current_view is None else current_view.copy()
    updated_view.update(new_view)

    return updated_view

def build_single_choropleth_map(df_plot, permit_type, map_title, zmax_override=None, force_scale=None, root_n=3):
    """
    Build a single choropleth map with improved color scaling.
    force_scale: None, 'root', or 'log' to force a specific scale type
    root_n: The root to use for root scaling (e.g., 2 for square root, 3 for cube root)
    """
    # Get the values for color scaling
    values = df_plot[permit_type]
    
    # Initialize tick values and text as None
    tick_vals = None
    tick_text = None
    
    # Determine scale type
    if force_scale:
        use_scale = force_scale
    else:
        use_scale = 'linear'
    
    # Handle different scale types
    if use_scale == 'root':
        zmin = 0
        zmax = zmax_override if zmax_override is not None else values.max()
        # Transform values for coloring but keep original for labels
        z_display = np.power(values, 1/root_n)
        z_max_transformed = np.power(zmax, 1/root_n)
        
        # Create tick values that will show untransformed numbers
        n_ticks = 5
        tick_vals = np.linspace(0, z_max_transformed, n_ticks)
        tick_text = [f"{int(val**root_n)}" for val in tick_vals]
        suffix = ""
        
    elif use_scale == 'log':
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        zmin = np.log10(epsilon)
        zmax = np.log10(zmax_override if zmax_override is not None else values.max() + epsilon)
        z_display = np.log10(values + epsilon)
        
        # Create tick values that will show untransformed numbers
        tick_vals = np.linspace(zmin, zmax, 5)
        tick_text = [f"{int(10**val)}" for val in tick_vals]
        suffix = ""
        
    elif use_scale == 'ln':
        epsilon = 1e-10
        zmin = np.log(epsilon)
        zmax = np.log(zmax_override if zmax_override is not None else values.max() + epsilon)
        z_display = np.log(values + epsilon)
        
        n_ticks = 5
        tick_vals = np.linspace(zmin, zmax, n_ticks)
        tick_text = [f"{int(np.exp(val))}" for val in tick_vals]
        suffix = ""
        
    else:  # linear scale
        zmin = 0
        zmax = zmax_override if zmax_override is not None else values.max()
        z_display = values
        suffix = ""

    # Get color scale parameters (using linear scale since we pre-transformed the values)
    _, _, colorscale, _ = get_colorscale_params(values, False)

    colorbar_dict = dict(
        title=dict(
            text=f"Count{suffix}",
            font=dict(color='rgba(255, 255, 255, 0.9)')
        ),
        tickfont=dict(color='rgba(255, 255, 255, 0.9)'),
        bgcolor='rgba(0,0,0,0)'
    )

    # Only add tick values and text if they were set
    if tick_vals is not None and tick_text is not None:
        colorbar_dict.update(dict(
            ticktext=tick_text,
            tickvals=tick_vals
        ))

    # Create the choropleth trace
    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=hex_geojson,
            featureidkey="properties.h3_index",
            locations=df_plot["h3_index"],
            z=z_display,
            marker=dict(
                line=dict(width=1, color="rgba(255, 255, 255, 0.5)")
            ),
            selected=dict(
                marker=dict(opacity=1)
            ),
            unselected=dict(
                marker=dict(opacity=0.3)
            ),
            selectedpoints=None,
            colorscale=colorscale,
            zmin=0 if use_scale == 'root' else zmin,
            zmax=z_max_transformed if use_scale == 'root' else zmax,
            hovertemplate="%{location}<br>Count: %{text}<extra></extra>",
            text=values.round(1),
            colorbar=colorbar_dict
        )
    )
    
    # Rest of the layout remains the same
    fig.update_layout(
        title=dict(
            text=map_title, 
            x=0.5,
            font=dict(color='rgba(255, 255, 255, 0.9)')
        ),
        margin=dict(r=0, t=30, l=0, b=0),
        dragmode="select",
        uirevision="constant",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        mapbox=dict(
            style=MAPBOX_STYLE,
            center={"lat": 40.7, "lon": -73.9},
            zoom=9
        )
    )
    
    return fig