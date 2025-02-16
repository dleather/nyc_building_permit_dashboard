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
    build_two_trace_mapbox,
    get_subrange_singlequarter_99,
    get_global_max_for_permit_type
)

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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
# 1) UPDATE GLOBAL_FILTER BASED ON TIME RANGE SLIDER SELECTION
# ------------------------------------------------------------------------------

@app.callback(
    Output("global_filter", "data", allow_duplicate=True),
    Input("period-range-slider", "value"),
    State("global_filter", "data"),
    prevent_initial_call=True
)
@log_callback
def update_range_slider(range_value, global_filter):
    """
    range_value will be [start_idx, end_idx].
    We'll store them in global_filter and return it.
    """
    logger.info("In callback update_range_slider, final new_filter=%s", global_filter)
    if range_value is None or len(range_value) != 2:
        return global_filter  # no change

    start_idx, end_idx = range_value
    start_idx = int(start_idx)
    end_idx = int(end_idx)

    # clamp them just in case
    start_idx = max(0, min(start_idx, len(quarters)-1))
    end_idx   = max(0, min(end_idx, len(quarters)-1))

    global_filter["startQuarterIndex"] = start_idx
    global_filter["endQuarterIndex"]   = end_idx
    
    # If currentQuarterIndex is outside [start_idx, end_idx], clamp to start_idx
    current_idx = global_filter.get("currentQuarterIndex", 0)
    if current_idx < start_idx or current_idx > end_idx:
        global_filter["currentQuarterIndex"] = start_idx

    return global_filter

# ------------------------------------------------------------------------------
# 2) PERMIT TYPE RADIO -> UPDATE PERMIT TYPE IN GLOBAL_FILTER
# ------------------------------------------------------------------------------
@app.callback(
    Output("global_filter", "data", allow_duplicate=True),
    Input("permit-type", "value"),  # Radio or Dropdown for permit type
    State("global_filter", "data"),
    prevent_initial_call=True
)
def update_permit_type(permit_type, global_filter):
    logger.info("In callback update_permit_type, final new_filter=%s", global_filter)
    global_filter["permitType"] = permit_type
    return global_filter


# ------------------------------------------------------------------------------
# 3) PLAY / PAUSE -> UPDATE "play" FIELD IN GLOBAL_FILTER
# ------------------------------------------------------------------------------
@app.callback(
    Output("global_filter", "data", allow_duplicate=True),
    Input("play-button", "n_clicks"),
    Input("pause-button", "n_clicks"),
    State("global_filter", "data"),
    prevent_initial_call=True
)
def toggle_play(play_clicks, pause_clicks, global_filter):
    logger.info("In callback toggle_play, final new_filter=%s", global_filter)
    ctx = dash.callback_context
    if not ctx.triggered:
        return global_filter

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "play-button":
        global_filter["play"] = True
    elif trigger_id == "pause-button":
        global_filter["play"] = False

    return global_filter


# ------------------------------------------------------------------------------
# 4) SPEED DROPDOWN -> UPDATE "speed" FIELD IN GLOBAL_FILTER
# ------------------------------------------------------------------------------
@app.callback(
    Output("global_filter", "data", allow_duplicate=True),
    Input("speed-dropdown", "value"),
    State("global_filter", "data"),
    prevent_initial_call=True
)
def update_speed_dropdown(selected_speed, global_filter):
    logger.info("In callback update_speed_dropdown, final new_filter=%s", global_filter)
    global_filter["speed"] = selected_speed
    return global_filter


# ------------------------------------------------------------------------------
# 5) CONTROL THE dcc.Interval (id="animation-interval") BASED ON "play" & "speed"
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
# 6) ON TICK OF THE INTERVAL -> ADVANCE currentQuarterIndex IN GLOBAL_FILTER
# ------------------------------------------------------------------------------
@app.callback(
    Output("global_filter", "data", allow_duplicate=True),
    Input("animation-interval", "n_intervals"),
    State("global_filter", "data"),
    prevent_initial_call=True
)
def advance_current_quarter(n_intervals, global_filter):
    logger.info("In callback advance_current_quarter, final new_filter=%s", global_filter)
    # Read the current quarter index and the start/end
    current_idx = global_filter.get("currentQuarterIndex", 0)
    start_idx   = global_filter.get("startQuarterIndex", 0)
    end_idx     = global_filter.get("endQuarterIndex", len(quarters) - 1)

    # Increment by 1
    new_idx = current_idx + 1
    if new_idx > end_idx:
        # wrap around to start_idx if you like
        new_idx = start_idx

    global_filter["currentQuarterIndex"] = new_idx
    return global_filter


# ------------------------------------------------------------------------------
# 7) UPDATE THE QUARTERLY MAP (id="map-quarterly") FROM GLOBAL_FILTER
# ------------------------------------------------------------------------------
from src.data_utils import ensure_all_hexes, global_quarterly_99, all_hexes

@app.callback(
    Output("map-quarterly", "figure"),
    [Input("global_filter", "data"), Input("map_view_store", "data")]
)
def update_quarterly_map(global_filter, map_view):
    permit_type = global_filter.get("permitType", "NB")
    selected_idx_list = global_filter.get("selectedIndicesQuarterly", None)
    current_idx = global_filter.get("currentQuarterIndex", 0)
    quarter_label = quarters[current_idx]
    
    df_all = permit_counts_wide.loc[
        permit_counts_wide["period"] == quarter_label
    ].copy()
    df_plot = df_all.sort_values("h3_index").reset_index(drop=True)
    
    fig = build_single_choropleth_map(df_plot, permit_type, "Quarterly View")
    fig.update_traces(selectedpoints=selected_idx_list)
    
    if map_view:
        fig.update_layout(
            map=dict(
                center=map_view.get("center"),
                zoom=map_view.get("zoom"),
                bearing=map_view.get("bearing"),
                pitch=map_view.get("pitch")
            ),
            uirevision="synced-maps"
        )
    return fig


# ------------------------------------------------------------------------------
# 8) UPDATE THE AGGREGATED MAP (id="map-aggregated") FROM GLOBAL_FILTER
# ------------------------------------------------------------------------------
@app.callback(
    Output("map-aggregated", "figure"),
    [Input("global_filter", "data"), Input("map_view_store", "data")]
)
def update_aggregated_map(global_filter, map_view):
    permit_type = global_filter.get("permitType", "NB")
    start_idx = global_filter.get("startQuarterIndex", 0)
    end_idx = global_filter.get("endQuarterIndex", len(quarters) - 1)
    
    start_label = quarters[start_idx]
    end_label = quarters[end_idx]
    
    # Prepare aggregated data over the selected time range
    df_sub = permit_counts_wide.loc[
        (permit_counts_wide["period"] >= start_label) &
        (permit_counts_wide["period"] <= end_label)
    ].copy()
    
    if df_sub.empty:
        return px.choropleth_mapbox()  # Return an empty figure if no data
    
    df_agg = df_sub.groupby("h3_index", as_index=False)[permit_type].sum()
    df_plot = df_agg.sort_values("h3_index").reset_index(drop=True)
    
    selected_idx_list = global_filter.get("selectedIndicesAggregated", None)
    
    # Build the single choropleth map for aggregated data
    fig = build_single_choropleth_map(df_plot, permit_type, "Aggregated View")
    
    # Apply selection styling
    if selected_idx_list is not None:
        fig.update_traces(selectedpoints=selected_idx_list, selector=dict(type="choroplethmapbox"))
    else:
        fig.update_traces(selectedpoints=None, selector=dict(type="choroplethmapbox"))
    
    # Update map view settings
    if map_view:
        fig.update_layout(
            mapbox=dict(
                style=MAPBOX_STYLE,
                accesstoken=mapbox_token,
                center=map_view.get("center"),
                bearing=map_view.get("bearing"),
                pitch=map_view.get("pitch")
            ),
            uirevision="synced-maps"
        )
    return fig


# ------------------------------------------------------------------------------
# 9) OPTIONAL: UPDATE THE TIME-SERIES ITSELF
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
    selected_range = [start_idx, end_idx] if start_idx != 0 or end_idx != len(quarters) - 1 else None
    fig = create_time_series_figure(agg_ts, permit_type, selected_range)

    return fig



# ------------------------------------------------------------------------------
# 10) UPDATE THE SELECTED HEXES
# ------------------------------------------------------------------------------
@app.callback(
    Output("global_filter", "data", allow_duplicate=True),
    [
        Input("map-quarterly", "selectedData"),
        Input("map-aggregated", "selectedData"),
        Input("clear-hexes", "n_clicks")
    ],
    [State("global_filter", "data")],
    prevent_initial_call=True
)
def update_selected_hexes(qtr_sel, agg_sel, clear_n_clicks, global_filter):
    ctx = dash.callback_context
    new_filter = dict(global_filter)
    
    # Clear selection when the user clicks "Clear Hexes"
    if ctx.triggered and "clear-hexes" in ctx.triggered[0]["prop_id"]:
        new_filter["selectedIndicesQuarterly"] = None
        new_filter["selectedIndicesAggregated"] = None
        return new_filter
    
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    sel_data = qtr_sel if trigger == "map-quarterly" else agg_sel
    
    if sel_data and "points" in sel_data:
        # pointIndex is the row index in the sorted DataFrame (i.e. the "point")
        idx_list = [pt["pointIndex"] for pt in sel_data["points"]]
        if trigger == "map-quarterly":
            new_filter["selectedIndicesQuarterly"] = idx_list
        else:
            new_filter["selectedIndicesAggregated"] = idx_list
    else:
        if trigger == "map-quarterly":
            new_filter["selectedIndicesQuarterly"] = None
        else:
            new_filter["selectedIndicesAggregated"] = None
    
    return new_filter

@app.callback(
    Output("global_filter", "data", allow_duplicate=True),
    Input("map-aggregated", "figure"),
    Input("map-quarterly", "figure"),
    State("global_filter", "data"),
    prevent_initial_call=True
)
def clear_reset_flag(_, __, global_filter):
    """Resets the resetMaps flag after maps have updated"""
    logger.info("In callback clear_reset_flag, final new_filter=%s", global_filter)
    if global_filter.get("resetMaps", False):
        new_filter = dict(global_filter)
        new_filter["resetMaps"] = False
        return new_filter
    return dash.no_update

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
    aggregated_title = f"{permit_label} Permits Issued from {start_label} - {end_label} (select hexes here)"
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

def build_single_choropleth_map(df_plot, permit_type, map_title):
    fig = go.Figure(
        go.Choroplethmap(
            geojson=hex_geojson,
            featureidkey="properties.h3_index",
            locations=df_plot["h3_index"],
            z=df_plot[permit_type],
            marker=dict(
                line=dict(width=1, color="white")
            ),
            selected=dict(
                marker=dict(opacity=1)
            ),
            unselected=dict(
                marker=dict(opacity=0.3)
            ),
            selectedpoints=None,
            colorscale="Reds",
            zmin=0,
            zmax=get_global_max_for_permit_type(permit_type),
            hoverinfo="location+z",
        )
    )

    fig.update_layout(
        title=dict(text=map_title, x=0.5),
        margin=dict(r=0, t=30, l=0, b=0),
        dragmode="select",
        uirevision="constant",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        mapbox=dict(
            style=MAPBOX_STYLE,
            accesstoken=mapbox_token,
            center={"lat": 40.7, "lon": -73.9},
            zoom=9
        )
    )
    return fig