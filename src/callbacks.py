# src/callbacks.py

from dash.dependencies import Input, Output, State
import dash
from dash import no_update, html
from src.app_instance import app
import logging

# Plotly for the time-series figure:
import plotly.express as px
import plotly.graph_objs as go

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
    get_subrange_singlequarter_99
)

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Add this to your layout somewhere
debug_div = html.Div([
    html.Pre(id='debug-output', style={'whiteSpace': 'pre-wrap'}),
], style={'display': 'none'})  # Set to 'block' to see debug output


# ------------------------------------------------------------------------------
# 1) UPDATE GLOBAL_FILTER BASED ON TIME RANGE SLIDER SELECTION
# ------------------------------------------------------------------------------

@app.callback(
    Output("global_filter", "data", allow_duplicate=True),
    Input("period-range-slider", "value"),
    State("global_filter", "data"),
    prevent_initial_call=True
)
def update_range_slider(range_value, global_filter):
    """
    range_value will be [start_idx, end_idx].
    We'll store them in global_filter and return it.
    """
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
    Input("global_filter", "data"),
    Input("map_view_store", "data")
)
def update_quarterly_map(global_filter, map_view):
    permit_type  = global_filter.get("permitType", "NB")
    current_idx  = global_filter.get("currentQuarterIndex", 0)
    start_idx    = global_filter["startQuarterIndex"]
    end_idx      = global_filter["endQuarterIndex"]
    selected_hex = global_filter["selectedHexes"]
    
    logger.info("update_quarterly_map triggered")
    logger.info("global_filter.selectedHexes: %s", selected_hex)
    
    start_label = quarters[start_idx]
    quarter_label = quarters[current_idx]
    end_label   = quarters[end_idx]
    
    df_sub_seln = permit_counts_wide.loc[
        (permit_counts_wide["period"] >= start_label) &
        (permit_counts_wide["period"] <= end_label) &
        (permit_counts_wide["h3_index"].isin(selected_hex))
    ]
    
    df_current = permit_counts_wide.loc[
        permit_counts_wide["period"] == quarter_label
    ].copy()
    
    if df_current.empty:
        return px.choropleth_mapbox()
    
    #df_current = ensure_all_hexes(df_current, permit_type)
    subrange_99 = get_subrange_singlequarter_99(permit_type, start_label, end_label)
    
    cmin_base = 0
    cmax_base = subrange_99
    
    if global_filter.get("resetMaps", False):
        df_top = df_current.copy()
    elif not selected_hex:
        df_top = df_current.copy()
    else:
        df_top = df_current[df_current["h3_index"].isin(selected_hex)]
    logger.info("df_top length: %s", len(df_top))
    #df_top = ensure_all_hexes(df_top, permit_type)
    #logger.info("df_top length: %s", len(df_top))

    cmin_top = 0
    cmax_top = df_sub_seln[permit_type].max() if not df_sub_seln.empty else subrange_99
    
    fig = build_two_trace_mapbox(
        df_base=df_current,
        df_top=df_top,
        permit_type=permit_type,
        cmin_base=cmin_base,
        cmax_base=cmax_base,
        cmin_top=cmin_top,
        cmax_top=cmax_top,
        map_title="Quarterly View"
    )
    
    # If map_view is not defined, fallback to default view settings
    if not map_view:
        map_view = {
            "center": {"lat": 40.7, "lon": -73.9},
            "zoom": 10,
            "bearing": 0,
            "pitch": 0
        }
    
    # Apply stored view settings to keep the maps in sync
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=map_view.get("center"),
            zoom=map_view.get("zoom"),
            bearing=map_view.get("bearing"),
            pitch=map_view.get("pitch")
        ),
        uirevision="synced-maps"  # fixed revision so that user interactions are preserved
    )
    return fig


# ------------------------------------------------------------------------------
# 8) UPDATE THE AGGREGATED MAP (id="map-aggregated") FROM GLOBAL_FILTER
# ------------------------------------------------------------------------------
@app.callback(
    Output("map-aggregated", "figure"),
    Input("global_filter", "data"),
    Input("map_view_store", "data")
)
def update_aggregated_map(global_filter, map_view):
    permit_type = global_filter.get("permitType", "NB")
    start_idx = global_filter.get("startQuarterIndex", 0)
    end_idx = global_filter.get("endQuarterIndex", len(quarters) - 1)
    selected_hex = global_filter.get("selectedHexes", [])
    reset_maps = global_filter.get("resetMaps", False)  # Check for reset flag
    logger.info("update_aggregated_map triggered")
    logger.info("global_filter.selectedHexes: %s", selected_hex)
    logger.info("reset_maps: %s", reset_maps)
    logger.info("start_idx: %s", start_idx)
    logger.info("end_idx: %s", end_idx)

    start_label = quarters[start_idx]
    end_label   = quarters[end_idx]
    
    # 1) Build the base DataFrame (aggregating over the selected time range)
    df_sub = permit_counts_wide.loc[
        (permit_counts_wide["period"] >= start_label) &
        (permit_counts_wide["period"] <= end_label)
    ].copy()
    logger.info("df_sub length: %s", len(df_sub))
    if df_sub.empty:
        return px.choropleth_mapbox()
    
    df_agg = df_sub.groupby("h3_index", as_index=False)[permit_type].sum()
    #df_agg = ensure_all_hexes(df_agg, permit_type)
    
    cmin_base = 0
    cmax_base = global_agg_99[permit_type]
    
    # 2) Determine the top layer based on hex selections or reset flag.
    logger.info("reset_maps: %s", reset_maps)
    logger.info("selected_hex: %s", selected_hex)
    if global_filter.get("resetMaps", False):
        df_top = df_agg.copy()
    elif not selected_hex:
        df_top = df_agg.copy()
    else:
        df_top = df_agg[df_agg["h3_index"].isin(selected_hex)]
    logger.info("df_top length: %s", len(df_top))
    
    # 3) Build the figure using our helper function
    cmin_top = 0
    cmax_top = df_top[permit_type].max() if not df_top.empty else 0
    fig = build_two_trace_mapbox(
        df_base=df_agg,
        df_top=df_top,
        permit_type=permit_type,
        cmin_base=cmin_base,
        cmax_base=cmax_base,
        cmin_top=cmin_top,
        cmax_top=cmax_top,
        map_title="Aggregated View"
    )
    
    # 4) Apply stored view settings and return the figure
    if not map_view:
        map_view = {
            "center": {"lat": 40.7, "lon": -73.9},
            "zoom": 10,
            "bearing": 0,
            "pitch": 0
        }
    
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=map_view.get("center"),
            zoom=map_view.get("zoom"),
            bearing=map_view.get("bearing"),
            pitch=map_view.get("pitch")
        ),
        uirevision="synced-maps"  # maintains the same ui revision across re-renders
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
    end_idx   = global_filter.get("endQuarterIndex", len(quarters) - 1)
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

    # 3) Build the basic line figure
    fig = px.line(
        agg_ts,
        x="quarter_idx",
        y=permit_type,
        template="plotly_white",
        markers=True
    )

    # 4) Style the line
    fig.update_traces(
        line_color='rgb(178, 24, 43)',
        marker_color='rgb(178, 24, 43)',
        line_width=2,
        marker_size=6
    )

    # 5) Build shape objects for:
    #    - The vertical dash line at current_idx
    #    - Shaded regions outside [start_idx, end_idx]
    shapes = [
        # The dashed vertical line indicating current quarter
        dict(
            type="line",
            xref="x",
            yref="paper",
            x0=current_idx,
            x1=current_idx,
            y0=0,
            y1=1,
            line=dict(color="gray", width=3, dash="dash")
        )
    ]

    # If the user's range does NOT start at quarter 0, shade region to the left
    if start_idx > 0:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=-0.5,                   # extends a bit left of x=0
                x1=start_idx - 0.5,        # just before the selected start
                y0=0,
                y1=1,
                fillcolor="rgba(200,200,200,0.3)",  # grayish
                line_width=0,
                layer="below"
            )
        )

    # If the user's range does NOT end at the final quarter, shade region to the right
    if end_idx < len(quarters) - 1:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=end_idx + 0.5,          # just after the selected end
                x1=len(quarters) - 0.5,    # extends to the final quarter's boundary
                y0=0,
                y1=1,
                fillcolor="rgba(200,200,200,0.3)",
                line_width=0,
                layer="below"
            )
        )

    # 6) Update layout with shapes & the usual style
    fig.update_layout(
        shapes=shapes,
        xaxis_title="Time Period",
        yaxis_title=permit_type,
        margin=dict(l=50, r=20, t=10, b=50),
        xaxis=dict(
            range=[-0.5, len(quarters) - 0.5],
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.3)',
            fixedrange=True
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.3)',
            zeroline=False,
            fixedrange=True
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        dragmode=False
    )

    # Configure tick labels (every 4 quarters, etc.)
    fig.update_xaxes(
        tickmode='array',
        tickvals=list(range(0, len(quarters), 4)),
        ticktext=[quarters[i] for i in range(0, len(quarters), 4)],
        tickangle=45
    )

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
    [
        State("global_filter", "data"),
        State("add-to-selection-toggle", "value")
    ],
    prevent_initial_call=True
)
def update_selected_hexes(qtr_sel, agg_sel, clear_n_clicks, global_filter, selection_mode="no"):
    ctx = dash.callback_context
    if not ctx.triggered:
        return global_filter

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Create a new copy of global_filter to modify
    new_filter = dict(global_filter)
    
    # Clear any previous resetMaps flag
    new_filter["resetMaps"] = False
    
    if trigger_id == "clear-hexes":
        new_filter["selectedHexes"] = []
        new_filter["resetMaps"] = True
        return new_filter

    # Handle new selections
    selected_data = qtr_sel if trigger_id == "map-quarterly" else agg_sel
    new_points = (selected_data or {}).get("points", [])
    new_sel = set(p["location"] for p in new_points)
    new_filter["selectedHexes"] = list(new_sel)
    
    return new_filter

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
        if 'mapbox.center' in rld:
            updated['center'] = rld['mapbox.center']
        if 'mapbox.zoom' in rld:
            updated['zoom'] = rld['mapbox.zoom']
        if 'mapbox.bearing' in rld:
            updated['bearing'] = rld['mapbox.bearing']
        if 'mapbox.pitch' in rld:
            updated['pitch'] = rld['mapbox.pitch']
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