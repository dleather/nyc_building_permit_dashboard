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
    Input("global_filter", "data")
)
def update_quarterly_map(global_filter):
    permit_type  = global_filter.get("permitType", "NB")
    current_idx  = global_filter.get("currentQuarterIndex", 0)
    start_idx    = global_filter["startQuarterIndex"]
    end_idx      = global_filter["endQuarterIndex"]
    selected_hex = global_filter["selectedHexes"] or []  # might be empty
    permit_type  = global_filter.get("permitType", "NB")

    # Convert indices to actual quarter labels
    start_label = quarters[start_idx]
    quarter_label = quarters[current_idx]
    end_label   = quarters[end_idx]

    # 1) Filter to the chosen subrange AND chosen hexes
    #    (this covers *all* quarters in the subrange, not just the current quarter)
    df_sub_seln = permit_counts_wide.loc[
        (permit_counts_wide["period"] >= start_label) &
        (permit_counts_wide["period"] <= end_label) &
        (permit_counts_wide["h3_index"].isin(selected_hex))
    ]

    # 1) Filter the data to *this quarter* only
    df_current = permit_counts_wide.loc[
        permit_counts_wide["period"] == quarter_label
    ].copy()
    
    
    if df_current.empty:
        return px.choropleth_mapbox()

    # Ensure all hexes are present:
    df_current = ensure_all_hexes(df_current, permit_type)

    # 2) Compute subrange 99th percentile for single-quarter data
    subrange_99 = get_subrange_singlequarter_99(permit_type, start_label, end_label)

    # 3) Decide cmin/cmax for base and top traces
    cmin_base = 0
    cmax_base = subrange_99

    # If no selection => treat as "select everything"
    if not selected_hex:
        selected_hex = df_current["h3_index"].tolist()

    df_top = df_current[df_current["h3_index"].isin(selected_hex)]
    if df_top.empty:
        df_top = df_current.copy()

    # Use the *same* subrange_99 for cmax_top
    cmin_top = 0
    if df_sub_seln.empty:
    # If user hasn't selected any hexes or if subrange is empty, fallback
        cmax_top = subrange_99
    else:
        cmax_top = df_sub_seln[permit_type].max()

    # 4) Build the 2-trace figure
    fig = build_two_trace_mapbox(
        df_base=df_current,
        df_top=df_top,
        permit_type=permit_type,
        cmin_base=cmin_base,
        cmax_base=cmax_base,
        cmin_top=cmin_top,
        cmax_top=cmax_top,
        current_idx=current_idx,
        map_title="Quarterly View"
    )
    return fig


# ------------------------------------------------------------------------------
# 8) UPDATE THE AGGREGATED MAP (id="map-aggregated") FROM GLOBAL_FILTER
# ------------------------------------------------------------------------------
@app.callback(
    Output("map-aggregated", "figure"),
    Input("global_filter", "data")
)
def update_aggregated_map(global_filter):
    permit_type  = global_filter.get("permitType", "NB")
    start_idx    = global_filter.get("startQuarterIndex", 0)
    end_idx      = global_filter.get("endQuarterIndex", len(quarters) - 1)
    current_idx  = global_filter.get("currentQuarterIndex", 0)
    selected_hex = global_filter.get("selectedHexes", [])

    start_label = quarters[start_idx]
    end_label   = quarters[end_idx]

    # ----------------------------------------------------------------------
    # 1) Build the "base" DF as usual: sum over subrange, fill missing with 0
    # ----------------------------------------------------------------------
    df_sub = permit_counts_wide.loc[
        (permit_counts_wide["period"] >= start_label) &
        (permit_counts_wide["period"] <= end_label)
    ].copy()
    if df_sub.empty:
        # Return some empty figure
        return px.choropleth_mapbox()

    df_agg = df_sub.groupby("h3_index", as_index=False)[permit_type].sum()
    df_agg = ensure_all_hexes(df_agg, permit_type)  # fill zero for missing

    # Base color scale (faint layer):
    cmin_base = 0
    cmax_base = global_agg_99[permit_type]  # or your old logic

    # ----------------------------------------------------------------------
    # 2) If no selection => treat as "select everything"
    # ----------------------------------------------------------------------
    if not selected_hex: 
        selected_hex = df_agg["h3_index"].tolist()

    # Now gather the top‐layer subset
    df_top = df_agg[df_agg["h3_index"].isin(selected_hex)]
    if df_top.empty:
        # Edge case: if somehow nothing matches, fallback
        df_top = df_agg.copy()

    # cmax for the top layer is *just the selected subset*
    # Or you might do 99th percentile => np.percentile(df_top[permit_type], 99)
    cmin_top = 0
    cmax_top = df_top[permit_type].max()

    # ----------------------------------------------------------------------
    # 3) Build the 2‐trace figure
    #    - Base layer is faint, no colorbar
    #    - Top layer colorbar is from [0..max(selected)]
    # ----------------------------------------------------------------------
    fig = build_two_trace_mapbox(
        df_base=df_agg,
        df_top=df_top,
        permit_type=permit_type,
        cmin_base=cmin_base,
        cmax_base=cmax_base,
        cmin_top=cmin_top,
        cmax_top=cmax_top,
        current_idx=current_idx,
        map_title="Aggregated View"
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

    if selected_hexes:
        df_filtered = permit_counts_wide[permit_counts_wide["h3_index"].isin(selected_hexes)]
    else:
        df_filtered = permit_counts_wide

    agg_ts = df_filtered.groupby("period")[permit_type].sum().reset_index()
    agg_ts["quarter_idx"] = agg_ts["period"].map(quarter_to_index)

    fig = px.line(
        agg_ts,
        x="quarter_idx",
        y=permit_type,
        template="plotly_white",
        markers=True
    )

    # Update line and marker colors to match Reds colormap
    fig.update_traces(
        line_color='rgb(178, 24, 43)',  # Less intense red from Reds colormap
        marker_color='rgb(178, 24, 43)',
        line_width=2,
        marker_size=6
    )

    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title=get_permit_label(permit_type),
        # Add a light gray vertical line for current time
        shapes=[
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
        ],
        # Minimize whitespace
        margin=dict(l=50, r=20, t=10, b=50),
        xaxis=dict(
            range=[-0.5, len(quarters) - 0.5],
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.3)',  # Light gray grid
            fixedrange=True  # Disable zoom on x-axis
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.3)',  # Light gray grid
            zeroline=False,
            fixedrange=True  # Disable zoom on y-axis
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        # Disable all modebar buttons
        showlegend=False,
        dragmode=False,
    )

    # Update configuration to disable all interactive features
    fig.update_xaxes(
        tickmode='array',
        tickvals=list(range(0, len(quarters), 4)),
        ticktext=[quarters[i] for i in range(0, len(quarters), 4)],
        tickangle=45
    )

    # Add static configuration
    fig.update_layout(
        modebar=dict(
            remove=[
                'zoom', 'pan', 'select', 'lasso2d', 'zoomIn2d', 
                'zoomOut2d', 'autoScale2d', 'resetScale2d',
                'hoverClosestCartesian', 'hoverCompareCartesian',
                'toggleSpikelines'
            ]
        )
    )

    return fig


# ------------------------------------------------------------------------------
# 10) UPDATE THE SELECTED HEXES
# ------------------------------------------------------------------------------
@app.callback(
    Output("global_filter", "data", allow_duplicate=True),
    Input("map-quarterly", "selectedData"),
    Input("map-aggregated", "selectedData"),
    State("global_filter", "data"),
    prevent_initial_call=True
)
def update_selected_hexes(quarterly_sel, aggregated_sel, global_filter):
    """
    Whenever user selects hexes on EITHER map, unify that selection.
    We store them in global_filter["selectedHexes"].
    """
    ctx = dash.callback_context

    # Which input triggered the callback?
    if not ctx.triggered or (quarterly_sel is None and aggregated_sel is None):
        return global_filter

    # Helper to extract h3 indices from the 'selectedData'
    def extract_hexes(selectedData):
        """
        'selectedData' is typically a dict with structure:
            {
                "points": [
                    {"location": "h3_index_string", ...},
                    {"location": "h3_index_string", ...},
                    ...
                ]
            }
        We want to return a list of those location values.
        """
        if not selectedData or "points" not in selectedData:
            return []
        return [p["location"] for p in selectedData["points"] if "location" in p]

    # Extract from whichever map triggered
    q_hexes = extract_hexes(quarterly_sel)
    a_hexes = extract_hexes(aggregated_sel)

    # Here you can decide on how to combine them:
    # Option A: Overwrite the selection with the most recently used map
    # Option B: Union them
    # For simplicity, let's assume we want to unify them:
    newly_selected = set(q_hexes) | set(a_hexes)

    # If you prefer to *only* keep the last map's selection, do:
    # newly_selected = set(q_hexes if q_hexes else a_hexes)

    global_filter["selectedHexes"] = list(newly_selected)
    return global_filter

# ------------------------------------------------------------------------------
# 11) Clear Button -> Clear the selected hexes
# ------------------------------------------------------------------------------
@app.callback(
    Output("global_filter", "data", allow_duplicate=True),
    Input("clear-hexes", "n_clicks"),
    State("global_filter", "data"),
    prevent_initial_call=True
)
def clear_hex_selection(n_clicks, global_filter):
    if n_clicks:
        global_filter["selectedHexes"] = []
    return global_filter

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