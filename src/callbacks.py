# src/callbacks.py

from dash.dependencies import Input, Output, State
import dash
from dash import no_update, html
from src.app_instance import app
from src.data_utils import permit_options
import logging

# Plotly for the time-series figure:
import plotly.express as px

# Data and helper functions
from src.data_utils import (
    quarters,
    quarter_to_index,
    create_map_for_single_quarter,
    create_map_for_aggregated
)

import numpy as np
import pandas as pd

# Suppose you have a global "permit_counts_wide" for the time-series
# or some other dataset to plot in the time-series
from src.data_utils import permit_counts_wide

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
@app.callback(
    Output("map-quarterly", "figure"),
    Input("global_filter", "data")
)
def update_quarterly_map(global_filter):
    current_idx = global_filter.get("currentQuarterIndex", 0)
    permit_type = global_filter.get("permitType", "NB")
    start_idx   = global_filter.get("startQuarterIndex", 0)
    end_idx     = global_filter.get("endQuarterIndex", len(quarters)-1)
    
    # Convert indices to strings
    quarter_label = quarters[current_idx]
    start_label   = quarters[start_idx]
    end_label     = quarters[end_idx]
    
    fig = create_map_for_single_quarter(
        quarter_label=quarter_label,
        start_quarter=start_label,
        end_quarter=end_label,
        permit_type=permit_type
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
    permit_type = global_filter.get("permitType", "NB")
    start_idx   = global_filter.get("startQuarterIndex", 0)
    end_idx     = global_filter.get("endQuarterIndex", len(quarters)-1)
    
    start_label = quarters[start_idx]
    end_label   = quarters[end_idx]
    
    fig = create_map_for_aggregated(
        start_quarter=start_label,
        end_quarter=end_label,
        permit_type=permit_type
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
    """
    Simple example: plot total permits over all quarters for the chosen permit type.
    Optionally highlight the range (start_idx -> end_idx) or the current quarter.
    """
    permit_type = global_filter.get("permitType", "NB")
    start_idx = global_filter.get("startQuarterIndex", 0)
    end_idx   = global_filter.get("endQuarterIndex", len(quarters) - 1)
    current_idx = global_filter.get("currentQuarterIndex", 0)

    # Build a simple line chart from permit_counts_wide
    # that sums over all hexes (or do something more granular if you like)
    agg_ts = permit_counts_wide.groupby("period")[permit_type].sum().reset_index()
    # Create a numeric index to plot on x-axis
    agg_ts["quarter_idx"] = agg_ts["period"].map(quarter_to_index)

    # Get proper permit labels
    permit_label = None
    for opt in permit_options:
        if opt["value"] == permit_type:
            permit_label = opt["label"]
            break

    if permit_label is None:
        permit_label = permit_type  # fallback

    fig = px.line(
        agg_ts,
        x="quarter_idx",
        y=permit_type,
        title=f"Time-Series of {permit_label}" + " Permits Issued",
        template="plotly_white",
        markers=True
    )

    fig.update_layout(
        xaxis_title="Time Period",  # rename the x-axis
        yaxis_title=permit_label,   # rename the y-axis
        shapes=[
            # your existing shapes
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=start_idx,
                x1=end_idx,
                y0=0,
                y1=1,
                fillcolor="lightblue",
                opacity=0.3,
                layer="below",
                line_width=0
            ),
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=current_idx,
                x1=current_idx,
                y0=0,
                y1=1,
                line=dict(color="red", width=2, dash="dot")
            )
        ],
        xaxis=dict(range=[-0.5, len(quarters) - 0.5])
    )

    # Show every 4th tick, rotated 45 degrees
    tick_indices = list(range(0, len(quarters), 4))
    fig.update_xaxes(
        tickmode='array',
        tickvals=tick_indices,  # Every 4th numeric x-value
        ticktext=[quarters[i] for i in tick_indices],  # Corresponding quarter labels
        tickangle=45  # Rotate labels 45 degrees
    )

    return fig

