# src/callbacks.py

from dash.dependencies import Input, Output, State
from dash import dash, no_update
from src.app import app

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


# ------------------------------------------------------------------------------
# 1) TIME-SERIES: BRUSH -> UPDATE GLOBAL_FILTER (startQuarterIndex, endQuarterIndex)
# ------------------------------------------------------------------------------
@app.callback(
    Output("global_filter", "data"),
    Input("time-series", "selectedData"),       # Brushing on the time-series
    State("global_filter", "data")
)
def update_time_range_from_timeseries(selected_data, global_filter):
    """
    When the user brushes on the time-series, set startQuarterIndex/endQuarterIndex in global_filter.
    If nothing is selected, reset to the full range (or do nothing).
    """
    if not selected_data:
        # No selection => maybe revert to the entire range
        global_filter["startQuarterIndex"] = 0
        global_filter["endQuarterIndex"]   = len(quarters) - 1
        return global_filter

    # selected_data["points"] is a list of points that were selected
    # We'll read their x-values to see which quarter indices were chosen
    x_indices = [pt["x"] for pt in selected_data.get("points", [])]
    if x_indices:
        start_idx = int(np.floor(min(x_indices)))
        end_idx   = int(np.ceil(max(x_indices)))
        # clamp to valid
        start_idx = max(0, start_idx)
        end_idx   = min(len(quarters) - 1, end_idx)
        global_filter["startQuarterIndex"] = start_idx
        global_filter["endQuarterIndex"]   = end_idx
    else:
        # if there's a range brush, you might check selected_data["range"]["x"]
        # or else reset if no points:
        global_filter["startQuarterIndex"] = 0
        global_filter["endQuarterIndex"]   = len(quarters) - 1

    return global_filter


# ------------------------------------------------------------------------------
# 2) PERMIT TYPE RADIO -> UPDATE PERMIT TYPE IN GLOBAL_FILTER
# ------------------------------------------------------------------------------
@app.callback(
    Output("global_filter", "data"),
    Input("permit-type", "value"),  # Radio or Dropdown for permit type
    State("global_filter", "data")
)
def update_permit_type(permit_type, global_filter):
    global_filter["permitType"] = permit_type
    return global_filter


# ------------------------------------------------------------------------------
# 3) PLAY / PAUSE -> UPDATE "play" FIELD IN GLOBAL_FILTER
# ------------------------------------------------------------------------------
@app.callback(
    Output("global_filter", "data"),
    Input("play-button", "n_clicks"),
    Input("pause-button", "n_clicks"),
    State("global_filter", "data")
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
    Output("global_filter", "data"),
    Input("speed-dropdown", "value"),
    State("global_filter", "data")
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
    Output("global_filter", "data"),
    Input("animation-interval", "n_intervals"),
    State("global_filter", "data")
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
    """
    Displays data for the single quarter at global_filter['currentQuarterIndex'].
    """
    current_idx = global_filter.get("currentQuarterIndex", 0)
    permit_type = global_filter.get("permitType", "NB")

    # Clamp the index
    if current_idx < 0 or current_idx >= len(quarters):
        current_idx = 0

    # Convert index -> quarter label
    quarter_label = quarters[current_idx]

    # Now call the data utility to build the figure
    return create_map_for_single_quarter(quarter_label, permit_type)


# ------------------------------------------------------------------------------
# 8) UPDATE THE AGGREGATED MAP (id="map-aggregated") FROM GLOBAL_FILTER
# ------------------------------------------------------------------------------
@app.callback(
    Output("map-aggregated", "figure"),
    Input("global_filter", "data")
)
def update_aggregated_map(global_filter):
    """
    Displays aggregated data from startQuarterIndex to endQuarterIndex.
    """
    start_idx = global_filter.get("startQuarterIndex", 0)
    end_idx   = global_filter.get("endQuarterIndex", len(quarters) - 1)
    permit_type = global_filter.get("permitType", "NB")

    # Clamp
    if start_idx < 0: start_idx = 0
    if end_idx >= len(quarters):
        end_idx = len(quarters) - 1

    # Convert indices -> quarter labels
    start_label = quarters[start_idx]
    end_label   = quarters[end_idx]

    return create_map_for_aggregated(start_label, end_label, permit_type)


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

    fig = px.line(
        agg_ts,
        x="quarter_idx",
        y=permit_type,
        title=f"Time-Series of {permit_type}",
        template="plotly_white",
        markers=True
    )

    # Add a shape for the selected range
    fig.update_layout(
        shapes=[
            # Shaded region for start_idx -> end_idx
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
            # Vertical line for currentQuarterIndex
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
        xaxis=dict(range=[-0.5, len(quarters) - 0.5])  # keep the entire domain visible
    )

    return fig
