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
    create_map_for_aggregated,
    hex_geojson,
    permit_counts_wide
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
@app.callback(
    Output("map-quarterly", "figure"),
    Input("global_filter", "data")
)
def update_quarterly_map(global_filter):
    current_idx = global_filter.get("currentQuarterIndex", 0)
    permit_type = global_filter.get("permitType", "NB")
    start_idx = global_filter.get("startQuarterIndex", 0)
    end_idx = global_filter.get("endQuarterIndex", len(quarters) - 1)
    selected_hex = global_filter.get("selectedHexes", [])
    
    quarter_label = quarters[current_idx]
    start_label = quarters[start_idx]
    end_label = quarters[end_idx]
    
    # Get the permit data for the current quarter.
    sub_quarter = permit_counts_wide.loc[
        permit_counts_wide["period"] == quarter_label,
        ["h3_index", "period", permit_type]
    ]
    
    # If any hexes are selected, filter down.
    if len(selected_hex) > 0:
        sub_quarter = sub_quarter[sub_quarter["h3_index"].isin(selected_hex)]
    
    # If nothing remains, return an empty figure.
    if sub_quarter.empty:
        fig = px.choropleth_mapbox()
        fig.update_layout(
            title_text="No data (or no hexes selected) for this quarter",
            mapbox_style="carto-positron",
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )
        return fig
    
    # Re-compute subrange max on data in the selected subrange.
    sub = permit_counts_wide[
        (permit_counts_wide["period"] >= start_label) &
        (permit_counts_wide["period"] <= end_label)
    ]
    if len(selected_hex) > 0:
        sub = sub[sub["h3_index"].isin(selected_hex)]
    subrange_max = sub[permit_type].max(skipna=True) if not sub.empty else 0
    
    # Decide whether to use a log scale based on the maximum.
    USE_LOG = (subrange_max > 20)
    if USE_LOG:
        # Create a log-transformed column.
        sub_quarter["log_count"] = np.log10(sub_quarter[permit_type] + 1)
        cmin = 0
        cmax = np.log10(subrange_max + 1) if subrange_max > 0 else 1
        color_col = "log_count"
    else:
        cmin = 0
        cmax = subrange_max
        color_col = permit_type
    
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
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    
    if USE_LOG:
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


# ------------------------------------------------------------------------------
# 8) UPDATE THE AGGREGATED MAP (id="map-aggregated") FROM GLOBAL_FILTER
# ------------------------------------------------------------------------------
@app.callback(
    Output("map-aggregated", "figure"),
    Input("global_filter", "data")
)
def update_aggregated_map(global_filter):
    permit_type = global_filter.get("permitType", "NB")
    start_idx = global_filter.get("startQuarterIndex", 0)
    end_idx = global_filter.get("endQuarterIndex", len(quarters) - 1)
    selected_hex = global_filter.get("selectedHexes", [])
    
    start_label = quarters[start_idx]
    end_label = quarters[end_idx]
    
    # Get the data over the subrange.
    sub = permit_counts_wide[
        (permit_counts_wide["period"] >= start_label) &
        (permit_counts_wide["period"] <= end_label)
    ]
    if len(selected_hex) > 0:
        sub = sub[sub["h3_index"].isin(selected_hex)]
    
    # If nothing is left, return an empty figure.
    if sub.empty:
        fig = px.choropleth_mapbox()
        fig.update_layout(
            title_text="No data (or no hexes selected) for selected time range",
            mapbox_style="carto-positron",
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )
        return fig
    
    # Aggregate permit counts across the subrange.
    grouped = sub.groupby("h3_index", as_index=False)[permit_type].sum()
    
    # Compute the maximum aggregated permit count.
    agg_max = grouped[permit_type].max(skipna=True)
    USE_LOG = (agg_max > 20)
    if USE_LOG:
        grouped["log_count"] = np.log10(grouped[permit_type] + 1)
        cmin = 0
        cmax = np.log10(agg_max + 1) if agg_max > 0 else 1
        color_col = "log_count"
    else:
        cmin = 0
        cmax = agg_max
        color_col = permit_type
    
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
        mapbox_style="carto-positron"
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    
    if USE_LOG:
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