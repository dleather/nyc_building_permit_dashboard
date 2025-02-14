# Tree View:
```
.
├─.python-version
├─assets
│ ├─background.png
│ └─style.css
├─codebase.md
├─data
├─pyproject.toml
├─run_codeweaver.bat
└─src
  ├─app.py
  ├─app_instance.py
  ├─callbacks.py
  ├─config.py
  ├─data_utils.py
  ├─debug.py
  └─layout.py
```

# Content:

## .python-version
```python-version
3.13

```

## assets\background.png
```png

```

## assets\style.css
```css

```

## codebase.md
```md
# Tree View:
```
.
├─.python-version
├─assets
│ ├─background.png
│ └─style.css
├─codebase.md
├─data
├─pyproject.toml
├─run_codeweaver.bat
└─src
  ├─app.py
  ├─app_instance.py
  ├─callbacks.py
  ├─config.py
  ├─data_utils.py
  ├─debug.py
  └─layout.py
```

# Content:

## .python-version
```python-version
3.13

```

## assets\background.png
```png

```

## assets\style.css
```css

```


```

## pyproject.toml
```toml
[project]
name = "nyc-building-permit-dashboard"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "asgiref>=3.8.1",
    "dash>=2.18.2",
    "dash-bootstrap-components>=1.7.1",
    "geopandas>=1.0.1",
    "pandas>=2.2.3",
    "plotly>=6.0.0",
    "shapely>=2.0.7",
    "uvicorn>=0.34.0",
]

```

## run_codeweaver.bat
```bat
@echo off
REM This batch file runs CodeWeaver in the current directory
REM and excludes the specified directories and files from the documentation.

REM Define ignore list patterns (each pattern on its own line for clarity)
set "IGNORE_LIST=\.git.*"
set "IGNORE_LIST=%IGNORE_LIST%,__pycache__"
set "IGNORE_LIST=%IGNORE_LIST%,\.venv"
set "IGNORE_LIST=%IGNORE_LIST%,node_modules"
set "IGNORE_LIST=%IGNORE_LIST%,build"
set "IGNORE_LIST=%IGNORE_LIST%,.*\.log"
set "IGNORE_LIST=%IGNORE_LIST%,temp"
set "IGNORE_LIST=%IGNORE_LIST%,^data(/|\\)"
set "IGNORE_LIST=%IGNORE_LIST%,README.html"
set "IGNORE_LIST=%IGNORE_LIST%,README.md"
set "IGNORE_LIST=%IGNORE_LIST%,README_files"
set "IGNORE_LIST=%IGNORE_LIST%,uv.lock"

REM Run CodeWeaver using the ignore list
C:\Users\davle\go\bin\CodeWeaver.exe -input . -ignore="%IGNORE_LIST%"

REM Pause so you can see any output in the command window before it closes.
pause
```

## src\app.py
```py
from dash import Dash
import dash_bootstrap_components as dbc
from asgiref.wsgi import WsgiToAsgi
import logging

# Set up logging before any other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # This ensures we override any existing logging config
)

logger = logging.getLogger(__name__)
logger.info("=== Starting app.py ===")

from src.app_instance import app, server
from src.layout import layout
app.layout = layout
from src import callbacks

# Import callbacks (this registers them)
#asgi_app = WsgiToAsgi(server)

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
    #import uvicorn
    #uvicorn.run("src.app:asgi_app", host="127.0.0.1", port=8000, reload=True) 
```

## src\app_instance.py
```py
from dash import Dash
import dash_bootstrap_components as dbc

# Include both Bootstrap CSS and the theme
external_stylesheets = [
    dbc.themes.FLATLY,  # Try FLATLY theme
    "https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css"
]

app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
server = app.server
```

## src\callbacks.py
```py
# src/callbacks.py

from dash.dependencies import Input, Output, State
import dash
from dash import no_update, html
from src.app_instance import app
from src.data_utils import permit_options, build_quarterly_figure_faded_px, build_quarterly_figure_two_traces, get_permit_label
import logging

# Plotly for the time-series figure:
import plotly.express as px
import plotly.graph_objs as go

# Data and helper functions
from src.data_utils import (
    quarters,
    quarter_to_index,
    create_map_for_single_quarter,
    create_map_for_aggregated,
    hex_geojson,
    permit_counts_wide,
    get_subrange_singlequarter_max,
    get_subrange_aggregated_max
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
    start_idx   = global_filter.get("startQuarterIndex", 0)
    end_idx     = global_filter.get("endQuarterIndex", len(quarters) - 1)
    selected_hex = global_filter.get("selectedHexes", [])

    quarter_label = quarters[current_idx]
    start_label   = quarters[start_idx]
    end_label     = quarters[end_idx]

    df_current = permit_counts_wide.loc[permit_counts_wide["period"] == quarter_label].copy()
    if df_current.empty:
        fig = px.choropleth_mapbox()
        fig.update_layout(
            # Removed inline title; title will be provided by layout outside the graph
            mapbox_style="carto-positron",
            margin={"r":0, "t":0, "l":0, "b":0}
        )
        return fig

    from src.data_utils import get_subrange_singlequarter_max
    cmax_base = get_subrange_singlequarter_max(permit_type, start_label, end_label)

    fig = build_quarterly_figure_two_traces(
        df=df_current,
        selected_hex=selected_hex,
        permit_type=permit_type,
        hex_geojson=hex_geojson,
        cmin_base=0,
        cmax_base=cmax_base,
        start_idx=start_idx,
        end_idx=end_idx,
        current_idx=current_idx
    )
    # Do not set the title here; the layout markdown will show the title.
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
    permit_label = get_permit_label(permit_type)

    df_sub = permit_counts_wide.loc[
        (permit_counts_wide["period"] >= start_label) &
        (permit_counts_wide["period"] <= end_label)
    ].copy()

    if df_sub.empty:
        fig = px.choropleth_mapbox()
        fig.update_layout(
            title_text="No data for selected time range.",
            mapbox_style="carto-positron",
            margin={"r":0,"t":0,"l":0,"b":0}
        )
        return fig

    # Aggregate
    df_agg = df_sub.groupby("h3_index", as_index=False)[permit_type].sum()

    # If we have selected hexes that might be 0, ensure they are present:
    if selected_hex:
        sel_df = pd.DataFrame({"h3_index": selected_hex})
        df_agg = pd.merge(df_agg, sel_df, on="h3_index", how="outer")
        df_agg[permit_type] = df_agg[permit_type].fillna(0)

    if df_agg.empty:
        fig = px.choropleth_mapbox()
        fig.update_layout(
            title_text="No aggregated data for selection/time range",
            mapbox_style="carto-positron",
            margin={"r":0,"t":0,"l":0,"b":0}
        )
        return fig

    # cmax for base = max of all hexes
    cmax_all = df_agg[permit_type].max()
    # cmax for selected = max of selected hexes
    cmax_sel = df_agg.loc[df_agg["h3_index"].isin(selected_hex), permit_type].max() if selected_hex else cmax_all
    if pd.isna(cmax_sel):
        cmax_sel = 0

    fig = build_quarterly_figure_two_traces(
        df=df_agg,
        selected_hex=selected_hex,
        permit_type=permit_type,
        hex_geojson=hex_geojson,
        cmin_base=0,
        cmax_base=cmax_all,
        cmin_selected=0,
        cmax_selected=cmax_sel,
        start_idx=start_idx,
        end_idx=end_idx,
        current_idx=current_idx
    )

    # Default drag mode
    fig.update_layout(dragmode='select')
    # Add a title that uses the permit_label and date range
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
    # remove the title from the px.line creation or update_layout call
    start_idx   = global_filter.get("startQuarterIndex", 0)
    end_idx     = global_filter.get("endQuarterIndex", len(quarters) - 1)
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
        # Remove the title here – we'll set it separately.
        template="plotly_white",
        markers=True
    )
    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title=get_permit_label(permit_type),
        shapes=[  # shaded areas and current time line
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
```

## src\config.py
```py
PROCESSED_DATA_PATH = "data/processed"
ANIMATION_INTERVAL = 1000

```

## src\data_utils.py
```py
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

# Expose these variables for use in callbacks and layout:
__all__ = [
    "hex_geojson", "permit_counts_wide", "quarters", "quarter_to_index",
    "permit_options", "permit_type_list", "global_color_scales"
]

```

## src\debug.py
```py
from dash import html, Output, Input
from src.app_instance import app
import logging

logger = logging.getLogger(__name__)

# Add this to your layout somewhere
debug_div = html.Div([
    html.Pre(id='debug-output', style={'whiteSpace': 'pre-wrap'}),
], style={'display': 'block'})  # Set to 'block' to see debug output

@app.callback(
    Output('debug-output', 'children'),
    [Input('global_filter', 'data'),
     Input('map-quarterly', 'figure'),
     Input('map-aggregated', 'figure')]
)
def debug_callback(global_filter, quarterly_fig, aggregated_fig):
    logger.info("Global Filter State: %s", global_filter)
    logger.info("Quarterly Figure Keys: %s", quarterly_fig.keys() if quarterly_fig else None)
    logger.info("Aggregated Figure Keys: %s", aggregated_fig.keys() if aggregated_fig else None)
    
    debug_info = f"""
    Global Filter: {global_filter}
    Quarterly Figure Present: {bool(quarterly_fig)}
    Aggregated Figure Present: {bool(aggregated_fig)}
    """
    return debug_info
```

## src\layout.py
```py
# src/layout.py

import dash_bootstrap_components as dbc
from dash import dcc, html
from src.config import ANIMATION_INTERVAL
from src.data_utils import quarters, permit_options
from src.debug import debug_div

layout = dbc.Container(
    fluid=True,
    children=[
        # Header
        dbc.Row(
            dbc.Col(
                html.H1("NYC Permits Dashboard", className="text-center my-3"),
                width=12
            )
        ),

        # Top Row: Time-Series with Title
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H3(id='time-series-title', className="text-center mb-2"),
                    dcc.Graph(id='time-series', style={'width': '100%', 'height': '400px'})
                ]),
                width=8
            ),
            # Controls column (same as before)
            dbc.Col([
                html.Div([
                    html.H4("Controls"),
                    html.Hr(),
                    # Permit type radios
                    html.Div([
                        html.Label("Permit Type"),
                        dcc.RadioItems(
                            id='permit-type',
                            options=permit_options,
                            value='NB',
                            labelStyle={'display': 'block', 'margin': '5px 0'}
                        ),
                    ], className="mb-3"),
                    # Animation Speed and Time Range Slider
                    html.Div([
                        html.Label("Animation Speed:"),
                        dcc.Dropdown(
                            id='speed-dropdown',
                            options=[
                                {'label': 'Slow', 'value': 1000},
                                {'label': 'Medium', 'value': 500},
                                {'label': 'Fast', 'value': 250}
                            ],
                            value=500,
                            clearable=False,
                            style={'width': '100%'}
                        ),
                        html.Div([
                            html.Label("Select Time Range:"),
                            dcc.RangeSlider(
                                id='period-range-slider',
                                min=0,
                                max=len(quarters) - 1,
                                value=[0, len(quarters) - 1],
                                step=1,
                                marks={
                                    i: quarters[i]
                                    for i in range(0, len(quarters), max(1, len(quarters)//8))
                                },
                                tooltip={"placement": "bottom"}
                            ),
                        ], className="mb-3"),
                        # Play/Pause/Clear Buttons
                        html.Div([
                            html.Button("▶️ Play", id='play-button', n_clicks=0,
                                        className="btn btn-secondary mb-2"),
                            html.Button("⏸ Pause", id='pause-button', n_clicks=0,
                                        className="btn btn-secondary mb-2"),
                            html.Button("🗓 Clear Time Range", id='clear-time-range', n_clicks=0,
                                        className="btn btn-secondary mb-2"),
                            html.Button("🗑️ Clear Hexes", id='clear-hexes', n_clicks=0,
                                        className="btn btn-secondary mb-2"),
                        ], className="mb-3")
                    ]),
                ], className="p-3 bg-light rounded")
            ], width=4)
        ], className="my-3"),

        # Bottom Row: Titles and Maps
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H3(id='map-aggregated-title', className="text-center mb-2"),
                    dcc.Graph(
                        id='map-aggregated',
                        figure={},
                        style={'width': '100%', 'height': '500px'},
                        config={
                            'modeBarButtonsToAdd': ['lasso2d', 'select2d'],
                            'displaylogo': False,
                        }
                    )
                ]),
                width=6
            ),
            dbc.Col(
                html.Div([
                    html.H3(id='map-quarterly-title', className="text-center mb-2"),
                    dcc.Graph(
                        id='map-quarterly',
                        style={'width': '100%', 'height': '500px'},
                        config={
                            "staticPlot": True,
                            "scrollZoom": False
                        }
                    )
                ]),
                width=6
            )
        ], className="my-3"),

        # Footer
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.Hr(),
                    html.P([
                        "Created by ",
                        html.A("David Leather", href="https://daveleather.com", target="_blank"),
                        ". Data from NYC Department of Buildings."
                    ], className="text-center")
                ]),
                width=12
            )
        ),

        # Hidden dcc.Store + Interval + Debug
        dcc.Store(
            id='global_filter',
            data={
                "startQuarterIndex": 0,
                "endQuarterIndex": len(quarters) - 1,
                "currentQuarterIndex": 0,
                "permitType": "NB",
                "play": False,
                "speed": 1000,
                "selectedHexes": []
            }
        ),
        dcc.Interval(
            id='animation-interval',
            interval=ANIMATION_INTERVAL,
            n_intervals=0,
            disabled=True
        ),
        debug_div
    ],
    className="p-4"
)

```

