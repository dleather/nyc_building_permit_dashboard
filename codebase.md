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
/* assets/custom.css */

/* Target the entire dropdown menu container and its descendants */
div[role="listbox"],
div[role="listbox"] * {
    background-color: rgb(20, 20, 20) !important;
    color: white !important;
}

/* Target each option and its inner content */
div[role="option"],
div[role="option"] * {
    background-color: rgb(20, 20, 20) !important;
    color: white !important;
}

/* Change background on hover for options */
div[role="option"]:hover,
div[role="option"]:hover * {
    background-color: rgb(50, 50, 50) !important;
    color: white !important;
}

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
/* assets/custom.css */

/* Target the entire dropdown menu container and its descendants */
div[role="listbox"],
div[role="listbox"] * {
    background-color: rgb(20, 20, 20) !important;
    color: white !important;
}

/* Target each option and its inner content */
div[role="option"],
div[role="option"] * {
    background-color: rgb(20, 20, 20) !important;
    color: white !important;
}

/* Change background on hover for options */
div[role="option"]:hover,
div[role="option"]:hover * {
    background-color: rgb(50, 50, 50) !important;
    color: white !important;
}

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
    "python-dotenv>=1.0.1",
    "shapely>=2.0.7",
    "uvicorn>=0.34.0",
]

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
    "https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"  # Add Font Awesome
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
    Input("global_filter", "data"),
    Input("map_view_store", "data")
)
def update_quarterly_map(global_filter, map_view):
    permit_type = global_filter.get("permitType", "NB")
    current_idx = global_filter.get("currentQuarterIndex", 0)
    start_idx = global_filter["startQuarterIndex"]
    end_idx = global_filter["endQuarterIndex"]
    selected_hex = global_filter.get("selectedHexes", [])
    
    quarter_label = quarters[current_idx]
    
    # Get all data for the current quarter
    df_all = permit_counts_wide.loc[
        permit_counts_wide["period"] == quarter_label
    ].copy()
    
    # When not in reset mode and the user has made a custom selection,
    # exclude the selected hexes from the base layer so they're not drawn twice.
    if not global_filter.get("resetMaps", False) and selected_hex:
        df_base = df_all[~df_all["h3_index"].isin(selected_hex)].copy()
        df_top = df_all[df_all["h3_index"].isin(selected_hex)].copy()
    else:
        # Otherwise, both layers show the full data.
        df_base = df_all.copy()
        df_top = df_all.copy()
    
    # Use the global max value within the selected time range (and for the given permit type)
    start_label = quarters[start_idx]
    end_label = quarters[end_idx]
    global_max = get_subrange_singlequarter_99(permit_type, start_label, end_label)
    use_log = global_max > 100
    
    cmin_base = 0
    cmax_base = global_max
    
    # For non-reset selection, use the same scale;
    # otherwise, the top layer shares scale with the base layer when resetMaps is True.
    cmin_top = 0
    cmax_top = cmax_base if global_filter.get("resetMaps", False) else global_max
    logger.info("Currently selectedHexes in quarterly map: %s", selected_hex)
    logger.info("df_top has hexes in quarterly map: %s", df_top["h3_index"].tolist())
    fig = build_two_trace_mapbox(
        df_base=df_base,
        df_top=df_top,
        permit_type=permit_type,
        cmin_base=cmin_base,
        cmax_base=cmax_base,
        cmin_top=cmin_top,
        cmax_top=cmax_top,
        map_title="Quarterly View",
        use_log_base=use_log,
        use_log_top=use_log
    )
    
    if not map_view:
        map_view = {
            "center": {"lat": 40.7, "lon": -73.9},
            "zoom": 10,
            "bearing": 0,
            "pitch": 0
        }
    
    fig.update_layout(
        mapbox=dict(
            style=MAPBOX_STYLE,
            accesstoken=mapbox_token,
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
    Input("global_filter", "data"),
    Input("map_view_store", "data")
)
def update_aggregated_map(global_filter, map_view):
    permit_type = global_filter.get("permitType", "NB")
    start_idx = global_filter.get("startQuarterIndex", 0)
    end_idx = global_filter.get("endQuarterIndex", len(quarters) - 1)
    selected_hex = global_filter.get("selectedHexes", [])
    reset_maps = global_filter.get("resetMaps", False)  # Check for reset flag

    start_label = quarters[start_idx]
    end_label   = quarters[end_idx]
    
    # 1) Build the base DataFrame (aggregating over the selected time range)
    df_sub = permit_counts_wide.loc[
        (permit_counts_wide["period"] >= start_label) &
        (permit_counts_wide["period"] <= end_label)
    ].copy()
    
    if df_sub.empty:
        return px.choropleth_mapbox()
    
    df_agg = df_sub.groupby("h3_index", as_index=False)[permit_type].sum()
    
    cmin_base = 0
    cmax_base = global_agg_99[permit_type]
    use_log = global_agg_99[permit_type] > 100
    
    # If there is a non-reset custom selection, exclude selected hexes from the base layer.
    if not reset_maps and selected_hex:
        df_base = df_agg[~df_agg["h3_index"].isin(selected_hex)].copy()
        df_top = df_agg[df_agg["h3_index"].isin(selected_hex)].copy()
    else:
        df_base = df_agg.copy()
        df_top = df_agg.copy()
    
    logger.info("df_top length: %s", len(df_top))
    
    cmin_top = 0
    if reset_maps:
        cmax_top = cmax_base
    else:
        cmax_top = df_top[permit_type].max() if not df_top.empty else 0
    logger.info("Currently selectedHexes in aggregated map: %s", selected_hex)
    logger.info("df_top has hexes in aggregated map: %s", df_top["h3_index"].tolist())
    # 3) Build the figure using our helper function
    fig = build_two_trace_mapbox(
        df_base=df_base,
        df_top=df_top,
        permit_type=permit_type,
        cmin_base=cmin_base,
        cmax_base=cmax_base,
        cmin_top=cmin_top,
        cmax_top=cmax_top,
        map_title="Aggregated View",
        use_log_base=use_log,
        use_log_top=use_log
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
            style=MAPBOX_STYLE,
            accesstoken=mapbox_token,
            center=map_view.get("center"),
            uirevision="synced-maps",  # maintains the same ui revision across re-renders
            bearing=map_view.get("bearing"),
            pitch=map_view.get("pitch")
        ),
        uirevision="synced-maps"  # maintains the same UI revision across re-renders
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
    [
        State("global_filter", "data"),
    ],
    prevent_initial_call=True
)
@log_callback
def update_selected_hexes(qtr_sel, agg_sel, clear_n_clicks, global_filter):
    logger.info("In callback update_selected_hexes, final new_filter=%s", global_filter)
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
    all_points = (selected_data or {}).get("points", [])

    # Filter out top-layer points:
    base_points = [p for p in all_points if p["curveNumber"] == 0]
    new_sel = set(p["location"] for p in base_points)

    new_filter["selectedHexes"] = list(new_sel)

    logger.info("update_selected_hexes new_sel after filtering: %s", new_sel)   
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
```

## src\config.py
```py
PROCESSED_DATA_PATH = "data/processed"
ANIMATION_INTERVAL = 1000
MAPBOX_STYLE = "carto-darkmatter"
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

    # Update the layout with darker theme and larger ticks
    fig.update_layout(
        plot_bgcolor='rgb(12, 12, 12)',
        paper_bgcolor='rgb(12, 12, 12)',
        font=dict(
            color='rgba(255, 255, 255, 0.7)'
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.08)',
            tickfont=dict(
                color='rgba(255, 255, 255, 0.7)',
                size=14  # Increased font size
            ),
            tickangle=45,  # Rotate labels 45 degrees
            dtick=4,  # Show every 4th tick (changed from 2)
            zeroline=False,
            gridwidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.08)',
            tickfont=dict(
                color='rgba(255, 255, 255, 0.7)',
                size=14  # Increased font size
            ),
            zeroline=False,
            gridwidth=1
        ),
        margin=dict(l=40, r=40, t=40, b=80),  # Increased bottom margin for rotated labels
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
            fillcolor='rgba(255, 255, 255, 0.05)',
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
    style={
        'backgroundColor': 'rgb(10, 10, 10)',
        'color': 'rgba(255, 255, 255, 0.8)',
        'minHeight': '100vh'
    },
    children=[
        # Header
        dbc.Row(
            dbc.Col(
                html.H1("NYC Permits Dashboard", 
                    className="text-center my-3",
                    style={'color': 'rgba(255, 255, 255, 0.9)'}
                ),
                width=12
            )
        ),

        # Top Row: Time-Series with Title
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H3(id='time-series-title', 
                        className="text-center mb-2",
                        style={'color': 'rgba(255, 255, 255, 0.8)'}
                    ),
                    dcc.Graph(id='time-series', style={'width': '100%', 'height': '400px'})
                ]),
                width=8
            ),
            # Controls column
            dbc.Col(
                html.Div([
                    html.H4("Controls", style={'color': 'rgba(255, 255, 255, 0.8)'}),
                    html.Hr(style={'borderColor': 'rgba(255, 255, 255, 0.2)'}),
                    dbc.Row([
                        dbc.Col(
                            html.Div([
                                html.Label("Permit Type", style={'color': 'rgba(255, 255, 255, 0.7)'}),
                                dcc.RadioItems(
                                    id="permit-type",
                                    options=permit_options,
                                    value="NB",
                                    className="mb-3",
                                    style={'color': 'rgba(255, 255, 255, 0.7)'},
                                    labelStyle={'marginRight': '10px'}
                                )
                            ]),
                            width=6
                        ),
                        dbc.Col(
                            html.Div([
                                html.Label("Animation Speed", style={'color': 'rgba(255, 255, 255, 0.7)'}),
                                dcc.RadioItems(
                                    id="speed-radio",
                                    options=[
                                        {"label": "Slow", "value": 2000},
                                        {"label": "Medium", "value": 1000},
                                        {"label": "Fast", "value": 500}
                                    ],
                                    value=1000,
                                    className="mb-3",
                                    style={'color': 'rgba(255, 255, 255, 0.7)'},
                                    labelStyle={'marginRight': '10px'}
                                )
                            ]),
                            width=6
                        )
                    ], className="mb-3"),
                    html.Label("Select Time Range:", style={'color': 'rgba(255, 255, 255, 0.7)'}),
                    dcc.RangeSlider(
                        id="period-range-slider",
                        min=0,
                        max=len(quarters) - 1,
                        value=[0, len(quarters) - 1],
                        marks={i: quarters[i] for i in range(0, len(quarters), 16)},
                        className="mb-3"
                    ),
                    html.Div([
                        dbc.Button([
                            html.I(className="fas fa-play me-1", style={"color": "black"}),
                            html.Span("Play", style={"color": "black"})
                        ], 
                            id="play-button", 
                            color="success",
                            className="me-2"
                        ),
                        dbc.Button([
                            html.I(className="fas fa-pause me-1", style={"color": "black"}),
                            html.Span("Pause", style={"color": "black"})
                        ], 
                            id="pause-button", 
                            color="warning",
                            className="me-2"
                        ),
                        dbc.Button([
                            html.I(className="fas fa-clock me-1"),
                            "Clear Time Range"
                        ], 
                            id="clear-time-range", 
                            color="secondary",
                            className="me-2"
                        ),
                        dbc.Button([
                            html.I(className="fas fa-eraser me-1"),
                            "Clear Hexes"
                        ], 
                            id="clear-hexes", 
                            color="danger",
                        ),
                    ], className="d-flex justify-content-start gap-2 mb-3")
                ], style={
                    'backgroundColor': 'rgb(15, 15, 15)',
                    'borderRadius': '5px',
                    'padding': '20px',
                    'height': '100%'
                }),
                width=4
            )
        ]),

        # Bottom Row: Maps
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H3(id='map-aggregated-title', 
                        className="text-center mb-2",
                        style={'color': 'rgba(255, 255, 255, 0.8)'}
                    ),
                    dcc.Graph(id='map-aggregated')
                ]),
                width=6
            ),
            dbc.Col(
                html.Div([
                    html.H3(id='map-quarterly-title', 
                        className="text-center mb-2",
                        style={'color': 'rgba(255, 255, 255, 0.8)'}
                    ),
                    dcc.Graph(id='map-quarterly')
                ]),
                width=6
            )
        ], className="my-3"),

        # Footer
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.Hr(style={'borderColor': 'rgba(255, 255, 255, 0.2)'}),
                    html.P([
                        "Created by ",
                        html.A("David Leather", 
                            href="https://daveleather.com",
                            target="_blank",
                            style={'color': 'rgba(255, 99, 132, 0.8)'}
                        ),
                        ". Data from NYC Department of Buildings."
                    ], className="text-center", style={'color': 'rgba(255, 255, 255, 0.6)'})
                ]),
                width=12
            )
        ),

        # Hidden dcc.Store for Map View (holds current map center and zoom)
        dcc.Store(
            id='map_view_store',
            data={
                'center': {'lat': 40.7, 'lon': -73.9},
                'zoom': 9,
                'bearing': 0,
                'pitch': 0
            }
        ),

        # Existing Stores, Interval, and Debug Div
        dcc.Store(
            id='global_filter',
            data={
                "startQuarterIndex": 0,
                "endQuarterIndex": len(quarters) - 1,
                "currentQuarterIndex": 0,
                "permitType": "NB",
                "play": False,
                "speed": 1000,
                "selectedHexes": [],
                "resetMaps": False
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

