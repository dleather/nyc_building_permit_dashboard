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
                                {'label': 'Medium', 'value': 750},
                                {'label': 'Fast', 'value': 500}
                            ],
                            value=750,
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

