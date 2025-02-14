# src/callbacks.py

from dash.dependencies import Input, Output, State
from src.app import app
import plotly.express as px
import pandas as pd
import dash
import numpy as np

# Dummy data
df_time = pd.DataFrame({
    "period": [f"Q{i}" for i in range(1, 13)],
    "permits": [i**2 for i in range(1, 13)]
})

# Import the app, layout, and data utilities:
from src.data_utils import (
    hex_geojson, permit_counts_wide, quarters, quarter_to_index,
    global_color_scales
)

# --------------- Helper functions ---------------

def update_map(period_idx, permit_type, selectedData):
    current_period = quarters[period_idx]
    df_period = permit_counts_wide[permit_counts_wide['period'] == current_period]
    global_min, global_max = global_color_scales[permit_type]

    fig = px.choropleth_map(
        df_period,
        geojson=hex_geojson,
        locations='h3_index',
        color=permit_type,
        featureidkey="properties.h3_index",
        map_style='basic',
        center={"lat": 40.7128, "lon": -73.9660},
        zoom=10,
        opacity=0.6,
        color_continuous_scale="Reds",
        range_color=[global_min, global_max],
    )
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        title=f"Map for {permit_type} during {current_period}",
    )
    fig.update_geos(fitbounds="locations", visible=False)

    if selectedData == "ALL_HEXES":
        all_indices = list(range(len(df_period)))
        fig.update_traces(selectedpoints=all_indices, selector=dict(type='choropleth'))
    elif selectedData and isinstance(selectedData, dict):
        selected_hex_ids = [pt['location'] for pt in selectedData['points']]
        new_selected = [i for i, h3id in enumerate(df_period['h3_index']) if h3id in selected_hex_ids]
        fig.update_traces(selectedpoints=new_selected, selector=dict(type='choropleth'))
    else:
        fig.update_traces(selectedpoints=[], selector=dict(type='choropleth'))
    return fig

def create_map_figure_for_range(x0, x1, permit_type):
    start_i = max(0, int(round(x0)))
    end_i   = min(len(quarters)-1, int(round(x1)))
    selected_quarters = quarters[start_i:end_i+1]
    df_range = permit_counts_wide[permit_counts_wide['period'].isin(selected_quarters)]
    agg_df = df_range.groupby('h3_index', as_index=False)[permit_type].sum()

    if agg_df.empty:
        clip_val = 0
        min_val = 0
    else:
        clip_val = np.percentile(agg_df[permit_type], 99)
        min_val = agg_df[permit_type].min()

    fig = px.choropleth_map(
        agg_df,
        geojson=hex_geojson,
        locations='h3_index',
        color=permit_type,
        featureidkey="properties.h3_index",
        center={"lat": 40.7128, "lon": -73.9660},
        map_style="basic",
        zoom=10,
        opacity=0.6,
        color_continuous_scale="Reds",
        range_color=[min_val, clip_val],
    )
    title_text = f"Map for {permit_type} aggregated from {quarters[start_i]} to {quarters[end_i]}"
    fig.update_layout(
        title=title_text,
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    fig.update_geos(visible=False)
    return fig

# --------------- Define Callbacks ---------------

# 1) Map callbacks
@app.callback(
    [Output('selected-hexes', 'data'),
     Output('hex-map', 'selectedData'),
     Output('hex-map', 'figure')],
    [Input('hex-map', 'selectedData'),
     Input('clear-hexes', 'n_clicks'),
     Input('period-slider', 'value'),
     Input('permit-type', 'value'),
     Input('selected-time-range', 'data')],
    [State('hex-map', 'figure')]
)
def handle_map_updates(selectedData, clear_hexes_click, period_idx, permit_type, time_range, current_figure):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if triggered_id == 'clear-hexes':
        if time_range:
            x0, x1 = time_range
            map_fig = create_map_figure_for_range(x0, x1, permit_type)
        else:
            map_fig = update_map(period_idx, permit_type, "ALL_HEXES")
        return "ALL_HEXES", None, map_fig

    if triggered_id in ['period-slider', 'permit-type', 'selected-time-range']:
        if time_range:
            x0, x1 = time_range
            map_fig = create_map_figure_for_range(x0, x1, permit_type)
        else:
            map_fig = update_map(period_idx, permit_type, selectedData)
        return dash.no_update, dash.no_update, map_fig

    if triggered_id == 'hex-map':
        return selectedData, dash.no_update, dash.no_update

    return dash.no_update, dash.no_update, dash.no_update

# 2) Time-series callback
@app.callback(
    Output('time-series', 'figure'),
    [Input('selected-hexes', 'data'),
     Input('permit-type', 'value')]
)
def update_time_series(selected_hex_data, permit_type):
    permit_labels = {
        "total_permit_count": "All Permits",
        "A1": "Type I - Major Alteration",
        "A2": "Type II - Minor Alteration",
        "A3": "Type III - Minor Alteration",
        "DM": "Demolition",
        "NB": "New Building"
    }
    title_label = permit_labels.get(permit_type, permit_type)

    if selected_hex_data == "ALL_HEXES" or not selected_hex_data:
        filtered_df = permit_counts_wide.copy()
        title_suffix = "(All Areas)"
    else:
        selected_hexes = [pt['location'] for pt in selected_hex_data['points']]
        filtered_df = permit_counts_wide[permit_counts_wide['h3_index'].isin(selected_hexes)]
        title_suffix = f"({len(selected_hexes)} Selected Areas)"

    ts = filtered_df.groupby('period')[permit_type].sum().reset_index()
    ts['period_idx'] = ts['period'].map(quarter_to_index)
    fig = px.line(
        ts,
        x='period_idx',
        y=permit_type,
        markers=True,
        template="plotly_white",
        title=f"{title_label} Over Time {title_suffix}"
    )
    fig.update_layout(
        xaxis=dict(
            title='Time Period',
            tickmode='array',
            tickvals=list(range(len(quarters)))[::4],
            ticktext=[quarters[i] for i in range(0, len(quarters), 4)],
            tickangle=45,
            range=[0, len(quarters)-1],
            rangeslider=dict(visible=True),
        ),
        yaxis=dict(
            title='Permit Count'
        ),
        margin=dict(l=50, r=20, t=60, b=50),
        dragmode='select',
        selectdirection='h',
        uirevision='time-series'
    )
    fig.update_traces(marker=dict(size=6), line=dict(width=2))
    return fig

# 3) Time-range selection and pause
@app.callback(
    [Output('selected-time-range', 'data'),
     Output('play-interval', 'disabled', allow_duplicate=True)],
    [Input('time-series', 'selectedData'),
     Input('clear-time-range', 'n_clicks')],
    prevent_initial_call="initial_duplicate"
)
def handle_time_range_and_clear(ts_selected, clear_time_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == "clear-time-range":
        return None, False
    elif triggered_id == "time-series":
        if ts_selected and "range" in ts_selected and "x" in ts_selected["range"]:
            x0, x1 = ts_selected["range"]["x"]
            return [x0, x1], True
        elif ts_selected and "points" in ts_selected and ts_selected["points"]:
            pts = [pt["x"] for pt in ts_selected["points"]]
            return [min(pts), max(pts)], True
    return dash.no_update, dash.no_update

# 4) Play/Pause toggle
@app.callback(
    [Output('play-interval', 'disabled', allow_duplicate=True),
     Output('play-button', 'children')],
    [Input('play-button', 'n_clicks')],
    [State('play-interval', 'disabled')],
    prevent_initial_call="initial_duplicate"
)
def toggle_play(n_clicks, current_disabled):
    if n_clicks is None or n_clicks == 0:
        return current_disabled, "▶️ Play"
    new_disabled = not current_disabled
    new_label = "⏸️ Pause" if not new_disabled else "▶️ Play"
    return new_disabled, new_label

# 5) Update animation speed from dropdown
@app.callback(
    Output('play-interval', 'interval'),
    Input('speed-dropdown', 'value')
)
def update_speed(value):
    return value

# 6) Auto-advance slider
@app.callback(
    Output('period-slider', 'value', allow_duplicate=True),
    Input('play-interval', 'n_intervals'),
    State('period-slider', 'value'),
    prevent_initial_call="initial_duplicate"
)
def update_slider(n_intervals, current_value):
    return (current_value + 1) % len(quarters)

# 7) Final step: update TS and map together
@app.callback(
    [Output('time-series', 'figure', allow_duplicate=True),
     Output('hex-map', 'figure', allow_duplicate=True)],
    [Input('period-slider', 'value'),
     Input('selected-time-range', 'data'),
     Input('play-interval', 'n_intervals')],
    [State('time-series', 'figure'),
     State('permit-type', 'value'),
     State('selected-hexes', 'data')],
    prevent_initial_call='initial_duplicate'
)
def update_ts_and_map(slider_value, time_range, _n, ts_fig_state, permit_type, selected_hex_data):
    ctx = dash.callback_context
    fig_ts = ts_fig_state if ts_fig_state else {}
    fig_ts.setdefault("layout", {})
    fig_ts["layout"]["shapes"] = []

    if time_range:
        x0, x1 = time_range
        fig_ts["layout"]["shapes"].append({
            "type": "rect",
            "xref": "x",
            "yref": "paper",
            "x0": x0,
            "x1": x1,
            "y0": 0,
            "y1": 1,
            "fillcolor": "LightSkyBlue",
            "opacity": 0.3,
            "layer": "below",
            "line": {"width": 0},
        })
        map_fig = create_map_figure_for_range(x0, x1, permit_type)
    else:
        fig_ts["layout"]["shapes"].append({
            "type": "line",
            "xref": "x",
            "yref": "paper",
            "x0": slider_value,
            "x1": slider_value,
            "y0": 0,
            "y1": 1,
            "line": {"color": "RoyalBlue", "dash": "dot", "width": 2},
        })
        map_fig = update_map(slider_value, permit_type, selected_hex_data)

    return fig_ts, map_fig

# 8) Sync TS range slider to the period slider
@app.callback(
    Output('period-slider', 'value', allow_duplicate=True),
    Input('time-series', 'relayoutData'),
    prevent_initial_call=True
)
def sync_with_timeseries_range(relayout_data):
    if relayout_data and 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
        x0 = float(relayout_data['xaxis.range[0]'])
        x1 = float(relayout_data['xaxis.range[1]'])
        mid = 0.5 * (x0 + x1)
        val = round(mid)
        val = max(0, min(len(quarters) - 1, val))
        return val
    return dash.no_update
