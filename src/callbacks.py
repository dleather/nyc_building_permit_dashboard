# src/callbacks.py

from dash.dependencies import Input, Output
from src.app import app
import plotly.express as px
import pandas as pd

# Dummy data
df_time = pd.DataFrame({
    "period": [f"Q{i}" for i in range(1, 13)],
    "permits": [i**2 for i in range(1, 13)]
})

@app.callback(
    Output("time-series-graph", "figure"),
    Input("permit-dropdown", "value")
)
def update_time_series(permit_value):
    fig = px.line(
        df_time,
        x="period",
        y="permits",
        title=f"Permits Over Time ({permit_value})"
    )
    return fig
