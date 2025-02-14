import dash
import dash_bootstrap_components as dbc
import uvicorn

from src.config import *
from src.layout import layout
from src import callbacks  # This registers all the callbacks

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX],
)
app.layout = layout

# Expose the underlying Flask server for Uvicorn
server = app.server

if __name__ == "__main__":
    # Run Uvicorn on port 8000
    uvicorn.run("src.app:server", host="127.0.0.1", port=8000, reload=True) 