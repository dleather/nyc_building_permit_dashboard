# src/app.py

import dash
import dash_bootstrap_components as dbc
import uvicorn

from src.layout import layout
from src import callbacks  # Make sure callbacks get registered

# Create the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX],  # You can switch themes easily
)

# Attach the layout
app.layout = layout

# Expose the underlying Flask server for Uvicorn
server = app.server

if __name__ == "__main__":
    # If someone runs `python src/app.py` directly, we'll start Uvicorn on port 8000
    uvicorn.run("src.app:server", host="127.0.0.1", port=8000, reload=True)