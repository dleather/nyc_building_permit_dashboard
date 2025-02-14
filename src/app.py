from dash import Dash
import dash_bootstrap_components as dbc
from asgiref.wsgi import WsgiToAsgi

# Initialize the Dash app
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX],
    suppress_callback_exceptions=True
)
server = app.server  # Expose server for production deployment

# Import and set the layout
from src.layout import layout
app.layout = layout

# Import callbacks (this registers them)
from src import callbacks

# Create ASGI app from WSGI app
asgi_app = WsgiToAsgi(server)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:asgi_app", host="127.0.0.1", port=8000, reload=True) 