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

# Ensure asgi_app is defined at the module level
asgi_app = WsgiToAsgi(server)

if __name__ == "__main__":
    # app.run_server(debug=True, port=8050)  # Disable this for production
    import uvicorn
    uvicorn.run("src.app:asgi_app", host="0.0.0.0", port=8000)  # Use a production server 