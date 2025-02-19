# Main application entry point for the NYC Permits Dashboard
# Sets up logging, initializes the Dash app, and configures the ASGI server
from dash import Dash
import dash_bootstrap_components as dbc
from asgiref.wsgi import WsgiToAsgi
import logging

# Configure logging to track application events and errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    force=True
)

logger = logging.getLogger(__name__)
logger.info("Starting application")

# Import and set up the Dash application components
from src.app_instance import app, server
from src.layout import layout
app.layout = layout  # Apply the dashboard layout
from src import callbacks  # Register callback functions

# Wrap WSGI app with ASGI for better performance
asgi_app = WsgiToAsgi(server)

# Run the server if executed directly
if __name__ == "__main__":
    import uvicorn
    # Start uvicorn server on all network interfaces
    uvicorn.run("src.app:asgi_app", host="0.0.0.0", port=8000)