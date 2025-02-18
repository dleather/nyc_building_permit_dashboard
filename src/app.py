from dash import Dash
import dash_bootstrap_components as dbc
from asgiref.wsgi import WsgiToAsgi
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

logger = logging.getLogger(__name__)
logger.info("Starting application")

from src.app_instance import app, server
from src.layout import layout
app.layout = layout
from src import callbacks

asgi_app = WsgiToAsgi(server)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:asgi_app", host="0.0.0.0", port=8000)