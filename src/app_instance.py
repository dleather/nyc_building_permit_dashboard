# Import required Dash libraries
from dash import Dash
import dash_bootstrap_components as dbc

# Load external CSS stylesheets for UI components and styling
# Using Flatly theme for clean, modern look
# Bootstrap 5.2.3 for responsive layout and components
# Font Awesome 6.0 for icons
external_stylesheets = [
    dbc.themes.FLATLY,  # Nice clean theme that works well with dark mode
    "https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
]

# Initialize the Dash app with our configuration
app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,  # Needed for dynamic callbacks
    meta_tags=[
        # Ensure proper scaling on mobile devices
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

# Get the Flask server instance for WSGI/ASGI deployment
server = app.server