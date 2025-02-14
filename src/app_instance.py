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