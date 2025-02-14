# src/layout.py

import dash_bootstrap_components as dbc
from dash import dcc, html
from src.config import ANIMATION_INTERVAL
from src.data_utils import quarters, permit_options
from src.debug import debug_div

layout = dbc.Container(
    fluid=True,
    children=[
        # Header
        dbc.Row(
            dbc.Col(
                html.H1("NYC Permits Dashboard", className="text-center my-3"),
                width=12
            )
        ),

        # Top Row: LEFT = Time Series (80%), RIGHT = Controls (20%)
        dbc.Row([
            # Time-Series (80%)
            dbc.Col(
                dcc.Graph(id='time-series', style={'width': '100%', 'height': '400px'}),
                width=8
            ),

            # Controls (20%)
            dbc.Col([
                html.Div([
                    html.H4("Controls"),
                    html.Hr(),

                    # Permit type radios
                    html.Div([
                        html.Label("Permit Type"),
                        dcc.RadioItems(
                            id='permit-type',
                            options=permit_options,
                            value='NB',
                            labelStyle={'display': 'block', 'margin': '5px 0'}
                        ),
                    ], className="mb-3"),

                    # Play/Pause / Clear Buttons
                    html.Div([
                        html.Button("▶️ Play", id='play-button', n_clicks=0,
                                    className="btn btn-primary mb-2"),
                        html.Button("⏸ Pause", id='pause-button', n_clicks=0, className="btn btn-secondary mb-2"),
                        html.Button("Clear Time Range", id='clear-time-range', n_clicks=0,
                                    className="btn btn-secondary mb-2"),
                        html.Button("Clear Hexes", id='clear-hexes', n_clicks=0,
                                    className="btn btn-secondary mb-2"),
                    ], className="mb-3"),

                    # Animation Speed
                    html.Div([
                        html.Label("Animation Speed:"),
                        dcc.Dropdown(
                            id='speed-dropdown',
                            options=[
                                {'label': 'Slow', 'value': 2000},
                                {'label': 'Medium', 'value': 1000},
                                {'label': 'Fast', 'value': 500}
                            ],
                            value=1000,
                            clearable=False,
                            style={'width': '100%'}
                        ),
                    ], className="mb-3"),

                    # Optional hidden quarter slider
                    dcc.Slider(
                        id='period-slider',
                        min=0,
                        max=len(quarters) - 1,
                        value=0,
                        marks={i: quarters[i] for i in range(
                               0, len(quarters), max(1, len(quarters) // 8))},
                        step=None,
                        tooltip={"placement": "bottom"},
                        className="d-none"
                    ),
                ], className="p-3 bg-light rounded")
            ], width=4),
        ], className="my-3"),

        # Bottom Row: LEFT = Quarter-by-quarter map, RIGHT = Aggregated map
        dbc.Row([
            dbc.Col(
                dcc.Graph(id='map-quarterly', style={'width': '100%', 'height': '500px'}),
                width=6
            ),
            dbc.Col(
                dcc.Graph(id='map-aggregated', style={'width': '100%', 'height': '500px'}),
                width=6
            )
        ], className="my-3"),

        # Footer
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.Hr(),
                    html.P("Created by Me. Data from NYC Open Data, etc.", className="text-center")
                ]),
                width=12
            )
        ),

        # Hidden Stores and Interval(s)
        dcc.Store(
            id='global_filter',
            data={
                "startQuarterIndex": 0,
                "endQuarterIndex": len(quarters) - 1,
                "currentQuarterIndex": 0,
                "permitType": "NB",
                "play": False,
                "speed": 1000
            }
        ),
        dcc.Interval(
            id='animation-interval',
            interval=ANIMATION_INTERVAL,
            n_intervals=0,
            disabled=True
        ),
        debug_div
    ],
    className="p-4"
)
