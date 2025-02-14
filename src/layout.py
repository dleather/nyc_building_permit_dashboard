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

        # Top Row: Time-Series with Title
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H3(id='time-series-title', className="text-center mb-2"),
                    dcc.Graph(id='time-series', style={'width': '100%', 'height': '400px'})
                ]),
                width=8
            ),
            # Controls column (same as before)
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
                    # Animation Speed and Time Range Slider
                    html.Div([
                        html.Label("Animation Speed:"),
                        dcc.Dropdown(
                            id='speed-dropdown',
                            options=[
                                {'label': 'Slow', 'value': 1000},
                                {'label': 'Medium', 'value': 500},
                                {'label': 'Fast', 'value': 250}
                            ],
                            value=500,
                            clearable=False,
                            style={'width': '100%'}
                        ),
                        html.Div([
                            html.Label("Select Time Range:"),
                            dcc.RangeSlider(
                                id='period-range-slider',
                                min=0,
                                max=len(quarters) - 1,
                                value=[0, len(quarters) - 1],
                                step=1,
                                marks={
                                    i: quarters[i]
                                    for i in range(0, len(quarters), max(1, len(quarters)//8))
                                },
                                tooltip={"placement": "bottom"}
                            ),
                        ], className="mb-3"),
                        # Play/Pause/Clear Buttons
                        html.Div([
                            html.Button("▶️ Play", id='play-button', n_clicks=0,
                                        className="btn btn-secondary mb-2"),
                            html.Button("⏸ Pause", id='pause-button', n_clicks=0,
                                        className="btn btn-secondary mb-2"),
                            html.Button("🗓 Clear Time Range", id='clear-time-range', n_clicks=0,
                                        className="btn btn-secondary mb-2"),
                            html.Button("🗑️ Clear Hexes", id='clear-hexes', n_clicks=0,
                                        className="btn btn-secondary mb-2"),
                        ], className="mb-3")
                    ]),
                ], className="p-3 bg-light rounded")
            ], width=4)
        ], className="my-3"),

        # Bottom Row: Titles and Maps
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H3(id='map-aggregated-title', className="text-center mb-2"),
                    dcc.Graph(
                        id='map-aggregated',
                        figure={},
                        style={'width': '100%', 'height': '500px'},
                        config={
                            'modeBarButtonsToAdd': ['lasso2d', 'select2d'],
                            'displaylogo': False,
                        }
                    )
                ]),
                width=6
            ),
            dbc.Col(
                html.Div([
                    html.H3(id='map-quarterly-title', className="text-center mb-2"),
                    dcc.Graph(
                        id='map-quarterly',
                        style={'width': '100%', 'height': '500px'},
                        config={
                            "staticPlot": True,
                            "scrollZoom": False
                        }
                    )
                ]),
                width=6
            )
        ], className="my-3"),

        # Footer
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.Hr(),
                    html.P([
                        "Created by ",
                        html.A("David Leather", href="https://daveleather.com", target="_blank"),
                        ". Data from NYC Department of Buildings."
                    ], className="text-center")
                ]),
                width=12
            )
        ),

        # Hidden dcc.Store + Interval + Debug
        dcc.Store(
            id='global_filter',
            data={
                "startQuarterIndex": 0,
                "endQuarterIndex": len(quarters) - 1,
                "currentQuarterIndex": 0,
                "permitType": "NB",
                "play": False,
                "speed": 1000,
                "selectedHexes": []
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
