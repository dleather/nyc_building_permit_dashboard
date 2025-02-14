# src/layout.py

import dash_bootstrap_components as dbc
from dash import dcc, html
from src.config import ANIMATION_INTERVAL
from src.data_utils import quarters, permit_options

layout = dbc.Container(
    fluid=True,
    children=[
        # Header
        dbc.Row(
            dbc.Col(
                html.H1("NYC Permits Dashboard"), width=12
            )
        ),
        # Controls: Permit type, play/pause, clear time etc.
        dbc.Row([
            dbc.Col([
                html.Div([
                    # Permit type radios:
                    dcc.RadioItems(
                        id='permit-type',
                        options=permit_options,
                        value='NB',
                        labelStyle={'display': 'block', 'margin': '5px 0'}
                    ),
                ], style={'width': '250px'}),
            ]),
            dbc.Col([
                html.Div([
                    html.Button("▶️ Play", id='play-button', n_clicks=0,
                                style={'padding': '5px 15px', 'marginBottom': '10px'}),
                    html.Button("Clear Time Range", id='clear-time-range', n_clicks=0,
                                style={'padding': '5px 15px', 'marginBottom': '10px'}),
                    html.Button("Clear Hexes", id='clear-hexes', n_clicks=0,
                                style={'padding': '5px 15px', 'marginBottom': '10px'}),
                    html.Label("Animation Speed:", style={'marginTop': '15px'}),
                    dcc.Dropdown(
                        id='speed-dropdown',
                        options=[
                            {'label': 'Slow', 'value': 2000},
                            {'label': 'Medium', 'value': 1000},
                            {'label': 'Fast', 'value': 500}
                        ],
                        value=1000,
                        clearable=False,
                        style={'width': '120px', 'marginTop': '5px'}
                    ),
                ], style={'display': 'flex', 'flexDirection': 'column', 'margin': '0 20px'}),
            ]),
            # Hidden Quarter slider for internal use:
            dbc.Col([
                dcc.Slider(
                    id='period-slider',
                    min=0,
                    max=len(quarters) - 1,
                    value=0,
                    marks={i: quarters[i] for i in range(0, len(quarters), max(1, len(quarters) // 8))},
                    step=None,
                    tooltip={"placement": "bottom"}
                )
            ], style={'flex': '1', 'display': 'none'}),
        ], className="my-3"),
        # Time-series graph:
        dbc.Row(
            dbc.Col(
                dcc.Graph(id='time-series', style={'width': '100%', 'height': '400px'}),
                width=12
            )
        ),
        # Map graph:
        dbc.Row(
            dbc.Col(
                dcc.Graph(id='hex-map', style={'width': '100%', 'height': '600px'}),
                width=12
            )
        ),
        # Footer:
        dbc.Row(
            dbc.Col([
                html.Hr(),
                html.P("Created by Me. Data from NYC Open Data, etc.")
            ])
        ),
        # Hidden stores and interval:
        dcc.Store(id='selected-hexes', data=None),
        dcc.Store(id='selected-time-range', data=None),
        dcc.Interval(
            id='play-interval',
            interval=ANIMATION_INTERVAL,
            n_intervals=0,
            disabled=True
        )
    ]
)
