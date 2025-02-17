# src/layout.py

import dash_bootstrap_components as dbc
from dash import dcc, html
from src.config import ANIMATION_INTERVAL
from src.data_utils import quarters, permit_options
from src.debug import debug_div
import dash_core_components as dcc  # ensure dcc is imported if not already

layout = dbc.Container(
    fluid=True,
    style={
        'backgroundColor': 'rgb(10, 10, 10)',
        'color': 'rgba(255, 255, 255, 0.8)',
        'minHeight': '100vh'
    },
    children=[
        # Header
        dbc.Row(
            dbc.Col(
                html.H1("NYC Permits Dashboard", 
                    className="text-center my-3",
                    style={'color': 'rgba(255, 255, 255, 0.9)'}
                ),
                width=12
            )
        ),

        # Top Row: Time-Series with Title
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H3(id='time-series-title', 
                        className="text-center mb-2",
                        style={'color': 'rgba(255, 255, 255, 0.8)'}
                    ),
                    dcc.Graph(id='time-series', style={'width': '100%', 'height': '400px'})
                ]),
                width=8
            ),
            # Controls column
            dbc.Col(
                html.Div([
                    html.H4("Controls", style={'color': 'rgba(255, 255, 255, 0.8)'}),
                    html.Hr(style={'borderColor': 'rgba(255, 255, 255, 0.2)'}),

                    # First row: Permit Type and Animation Speed
                    dbc.Row([
                        dbc.Col([
                            html.Label("Permit Type", style={'color': 'rgba(255, 255, 255, 0.7)'}),
                            dcc.RadioItems(
                                id="permit-type",
                                options=permit_options,
                                value="NB",
                                className="mb-3",
                                style={'color': 'rgba(255, 255, 255, 0.7)'},
                                labelStyle={'marginRight': '10px'}
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Animation Speed", style={'color': 'rgba(255, 255, 255, 0.7)'}),
                            dcc.RadioItems(
                                id="speed-radio",
                                options=[
                                    {"label": "Slow", "value": 2000},
                                    {"label": "Medium", "value": 1000},
                                    {"label": "Fast", "value": 500}
                                ],
                                value=1000,
                                className="mb-3",
                                style={'color': 'rgba(255, 255, 255, 0.7)'},
                                labelStyle={'marginRight': '10px'}
                            )
                        ], width=6)
                    ], className="mb-3"),

                    # Second row: Aggregated Map Scale and Quarterly Map Scale (aligned vertically)
                    dbc.Row([
                        dbc.Col([
                            html.Label("Aggregated Map Scale", style={'color': 'rgba(255, 255, 255, 0.7)'}),
                            dcc.Dropdown(
                                id='scale-type-aggregated',
                                options=[
                                    {'label': 'Linear', 'value': 'linear'},
                                    {'label': 'Ln', 'value': 'ln'},
                                    {'label': 'Log', 'value': 'log'},
                                    {'label': 'sqrt', 'value': 'sqrt'},
                                    {'label': 'cube-root', 'value': 'cube-root'},
                                    {'label': '4th root', 'value': '4th-root'},
                                    {'label': '5th-root', 'value': '5th-root'},
                                    {'label': '6th root', 'value': '6th-root'},
                                ],
                                value='4th-root',
                                className='dark-dropdown',
                                style={
                                    'width': '100%', 
                                    'marginBottom': '10px',
                                    'color': 'black'
                                }
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Quarterly Map Scale", style={'color': 'rgba(255, 255, 255, 0.7)'}),
                            dcc.Dropdown(
                                id='scale-type-quarterly',
                                options=[
                                    {'label': 'Linear', 'value': 'linear'},
                                    {'label': 'Ln', 'value': 'ln'},
                                    {'label': 'Log', 'value': 'log'},
                                    {'label': 'sqrt', 'value': 'sqrt'},
                                    {'label': 'cube-root', 'value': 'cube-root'},
                                    {'label': '4th root', 'value': '4th-root'},
                                    {'label': '5th-root', 'value': '5th-root'},
                                    {'label': '6th root', 'value': '6th-root'},
                                ],
                                value='6th-root',
                                clearable=False,
                                style={
                                    'marginBottom': '10px', 
                                    'width': '100%',
                                    'color': 'black'
                                }
                            )
                        ], width=6)
                    ]),

                    html.Label("Select Time Range:", style={'color': 'rgba(255, 255, 255, 0.7)'}),
                    dcc.RangeSlider(
                        id="period-range-slider",
                        min=0,
                        max=len(quarters) - 1,
                        value=[0, len(quarters) - 1],
                        marks={i: quarters[i] for i in range(0, len(quarters), 16)},
                        className="mb-3"
                    ),
                    html.Div([
                        dbc.Button([
                            html.I(className="fas fa-play me-1", style={"color": "black"}),
                            html.Span("Play", style={"color": "black"})
                        ], 
                            id="play-button", 
                            color="success",
                            className="me-2"
                        ),
                        dbc.Button([
                            html.I(className="fas fa-pause me-1", style={"color": "black"}),
                            html.Span("Pause", style={"color": "black"})
                        ], 
                            id="pause-button", 
                            color="warning",
                            className="me-2"
                        ),
                        dbc.Button([
                            html.I(className="fas fa-clock me-1", style={"color": "black"}),
                            html.Span("Clear Time Range", style={"color": "black"})
                        ], 
                            id="clear-time-range", 
                            color="secondary",
                            className="me-2"
                        ),
                        dbc.Button([
                            html.I(className="fas fa-eraser me-1", style={"color": "black"}),
                            html.Span("Clear Hexes", style={"color": "black"})
                        ], 
                            id="clear-hexes", 
                            color="danger"
                        ),
                    ], className="d-flex justify-content-start gap-2 mb-3"),
                ], style={
                    'backgroundColor': 'rgb(15, 15, 15)',
                    'borderRadius': '5px',
                    'padding': '20px',
                    'height': '100%'
                }),
                width=4
            )
        ]),

        # Bottom Row: Maps
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H3(id='map-aggregated-title', 
                        className="text-center mb-2",
                        style={'color': 'rgba(255, 255, 255, 0.8)'}
                    ),
                    dcc.Graph(id='map-aggregated')
                ]),
                width=6
            ),
            dbc.Col(
                html.Div([
                    html.H3(id='map-quarterly-title', 
                        className="text-center mb-2",
                        style={'color': 'rgba(255, 255, 255, 0.8)'}
                    ),
                    dcc.Graph(id='map-quarterly')
                ]),
                width=6
            )
        ], className="my-3"),

        # Footer
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.Hr(style={'borderColor': 'rgba(255, 255, 255, 0.2)'}),
                    html.P([
                        "Created by ",
                        html.A("David Leather", 
                            href="https://daveleather.com",
                            target="_blank",
                            style={'color': 'rgba(255, 99, 132, 0.8)'}
                        ),
                        ". Data from NYC Department of Buildings."
                    ], className="text-center", style={'color': 'rgba(255, 255, 255, 0.6)'})
                ]),
                width=12
            )
        ),

        # Hidden dcc.Store for Map View (holds current map center and zoom)
        dcc.Store(
            id='map_view_store',
            data={
                'center': {'lat': 40.7, 'lon': -73.9},
                'zoom': 9,
                'bearing': 0,
                'pitch': 0
            }
        ),

        # Existing Stores, Interval, and Debug Div
        dcc.Store(
            id='global_filter',
            data={
                "startQuarterIndex": 0,
                "endQuarterIndex": len(quarters) - 1,
                "currentQuarterIndex": 0,
                "permitType": "NB",
                "play": False,
                "speed": 1000,
                "selectedHexes": [],
                "resetMaps": False
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
