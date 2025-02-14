# src/layout.py

import dash_bootstrap_components as dbc
from dash import html, dcc

layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row([
            dbc.Col(
                html.H1("NYC Permits Dashboard (Uvicorn Edition)"),
                width=12
            )
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Permit Type"),
                dcc.Dropdown(
                    id="permit-dropdown",
                    options=[
                        {"label": "New Building", "value": "NB"},
                        {"label": "Demolition", "value": "DM"},
                    ],
                    value="NB",
                    clearable=False
                ),
                dbc.Button("Play", id="play-button", color="primary", className="mt-2"),
                dbc.Button("Clear Time Range", id="clear-time", color="secondary", className="mt-2 ms-2"),
            ], width=3),
            dbc.Col([
                dcc.Graph(id="time-series-graph")
            ], width=9)
        ], className="my-3"),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="map-graph")
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P("Created by Me. Data from NYC Open Data, etc.")
            ])
        ])
    ]
)
