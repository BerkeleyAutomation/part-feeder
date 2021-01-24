# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly
import numpy as np
import json
import math
from urllib.parse import parse_qs

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    dcc.Interval(
        id='update_interval',
        interval=20,
        ),
    dcc.Graph(
        id='live_update_graph',
        style={'width': '50vw', 'height': '50vw', 'margin': 'auto'},
        config={'displayModeBar': False})
])

# points = np.array([]).reshape((-1, 2)).astype(np.int32)
points = np.array([(-25, -33), (25, -33), (25, 20), (0, 45), (-25, 20)]).T # House shape

@app.callback(
    Output('live_update_graph', 'figure'),
    Input('update_interval', 'n_intervals'))
def update_graph(n):
    global points

    fig = go.Figure(
        layout=go.Layout(
            xaxis=go.layout.XAxis(range=(-50, 50)),
            yaxis=go.layout.YAxis(scaleanchor='x', range=(-50, 50)),
            showlegend=False))

    theta = np.pi/50
    s, c = np.sin(theta), np.cos(theta)
    matrix = np.array([[c, -s],
                      [s, c]])
    points = np.dot(matrix, points)

    fig.add_trace(go.Scatter(
        x=points[0],
        y=points[1],
        mode='lines',
        fill='toself'))

    return fig 

if __name__ == '__main__':
    app.run_server(debug=True)