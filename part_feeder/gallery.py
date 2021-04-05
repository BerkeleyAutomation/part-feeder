import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import plotly
import plotly.graph_objs as go

import numpy as np

import json

from .db import get_db
from .feeder import create_base_figure, create_graph_from_figure
from .engine import scale_and_center_polygon
grid = 3


def init_gallery(app):
    dash_app = dash.Dash(
        server=app,
        routes_pathname_prefix='/part-feeder/gallery/',
        update_title=None,
        title='Gallery',
        external_stylesheets=[dbc.themes.BOOTSTRAP]
    )

    dash_app.layout = html.Div([
        dcc.Markdown(
            children='''
            # Gallery
            
            Click on any polygon to view its plan!
            ''',
            style={'text-align': 'center'}
        ),
        html.Div(
            children=dbc.Row(
                [
                    dbc.Col(html.Button('Previous Page', id='prev', n_clicks=0, style={'text-align': 'center'})),
                    dbc.Col(html.Button('Next Page', id='next', n_clicks=0, style={'text-align': 'center'}))
                ]
            ),
            style={'text-align': 'center'}
        ),
        html.Div(
            id='page-content',
            children=[]
        ),
        dcc.Store(
            id='index',
            data=0,
            storage_type='memory'
        )
    ])

    @dash_app.callback(
        Output('page-content', 'children'),
        Input('index', 'data')
    )
    def generate_grid(index):
        db, Part = get_db()

        polygons = Part.query.slice(index, min(Part.query.count(), index+grid**2)).all()

        polygons = list(map(lambda p: json.loads(p.points), polygons))

        ret = []
        for i in range(grid):
            row = []
            for j in range(grid):
                idx = i*grid+j
                if idx < len(polygons):
                    graph = create_graph(polygons[idx])
                    link = html.A(children=graph, href='/part-feeder/feeder/?' + dict_to_query_string(polygons[idx]))
                    row.append(dbc.Col(link))
                else:
                    row.append(dbc.Col())

            ret.append(dbc.Row(row, no_gutters=True))

        return dbc.Container(children=ret)

    @dash_app.callback(
        Output('index', 'data'),
        Output('next', 'disabled'),
        Output('prev', 'disabled'),
        Input('next', 'n_clicks'),
        Input('prev', 'n_clicks'),
        State('index', 'data')
    )
    def change_page(_n1, _n2, index):
        ctx = dash.callback_context

        ret_idx = index
        if ctx.triggered:
            button = ctx.triggered[0]['prop_id'].split('.')[0]
            if button == 'next':
                ret_idx += grid ** 2
            elif button == 'prev':
                ret_idx -= grid ** 2

        db, Part = get_db()
        next_disabled = ret_idx + grid ** 2 >= Part.query.count()
        prev_disabled = ret_idx == 0

        return ret_idx, next_disabled, prev_disabled

    return dash_app.server


def dict_to_query_string(points):
    points = {key: [f'{key}={i}' for i in points[key]] for key in points}
    return '&'.join(points['x']) + '&' + '&'.join(points['y'])


def create_graph(points: dict):
    x = list(map(int, points['x']))
    y = list(map(int, points['y']))
    points = np.hstack((np.array(x).reshape((-1, 1)), np.array(y).reshape((-1, 1))))
    points = scale_and_center_polygon(points)
    draw_points = np.vstack((points, points[0]))

    fig = create_base_figure()
    fig.add_trace(go.Scatter(
        x=draw_points[:, 0],
        y=draw_points[:, 1],
        mode='lines',
        fill='toself',
        fillcolor='#2D4262'
    ))

    fig.update_layout(xaxis_showticklabels=False, yaxis_showticklabels=False)
    fig.update_layout(xaxis_range=(-60, 60), yaxis_range=(-60, 60))

    return dcc.Graph(figure=fig, config={'displayModeBar': False, 'staticPlot': True})
