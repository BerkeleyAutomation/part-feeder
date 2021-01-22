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

snap_threshold = 4

def create_canvas():
    start, end = -100, 100
    length = end - start + 1
    base = np.linspace(start, end, length, dtype=np.int32)
    data = np.hstack((np.repeat(base, length).reshape((-1, 1)), np.tile(base, length).reshape((-1, 1))))

    fig = go.Figure(data=go.Scattergl(
        x = data[:, 0],
        y = data[:, 1],
        mode='markers',
        opacity=0
    ), 
    layout=go.Layout(
        yaxis=go.layout.YAxis(scaleanchor='x', range=[-100, 100]),
        xaxis=go.layout.XAxis(range=[-100, 100])
    ))
    return fig

app.layout = html.Div(children=[
    dcc.Location(id='url', refresh=False),
    dcc.Graph(
        id='draw_polygon_graph', 
        style={'width': '50vw', 'height': '50vw', 'margin': 'auto'},
        config={'displayModeBar': False},
        figure=create_canvas()), 
    html.Div(className='row', children=[
        html.Div([
            dcc.Markdown("""
                **Hover Data**

                Mouse over values in the graph.
            """),
            html.Pre(id='hover-data')
        ], className='three columns'),

        html.Div([
            dcc.Markdown("""
                **Click Data**

                Click on points in the graph.
            """),
            html.Pre(id='click-data'),
        ], className='three columns')]),
    html.Div(id='draw_polygon_data', style={'display': 'none'}),
    html.Div(id='draw_polygon_graph_data', style={'display': 'none'}, children=plotly.io.to_json(create_canvas()))
])

# points = np.array([]).reshape((-1, 2)).astype(np.int32)

@app.callback(
    Output('draw_polygon_data', 'children'),
    Output('draw_polygon_graph', 'figure'),
    Input('draw_polygon_graph', 'hoverData'), 
    Input('draw_polygon_graph', 'clickData'),
    State('draw_polygon_data', 'children'),
    State('draw_polygon_graph', 'figure'))
def update_draw_polygon(hoverData, clickData, data, fig):
    data = json.loads(data or 'null')
    fig = go.Figure(**fig)

    if not data:
        data = {
            'points': [],
            'finish': False
            }

    points = np.array(data['points']).reshape((-1, 2))

    if not data['finish'] and clickData:
        point = np.array([clickData['points'][0]['x'], clickData['points'][0]['y']])

        if len(points) > 1 and math.dist(point, points[0]) < snap_threshold:
            data['finish'] = True

            fig.update_traces(
                        x=points[:, 0],
                        y=points[:, 1],
                        fill='toself',
                        selector={'uid': 'polygon_layer'})
        elif not any(all(p == point) for p in points):
            points = np.vstack((points, point))

            data['points'] = points.tolist()

            if len(points) == 1:
                fig.add_trace(go.Scattergl(
                    x=points[:, 0],
                    y=points[:, 1],
                    mode='lines',
                    opacity=1,
                    uid='polygon_layer'))
            else:
                fig.update_traces(
                    x=points[:, 0],
                    y=points[:, 1],
                    selector={'uid': 'polygon_layer'})

    # if hoverData:
    #     point = np.array([hoverData['points'][0]['x'], hoverData['points'][0]['y']])
    #     hover_points = np.vstack((points, point))

    #     if len(points) > 1 and math.dist(point, points[0]) < snap_threshold:
    #         point = points[0]

    #     if len(hover_points) > 1:
    #         fig.update_traces(
    #             x=hover_points[:, 0],
    #             y=hover_points[:, 1],
    #             selector={'uid': 'polygon_layer'})

    return json.dumps(data), fig

# @app.callback(
#     Output('draw_polygon_graph', 'figure'),
#     Input('data', 'children'))
# def update_graph(data):
#     data = json.loads(data)


#     if clickData:
#         point = np.array([clickData['points'][0]['x'], clickData['points'][0]['y']], dtype=np.int32)

#         if len(points) > 1 and np.linalg.norm(point-points[0]) < snap_threshold:
#             point = points[0]
#             fig.update_traces(
#                     x=points[:, 0],
#                     y=points[:, 1],
#                     selector={'uid': 'polygon_layer'})

#         elif not any(all(p == point) for p in points):
#             points = np.vstack((points, point))
#             if len(points) == 1:
#                 fig.add_trace(go.Scattergl(
#                     x=points[:, 0],
#                     y=points[:, 1],
#                     mode='lines',
#                     opacity=1,
#                     uid='polygon_layer'))
#             else:
#                 fig.update_traces(
#                     x=points[:, 0],
#                     y=points[:, 1],
#                     selector={'uid': 'polygon_layer'})
#     if hoverData:
#         point = np.array([hoverData['points'][0]['x'], hoverData['points'][0]['y']], dtype=np.int32)

#         if len(points) > 1 and np.linalg.norm(point-points[0]) < snap_threshold:
#             point = points[0]

#         hover_points = np.vstack((points, point))
#         if len(hover_points) > 1:
#             fig.update_traces(
#                 x=hover_points[:, 0],
#                 y=hover_points[:, 1],
#                 selector={'uid': 'polygon_layer'})
#     return fig

@app.callback(
    Output('hover-data', 'children'),
    Input('draw_polygon_graph', 'hoverData'),
    State('draw_polygon_data', 'children'),
    State('draw_polygon_graph', 'figure'))
def display_hover_data(hoverData, data, fig):
    # return json.dumps(fig, indent=2)
    return data

@app.callback(
    Output('click-data', 'children'),
    Input('url', 'search'))
def display_click_data(search):
    if search and search[0] == '?':
        search = search[1:]
    points = parse_qs(search)

    assert 'x' in points
    assert 'y' in points
    assert len(points['x']) == len(points['y'])

    

if __name__ == '__main__':
    app.run_server(debug=True)