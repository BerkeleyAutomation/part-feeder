import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly
import plotly.graph_objs as go

import numpy as np

import json, math
from urllib.parse import parse_qs

import engine

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page_content')])

@app.callback(
    Output('page_content', 'children'),
    Input('url', 'search'))
def display_feeder(search: str):
    if search and search[0] == '?':
        search = search[1:]
    points = parse_qs(search)

    assert 'x' in points
    assert 'y' in points
    assert len(points['x']) == len(points['y'])

    x = list(map(int, points['x']))
    y = list(map(int, points['y']))

    points = np.hstack((np.array(x).reshape((-1, 1)), np.array(y).reshape((-1, 1))))
    points = engine.scale_and_center_polygon(points)

    return create_page(points)

def create_page(points):

    convex_hull = engine.convex_hull(points)
    antipodal_pairs = engine.antipodal_pairs(convex_hull)
    piecewise_func, diameter_func = engine.make_diameter_function(convex_hull)
    maxima, minima = engine.find_extrema(piecewise_func, diameter_func)
    squeeze_func = engine.make_squeeze_func(piecewise_func, diameter_func)

    ch_graph = create_graph_from_figure(create_convex_hull_figure(points, convex_hull), 'ch_fig')
    ap_graph = create_graph_from_figure(create_antipodal_pairs_figure(points, convex_hull, antipodal_pairs), 'ap_fig')
    dia_graph = create_graph_from_figure(create_diameter_figure(diameter_func, minima, maxima), 'dia_fig')
    sq_graph = create_graph_from_figure(create_squeeze_figure(squeeze_func), 'sq_fig')
    return [ch_graph, ap_graph, dia_graph, sq_graph]

def create_convex_hull_figure(points, convex_hull):
    draw_points = np.vstack((points, points[0]))
    draw_ch = np.vstack((convex_hull, convex_hull[0]))

    convex_hull_fig = create_base_figure()
    convex_hull_fig.add_trace(go.Scatter(
        x=draw_points[:, 0],
        y=draw_points[:, 1],
        mode='lines',
        fill='toself'))
    convex_hull_fig.add_trace(go.Scatter(
        x=draw_ch[:, 0],
        y=draw_ch[:, 1],
        mode='lines',
        line=go.scatter.Line(color='black')
    ))

    return convex_hull_fig

def create_antipodal_pairs_figure(points, convex_hull, antipodal_pairs):
    draw_points = np.vstack((points, points[0]))
    draw_ch = np.vstack((convex_hull, convex_hull[0]))

    fig = create_base_figure()

    fig.add_trace(go.Scatter(
        x=draw_points[:, 0],
        y=draw_points[:, 1],
        mode='lines',
        fill='toself'))
    for p1, p2 in antipodal_pairs:
        x1, y1 = convex_hull[p1]
        x2, y2 = convex_hull[p2]
        fig.add_trace(go.Scatter(
            x=[x1, x2], 
            y=[y1, y2],
            mode='lines',
            line=go.scatter.Line(color='black')
        ))
    return fig

def create_diameter_figure(diameter_func, minima, maxima):
    fig = go.Figure(layout=go.Layout(showlegend=False))

    x = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    y = np.array([diameter_func(t) for t in x])

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=go.scatter.Line(color='black')))

    for e in minima:
        fig.add_trace(go.Scatter(
            x=[e, e],
            y=[0, diameter_func(e)],
            mode='lines',
            line=go.scatter.Line(color='blue')))

    for e in maxima:
        fig.add_trace(go.Scatter(
            x=[e, e],
            y=[0, diameter_func(e)],
            mode='lines',
            line=go.scatter.Line(color='red')))

    return fig

def create_squeeze_figure(squeeze_func):
    x = np.linspace(0, 2*np.pi, 1000)
    y = np.array([squeeze_func(t) for t in x])

    fig = create_base_figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=go.scatter.Line(color='black')))

    return fig

def create_base_figure():
    fig = go.Figure(
        layout=go.Layout(
            yaxis=go.layout.YAxis(scaleanchor='x'),
            showlegend=False))
    return fig

def create_graph_from_figure(figure, id):
    graph = dcc.Graph(
        id=id, 
        figure=figure,
        style={'width': '50vw', 'height': '50vw', 'margin': 'auto'},
        config={'displayModeBar': False})

    return graph

if __name__ == '__main__':
    app.run_server(debug=True)