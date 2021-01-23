import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly
import plotly.graph_objs as go

import numpy as np

import json, math
from urllib.parse import parse_qs

from . import engine
from . import explanations

def init_feeder(server):
    """Create a Plotly Dash dashboard."""
    dash_app = dash.Dash(
        server=server,
        routes_pathname_prefix='/feeder/'
        )

    dash_app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='page_content')
        ])

    init_callbacks(dash_app)

    return dash_app.server

def init_callbacks(app):

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
    s_domain = (0, 2*np.pi)     # s_domain refers to all squeeze function related domains
    pg_domain = (0, 4*np.pi)    # pg_domain refers to all push-grasp function related domains

    ch = engine.convex_hull(points)
    antipodal_pairs = engine.antipodal_pairs(ch)

    # Diameter function
    piecewise_diameter = engine.make_diameter_function(ch)
    piecewise_diameter_range = engine.generate_range(piecewise_diameter, period=np.pi, domain=s_domain)
    diameter_callable = engine.generate_callable(piecewise_diameter_range)
    diameter_maxima, diameter_minima = engine.find_extrema(piecewise_diameter_range, domain=s_domain)

    # Squeeze function
    squeeze_func = engine.make_transfer_function(piecewise_diameter_range, domain=s_domain)
    squeeze_callable = engine.make_transfer_callable(squeeze_func, domain=s_domain)

    # Radius function
    piecewise_radius = engine.make_radius_function(ch)
    piecewise_radius_range = engine.generate_range(piecewise_radius, period=2*np.pi, domain=pg_domain)
    radius_callable = engine.generate_callable(piecewise_radius_range)
    radius_maxima, radius_minima = engine.find_extrema(piecewise_radius_range, domain=pg_domain)

    # Push function
    push_func = engine.make_transfer_function(piecewise_radius_range, domain=pg_domain)
    push_callable = engine.make_transfer_callable(push_func, domain=pg_domain)

    # Push-grasp function
    dia_extended = engine.generate_range(piecewise_diameter, period=np.pi, domain=pg_domain)
    push_grasp_func = engine.make_push_grasp_function(
        dia_extended, piecewise_radius_range, domain=pg_domain)
    push_grasp_callable = engine.make_transfer_callable(push_grasp_func, domain=pg_domain)

    # Create figures
    ch_ap_graph = create_graph_from_figure(
        create_antipodal_pairs_figure(points, ch, antipodal_pairs), 'ch_ap_fig')
    dia_graph = create_graph_from_figure(
        create_function_figure(diameter_callable, diameter_minima, diameter_maxima, s_domain), 'dia_fig')
    sq_graph = sq_graph = create_graph_from_figure(
        create_transfer_figure(squeeze_callable, s_domain), 'sq_fig')

    rad_graph = create_graph_from_figure(
        create_function_figure(radius_callable, radius_minima, radius_maxima, pg_domain), 'rad_fig')
    pu_graph = create_graph_from_figure(
        create_transfer_figure(push_callable, pg_domain), 'pu_fig')

    pg_graph = create_graph_from_figure(
        create_transfer_figure(push_grasp_callable, pg_domain), 'pg_fig')

    return [explanations.ch_ap, ch_ap_graph, 
            explanations.dia, dia_graph, 
            explanations.sq, sq_graph,
            explanations.rad, rad_graph,
            explanations.pu, pu_graph,
            explanations.pg, pg_graph]

def create_antipodal_pairs_figure(points, convex_hull, antipodal_pairs):
    draw_points = np.vstack((points, points[0]))
    draw_ch = np.vstack((convex_hull, convex_hull[0]))

    fig = create_base_figure()

    # Plot base polygon
    fig.add_trace(go.Scatter(
        x=draw_points[:, 0],
        y=draw_points[:, 1],
        mode='lines',
        fill='toself'))
    # Plot convex hull
    fig.add_trace(go.Scatter(
        x=draw_ch[:, 0],
        y=draw_ch[:, 1],
        mode='lines',
        line=go.scatter.Line(color='red')
    ))
    # Plot antipodal pairs
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

def create_function_figure(func_callable, minima, maxima, domain=(0, 2*np.pi)):
    steps = round((domain[1]-domain[0])/(np.pi/2)) + 1
    fig = go.Figure(
        layout=go.Layout(
            showlegend=False,
            xaxis=go.layout.XAxis(
                tickmode='array',
                tickvals=np.linspace(*domain, steps))
            )
        )

    x = np.linspace(*domain, 1000)
    y = np.array([func_callable(t) for t in x])

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=go.scatter.Line(color='black')))

    y_lower = max(0, min(y)-20)

    for e in minima:
        fig.add_trace(go.Scatter(
            x=[e, e],
            y=[y_lower, func_callable(e)],
            mode='lines',
            line=go.scatter.Line(color='blue')))

    for e in maxima:
        fig.add_trace(go.Scatter(
            x=[e, e],
            y=[y_lower, func_callable(e)],
            mode='lines',
            line=go.scatter.Line(color='red')))

    return fig

def create_transfer_figure(transfer_callable, domain=(0, 2*np.pi)):
    x = np.linspace(*domain, 1000)
    y = np.array([transfer_callable(t) for t in x])

    steps = round((domain[1]-domain[0])/(np.pi/2)) + 1

    fig = create_base_figure()
    fig.update_layout(
        yaxis=go.layout.YAxis(
            tickmode='array',
            tickvals=np.linspace(*domain, steps)),
        xaxis=go.layout.XAxis(
            tickmode='array',
            tickvals=np.linspace(*domain, steps))
        )

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