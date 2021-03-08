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
from . import anim
from . import utils

displays = {}

# global dropdown selector codes
sq_anim = 'sq_anim'
pg_anim = 'pg_anim'
stop = 'stop'

error_message = 'Error! Polygon vertices not correctly specified. Please go back to the previous page and try again.'

def init_feeder(server):
    """Create the plots using plotly and dash."""
    dash_app = dash.Dash(
        server=server,
        routes_pathname_prefix='/feeder/',
        update_title=None,
        title='Part Feeder',
        external_scripts=["https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"]
        )

    dash_app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='page_content'),       # this displays all the plots and explanations
        dcc.Interval(
            id='data_update_interval',
            interval=10_000,
            disabled=False
            ),
        dcc.Interval(
            id='anim_update_interval',
            interval=50,
            disabled=False
            ),
        dcc.Store(
            id='anim_data',
            data=[],
            storage_type='memory'
            ),
        dcc.Store(
            id='anim_holding_data',     # hold data here, wait for clientside callback to update it.
            data=[],
            storage_type='memory'
        ),
        dcc.Store(
            id='prev_anim',
            storage_type='memory'
        ),
        dcc.Dropdown(       # dropdown selector to determine which animation to display.
            id='anim_selector',
            options=[
                {'label': 'Squeeze Plan', 'value': sq_anim},
                {'label': 'Push Grasp Plan', 'value': pg_anim},
                {'label': 'Stop Animation', 'value': stop}
            ],
            searchable=False,
            style={'display': 'none'},
            clearable=False,
            value=pg_anim
            ),
        dcc.Graph(          # animation figure
            id='anim', 
            style={'height': '50vh', 'margin': 'auto', 'display': 'none'},
            config={'displayModeBar': False, 'staticPlot': True}
            ),
        html.Div(
            id='loading',
            children='Loading...',
            style={'margin': 'auto', 'width': '100%', 'padding': '10px', 'text-align': 'center', 'font-size': 'large'}
            )
        ]
    )

    init_callbacks(dash_app)

    return dash_app.server


def init_callbacks(app):

    @app.callback(
        Output('page_content', 'children'),
        Input('url', 'search'))
    def display_feeder(search: str):
        """
        This parses a list of vertices describing a polygon from the query string in the URL.
        """
        if search and search[0] == '?':
            search = search[1:]
        if not search:
            return error_message
        points = parse_qs(search)

        if 'x' not in points or 'y' not in points or len(points['x']) != len(points['y']) or len(points['x']) < 3:
            return error_message

        assert 'x' in points
        assert 'y' in points
        assert len(points['x']) == len(points['y'])

        x = list(map(int, points['x']))
        y = list(map(int, points['y']))

        points = np.hstack((np.array(x).reshape((-1, 1)), np.array(y).reshape((-1, 1))))
        points = engine.scale_and_center_polygon(points)

        return create_page(points, search)

    app.clientside_callback(  # This clientside callback is a little bit of a hack but it works.
        """
        function(n, value, data, holding_data, prev) {
            if (value !== "stop" && value !== prev) {
                // clear out data
                data.length = 0;
            }
            for (var i=0; i<holding_data.length; i++) {
                data.push(holding_data.shift());
            }
            if (data.length > 1) {
                return data.shift(); //[data[0], data.slice(1)];
            } else if (data.length == 1) {
                return data[0]; //[data[0], data];
            }
        }
        """,
        Output('anim', 'figure'),
        Input('anim_update_interval', 'n_intervals'),
        Input('anim_selector', 'value'),
        State('anim_data', 'data'),
        State('anim_holding_data', 'data'),
        State('prev_anim', 'data')
        )

    @app.callback(
        Output('anim_holding_data', 'data'),
        Output('data_update_interval', 'disabled'),
        Output('anim_update_interval', 'disabled'),
        Output('prev_anim', 'data'),
        Input('data_update_interval', 'n_intervals'),
        Input('anim_selector', 'value'),
        State('url', 'search'),
        State('prev_anim', 'data')
        )
    def update_anim_data(n, value, search, prev):
        loops = 5
        ctx = dash.callback_context

        if ctx.triggered:
            if search and search[0] == '?':
                search = search[1:]

            ### TODO one display per session
            d = displays.get(search, None)
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger == 'anim_selector' and value == stop:
                return [], True, True, prev

            # selector has not changed. continue calling for data.
            elif d and value != stop:
                return d.get(value).step_draw(loops=loops), False, False, value
        # print('I got here! disabling update intervals. ')
        if value == stop:
            return [], True, True, prev
        else:
            return [], False, False, value

    @app.callback(
        Output('anim_selector', 'style'),
        Output('anim', 'style'),
        Output('loading', 'style'),
        Input('page_content', 'children')
    )
    def show_anim(content):
        if content and content != error_message:
            return {'display': 'block'}, {'display': 'block', 'height': '50vh', 'margin': 'auto'}, {'display': 'none'}
        elif content == error_message:
            return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
        return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}


def create_page(points, hash_str):
    s_domain = (0, 2*np.pi)     # s_domain refers to all squeeze function related domains
    pg_domain = (0, 4*np.pi)    # pg_domain refers to all push-grasp function related domains

    ch = engine.convex_hull(points)
    antipodal_pairs = engine.antipodal_pairs(ch)

    # Diameter function
    piecewise_diameter = engine.make_diameter_function(ch)

    bounded_piecewise_diameter = engine.generate_bounded_piecewise_func(piecewise_diameter, period=np.pi)
    diameter_callable = engine.generate_bounded_callable(bounded_piecewise_diameter, period=np.pi)
    diameter_maxima, diameter_minima = engine.find_bounded_extrema(bounded_piecewise_diameter,
                                                                   period=np.pi, domain=s_domain)

    # Squeeze function
    squeeze_func = engine.generate_transfer_from_extrema(diameter_minima, diameter_maxima)
    squeeze_callable = engine.generate_transfer_extrema_callable(squeeze_func, period=np.pi)

    # Radius function
    piecewise_radius = engine.make_radius_function(ch)
    bounded_piecewise_radius = engine.generate_bounded_piecewise_func(piecewise_radius, period=2*np.pi)
    radius_callable = engine.generate_bounded_callable(bounded_piecewise_radius, period=2*np.pi)
    radius_maxima, radius_minima = engine.find_bounded_extrema(bounded_piecewise_radius,
                                                               period=2*np.pi, domain=pg_domain)

    # Push function
    push_func = engine.generate_transfer_from_extrema(radius_minima, radius_maxima)
    push_callable = engine.generate_transfer_extrema_callable(push_func, period=2*np.pi)

    # Push-grasp function
    extended_squeeze = engine.generate_transfer_from_extrema(
        *reversed(engine.find_bounded_extrema(bounded_piecewise_diameter, period=np.pi, domain=pg_domain, buffer=2))
    )
    push_grasp_func = engine.generate_bounded_push_grasp_function(push_func, extended_squeeze)
    push_grasp_callable = engine.generate_transfer_extrema_callable(push_grasp_func, period=2*np.pi)

    # generate squeeze plan
    sq_intervals = engine.generate_intervals(squeeze_func, default_T=np.pi)
    sq_plan = engine.generate_plan(sq_intervals)

    # generate push-grasp plan
    pg_intervals = engine.generate_intervals(push_grasp_func, default_T=2*np.pi)
    pg_plan = engine.generate_plan(pg_intervals)

    # create display
    d1 = anim.SqueezeDisplay(points, sq_plan, diameter_callable, squeeze_callable)
    d2 = anim.PushGraspDisplay(points, pg_plan, radius_callable, diameter_callable, push_callable, push_grasp_callable)
    displays[hash_str] = {sq_anim: d1, pg_anim: d2}

    # Create figures
    ch_ap_graph = create_graph_from_figure(
        create_antipodal_pairs_figure(points, ch, antipodal_pairs), 'ch_ap_fig')
    dia_graph = create_graph_from_figure(
        create_function_figure(diameter_callable, diameter_minima, diameter_maxima, s_domain), 'dia_fig')
    sq_graph = create_graph_from_figure(
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
            explanations.pg, pg_graph,
            explanations.anim]


def create_antipodal_pairs_figure(points, convex_hull, antipodal_pairs):
    draw_points = np.vstack((points, points[0]))
    draw_ch = np.vstack((convex_hull, convex_hull[0]))

    fig = create_base_figure()

    # Plot base polygon
    fig.add_trace(go.Scatter(
        x=draw_points[:, 0],
        y=draw_points[:, 1],
        mode='lines',
        fill='toself',
        fillcolor='#2D4262'
        ))
    # Plot convex hull
    fig.add_trace(go.Scatter(
        x=draw_ch[:, 0],
        y=draw_ch[:, 1],
        mode='lines',
        line=go.scatter.Line(color='black', width=4)
    ))
    # Plot antipodal pairs
    for p1, p2 in antipodal_pairs:
        x1, y1 = convex_hull[p1]
        x2, y2 = convex_hull[p2]
        fig.add_trace(go.Scatter(
            x=[x1, x2], 
            y=[y1, y2],
            mode='lines',
            line=go.scatter.Line(color='#DB9501')
        ))
    return fig


def create_function_figure(func_callable, minima, maxima, domain=(0, 2*np.pi)):
    steps = round((domain[1]-domain[0])/(np.pi/2)) + 1
    fig = go.Figure(
        layout=go.Layout(
            showlegend=False,
            xaxis=go.layout.XAxis(
                tickmode='array',
                tickvals=np.linspace(*domain, steps),
                ticktext=utils.generate_latex_text(domain[0], steps),
                range=domain,
                fixedrange=True
            ))
        )

    x = np.linspace(*domain, 1000)
    y = np.array([func_callable(t) for t in x])

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=go.scatter.Line(color='black')
    ))

    y_lower = max(0, min(y)-20)

    for e in minima:
        fig.add_trace(go.Scatter(
            x=[e, e],
            y=[y_lower, func_callable(e)],
            mode='lines',
            line=go.scatter.Line(color='blue')
        ))

    for e in maxima:
        fig.add_trace(go.Scatter(
            x=[e, e],
            y=[y_lower, func_callable(e)],
            mode='lines',
            line=go.scatter.Line(color='red')
        ))

    return fig


def create_transfer_figure(transfer_callable, domain=(0, 2*np.pi)):
    x = np.linspace(*domain, 1000)
    y = np.array([transfer_callable(t) for t in x])

    steps = round((domain[1]-domain[0])/(np.pi/2)) + 1

    fig = create_base_figure()
    fig.update_layout(
        yaxis=go.layout.YAxis(
            tickmode='array',
            tickvals=np.linspace(*domain, steps),
            ticktext=utils.generate_latex_text(domain[0], steps)
            ),
        xaxis=go.layout.XAxis(
            tickmode='array',
            tickvals=np.linspace(*domain, steps),
            ticktext=utils.generate_latex_text(domain[0], steps)  # [r'$$\pi/2$$' for _ in range(steps)]
            ),
        )

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=go.scatter.Line(color='black')
    ))

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
        style={'width': '50vw', 'height': '50vw', 'max-width': '800px', 'max-height': '800px', 'margin': 'auto'},
        config={'displayModeBar': False, 'staticPlot': True}
    )

    return graph