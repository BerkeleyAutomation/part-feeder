"""
This module animates the squeeze plan to orient the part up to symmetry. Uses the 2D physics simulator
pymunk to simulate rigid polygons being squeezed and oriented up to symmetry.
"""
import numpy as np

import pymunk
from pymunk.vec2d import Vec2d

import math, random
from threading import Lock

import plotly.graph_objs as go
import plotly


class Gripper:
    """A pair of parallel plate grippers"""

    # Collision filter so that grippers do not collide with each other
    filter = pymunk.ShapeFilter(categories=0b01, mask=0b10)

    def __init__(self, x, y, angle, distance=200, length=200, velocity=50):
        # computing slope, slope of perpendicular
        self.angle = angle
        self.slope = math.tan(angle)
        self.orth_slope = -1 / self.slope if not np.isclose(0, self.slope) else -math.inf
        length, distance = length / 2, distance / 2

        if math.isfinite(self.orth_slope):
            # starting position offsets
            self.x_offset = distance / math.sqrt(self.orth_slope ** 2 + 1)
            self.y_offset = (self.orth_slope * distance) / math.sqrt(self.orth_slope ** 2 + 1)

            # squeeze velocity components, magnitude of velocity vector = velocity
            self.vel_x = velocity / math.sqrt(self.orth_slope ** 2 + 1)
            self.vel_y = velocity * self.orth_slope / math.sqrt(self.orth_slope ** 2 + 1)
        else:
            self.x_offset = 0
            self.y_offset = distance

            self.vel_x = 0
            self.vel_y = velocity

        # bottom gripper
        self.bot_vel = Vec2d(self.vel_x, self.vel_y)  # squeeze velocity vector
        self.bot_pos = Vec2d(x - self.x_offset, y - self.y_offset)  # starting position vector
        self.bot, self.bot_seg = Gripper.make_gripper(self.bot_pos, length, angle)

        # top gripper
        self.top_vel = -Vec2d(self.vel_x, self.vel_y)  # squeeze velocity vector
        self.top_pos = Vec2d(x + self.x_offset, y + self.y_offset)  # starting position vector
        self.top, self.top_seg = Gripper.make_gripper(self.top_pos, length, angle)

    @staticmethod
    def make_gripper(pos, length, angle, radius=2):
        """Creates a single jaw of a gripper. Grippers are represented as kinematic bodies with
        line segments for shape. They are frictionless."""
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = pos
        body.angle = angle

        segment = pymunk.Segment(body, a=(-length, 0), b=(length, 0), radius=radius)
        segment.friction = 0
        segment.elasticity = 0
        segment.filter = Gripper.filter

        return body, segment

    def squeeze(self):
        self.bot.velocity = self.bot_vel
        self.top.velocity = self.top_vel

    def unsqueeze(self):
        self.bot.velocity = -self.bot_vel
        self.top.velocity = -self.top_vel

    def push(self):
        self.bot.velocity = self.bot_vel

    def grasp(self):
        self.top.velocity = self.top_vel

    def stop(self):
        self.bot.velocity = 0, 0
        self.top.velocity = 0, 0

    def limit_unsqueeze(self):
        """When unsqueezing, ensures that both grippers stop at their original position"""
        eps = 10

        if (self.top.position - self.top_pos).length < eps:
            self.top.velocity = 0, 0
        if (self.bot.position - self.bot_pos).length < eps:
            self.bot.velocity = 0, 0

        if self.distance() > (self.top_pos - self.bot_pos).length:
            self.stop()

    def reset_pos_func(self):
        self.bot.position_func = pymunk.Body.update_position
        self.top.position_func = pymunk.Body.update_position

    def reset_vel_func(self):
        self.bot.velocity_func = pymunk.Body.update_velocity
        self.top.velocity_func = pymunk.Body.update_velocity

    def distance(self):
        """Returns the distance between the grippers"""
        return self.bot.position.get_distance(self.top.position)


class Polygon:
    """Polygon class. Polygons are frictionless and have infinite moment of inertia until they are
    squeezed by both plates. They are represented as rigid dynamic bodies."""

    # Collision filter when squeezing: collides with everything
    squeeze_filter = pymunk.ShapeFilter(categories=0b10, mask=pymunk.ShapeFilter.ALL_MASKS())

    # Collision filter when moving to next gripper: does not collide with grippers
    move_filter = pymunk.ShapeFilter(categories=0b10, mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b01)

    def __init__(self, x, y, points, angle=None):
        self.points = list(map(tuple, points))
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.position = x, y

        self.poly = pymunk.Poly(self.body, self.points, radius=.5)
        self.poly.mass = 1e3
        self.poly.friction = 0
        self.poly.elasticity = 0

        if angle is None:
            self.body.angle = random.uniform(0, 2 * math.pi)
        else:
            self.body.angle = angle

    def reset_pos_func(self):
        self.body.position_func = pymunk.Body.update_position

    def reset_vel_func(self):
        self.body.velocity_func = pymunk.Body.update_velocity

    def move(self):
        self.body.velocity = 100, 0
        self.poly.filter = Polygon.move_filter

    def squeeze(self):
        self.poly.filter = Polygon.squeeze_filter
        self.reset_vel_func()

    @staticmethod
    def zero_velocity(body, gravity, damping, dt):
        pymunk.Body.update_velocity(body, gravity, damping, dt)
        body.angular_velocity = 0
        body.velocity = 0, 0


class Display:
    """Display class controls the simulation"""

    # Override in subclass
    TOTAL_TIME = None

    def __init__(self, points, angles):
        self.rows = [-150, 150]
        self.angles = angles
        self.points = points

        spacing = 300

        self.gripper_pos = np.array([i * spacing for i in range(1, len(angles) + 1)])
        self.start_pos = 0
        self.del_pos = (len(angles) + 2) * spacing  ### Keep one display location?
        self.display_pos = (len(angles) + 1) * spacing

        self.xlim = (self.start_pos, self.del_pos)
        self.ylim = (-300, 300)

        self.space = pymunk.Space(threaded=True)
        self.space.threads = 2
        self.space.gravity = 0, 0
        self.space.damping = 1
        self.step_size = 1 / 50  # time between each frame

        self.grippers = [[] for _ in self.rows]
        self.init_grippers()
        self.polygons = [[] for _ in self.rows]

        self.init_draw_points()

        self.lock = Lock()

    def init_draw_points(self):
        """Initializes the draw points"""
        self.draw_points = np.vstack((self.points, self.points[0]))
        self.draw_points = self.draw_points.T

        # thick line for alignment purposes
        self.thick_line = self.draw_points[:, :2]
        for i in range(1, len(self.draw_points[0]) - 1):
            if math.dist(self.draw_points[:, i].flatten(), self.draw_points[:, i + 1].flatten()) > \
                    math.dist(self.thick_line[:, 0].flatten(), self.thick_line[:, 1].flatten()):
                self.thick_line = self.draw_points[:, i:i + 2]

    def init_grippers(self):
        """Initializes the grippers according to their angles and adds them to the space."""
        for i, r in enumerate(self.rows):
            for angle, xpos in zip(self.angles, self.gripper_pos):
                g = Gripper(xpos, r, angle)

                self.grippers[i].append(g)
                self.space.add(g.top, g.top_seg)
                self.space.add(g.bot, g.bot_seg)

    def add_polygon(self):
        """Adds a polygon to each row in the space. Polygon angles are randomly sampled from [0, 2*pi)."""
        for i, r in enumerate(self.rows):
            p = Polygon(self.start_pos, r, self.points)
            self.polygons[i].insert(0, p)
            self.space.add(p.body, p.poly)

    def step(self, dt):
        """
        This method should be overridden by subclasses. Squeeze plans and Push-grasp plans have
        different animation timing and steps.
        """
        # for x in range(10):
        #     self.space.step(self.step_size / 10)
        self.space.step(self.step_size)

    def step_draw(self, loops=1):
        """Steps the environment and returns a frame for each step for one loop of the cycle. """
        with self.lock:
            figs = []
            # steps = round(seconds/self.step_size)
            steps = type(self).TOTAL_TIME
            for _ in range(loops):
                for i in range(steps):
                    self.step(i)
                    if i % 16 == 0:
                        figs.append(self.draw())

            return figs

    def draw(self):
        """
        Returns a plotly figure drawing the current space. This was made specifically to draw the
        animations, so only supports segments and polygons.
        """

        traces = self.__get_traces()
        fig = {
            'data': traces,
            'layout': {
                'xaxis': {
                    'range': self.xlim,
                    'showticklabels': False,
                    'showgrid': False,
                    'zeroline': False
                },
                'yaxis': {
                    'scaleanchor': 'x',
                    'range': self.ylim,
                    'showticklabels': False,
                    'showgrid': False,
                    'zeroline': False
                },
                'showlegend': False  # ,
                # 'title': str(np.around([g.distance() for g in self.grippers], 3)) +
                # str(np.around([p.body.angle for p in self.polygons], 3))
                # 'title': str(np.around(np.array(self.stop_rotate_angle)*180/math.pi, 3)) +
                #          str(np.around([p.body.angle*180/math.pi for p in self.polygons], 3)) + '\n' +
                #          str(np.around(self.stop_rotate_angle, 3)) +
                #          str(np.around([p.body.angle for p in self.polygons], 3))
            }
        }

        return fig

    def __get_traces(self):
        """Returns the traces for all the segments and polygons in the animation. """
        traces = []

        for shape in self.space.shapes:
            theta = shape.body.angle

            # rotation matrix
            s, c = np.sin(theta), np.cos(theta)
            matrix = np.array([[c, -s],
                               [s, c]])

            # body position in world coordinates
            x, y = shape.body.position
            pos = np.array([x, y]).reshape((2, 1))

            if type(shape) is pymunk.Segment:
                x1, y1 = shape.a
                x2, y2 = shape.b

                # segment endpoints in local coordinates
                points = np.array(
                    [[x1, x2],
                     [y1, y2]])

                rotated = np.dot(matrix, points) + pos

                line = {
                    'type': 'scatter',
                    'x': rotated[0].tolist(),
                    'y': rotated[1].tolist(),
                    'line': {
                        'color': 'black',
                        'width': 2 * shape.radius
                    }
                }

                traces.append(line)

            elif type(shape) is pymunk.Poly:
                rotated = np.dot(matrix, self.draw_points) + pos
                rotated_line = np.dot(matrix, self.thick_line) + pos

                poly = {
                    'type': 'scatter',
                    'x': rotated[0].tolist(),
                    'y': rotated[1].tolist(),
                    'mode': 'lines',
                    'fill': 'toself',
                    'fillcolor': '#2D4262',  # Berkeley Blue!
                    'line': {
                        'color': 'black',
                    }
                }

                # thick alignment line
                line = {
                    'type': 'scatter',
                    'x': rotated_line[0].tolist(),
                    'y': rotated_line[1].tolist(),
                    'mode': 'lines',
                    'line': {
                        'color': '#DB9501',  # Berkeley Gold!
                        'width': 4
                    }
                }

                traces.append(poly)
                traces.append(line)

                # debug line
                # debug_line = np.dot(matrix, np.array([[-50, 50], [0, 0]])) + pos
                # debug_line_fig = {
                #     'type': 'scattergl',
                #     'x': debug_line[0].tolist(),
                #     'y': debug_line[1].tolist(),
                #     'mode': 'lines'
                # }
                # traces.append(debug_line_fig)

        return traces


class SqueezeDisplay(Display):
    """
    SqueezeDisplay inherits Display base class. Generates a squeeze plan animation.
    """

    MOVE_PART_TIME = 200
    SQUEEZE_PART_TIME = 150
    UNSQUEEZE_PART_TIME = 150
    TOTAL_TIME = MOVE_PART_TIME + SQUEEZE_PART_TIME + UNSQUEEZE_PART_TIME

    def __init__(self, points, angles, diameter_callable, squeeze_callable):
        super().__init__(points, angles)

        self.grippers_min_dist = [[] for _ in self.rows]
        self.polygon_rotate_dist = [[] for _ in self.rows]
        self.stop_rotate_angle = [[] for _ in self.rows]

        self.diameter_callable = diameter_callable
        self.squeeze_callable = squeeze_callable

    def step(self, dt):
        super().step(dt)

        dt = dt % SqueezeDisplay.TOTAL_TIME
        if dt == 0:
            # init move phase
            # create a new box; start moving all boxes to next one
            self.add_polygon()

            for r in self.polygons:
                for p in r:
                    p.reset_vel_func()
                    p.move()

            # stop the grippers
            for r in self.grippers:
                for g in r:
                    g.stop()
        elif 0 + 50 < dt < 200:
            # move phase

            # stop polygons when they are in between a gripper
            for r in self.polygons:
                if r and (b := r[-1].body).position.x >= self.del_pos:
                    self.space.remove(b, *b.shapes)
                    r.pop()
                for p in r:
                    if any(np.abs(p.body.position.x - self.gripper_pos) < 3) or \
                            abs(p.body.position.x - self.display_pos) < 3:
                        p.body.velocity = 0, 0
        elif dt == 200:
            # init squeeze phase
            self.space.damping = 0

            self.grippers_min_dist = [[50] * len(self.grippers[0]) for _ in self.rows]

            # distance to make the polygon able to rotate
            self.polygon_rotate_dist = [[0] * len(self.polygons[0]) for _ in self.rows]

            # angle to stop squeezing the part
            self.stop_rotate_angle = [[0] * len(self.polygons[0]) for _ in self.rows]

            for row_idx, row in enumerate(self.polygons):
                for i, p in enumerate(row[:len(self.grippers[0])]):
                    p.squeeze()
                    p.body.moment = math.inf

                    # angle of gripper relative to polygon
                    rel_angle = (self.grippers[row_idx][i].angle % (2 * np.pi) - p.body.angle % (2 * np.pi)) % (
                                2 * np.pi)
                    self.polygon_rotate_dist[row_idx][i] = self.diameter_callable(rel_angle)

                    # relative output angle of polygon
                    rel_output_angle = self.squeeze_callable(rel_angle)

                    # output angle of polygon in world frame
                    output_angle = (self.grippers[row_idx][i].angle % (2 * np.pi) - rel_output_angle) % (2 * np.pi)
                    self.stop_rotate_angle[row_idx][i] = output_angle % (2 * np.pi)

                    self.grippers_min_dist[row_idx][i] = self.diameter_callable(rel_output_angle)

            for r in self.grippers:
                for g in r:
                    g.squeeze()
        elif 200 < dt < 350:
            # squeeze phase
            for row_idx, row in enumerate(self.grippers):
                for i, g in enumerate(row):
                    distance = g.distance()

                    if i < len(self.polygons[0]) and abs(distance - self.polygon_rotate_dist[row_idx][i]) < 10:
                        self.polygons[row_idx][i].body.moment = 1e6
                    stop = False
                    if i < len(self.polygons[0]):
                        if abs(self.polygons[row_idx][i].body.angle % (2 * np.pi) - self.stop_rotate_angle[row_idx][i]) \
                                < 0.05:
                            stop = True
                    if abs(distance - self.grippers_min_dist[row_idx][i]) < 3:
                        stop = True
                    elif distance < 3:
                        stop = True
                    if stop:
                        g.stop()
                        if i < len(self.polygons[0]):
                            self.polygons[row_idx][i].poly.filter = Polygon.move_filter
                            self.polygons[row_idx][i].body.angle = self.stop_rotate_angle[row_idx][i]

        elif dt == 350:
            # init unsqueeze phase
            self.space.damping = 1
            for r in self.polygons:
                for p in r:
                    p.body.velocity_func = Polygon.zero_velocity
            for r in self.grippers:
                for g in r:
                    g.unsqueeze()
        elif dt > 350:
            # unsqueeze phase
            for r in self.grippers:
                for g in r:
                    g.limit_unsqueeze()



class PushGraspDisplay(Display):
    """
    PushGraspDisplay class inherits from Display base class. Generates a push-grasp plan animation.
    """

    MOVE_PART_TIME = 200
    PUSH_PART_TIME = 150
    SQUEEZE_PART_TIME = 150
    UNSQUEEZE_PART_TIME = 150
    TOTAL_TIME = MOVE_PART_TIME + PUSH_PART_TIME + SQUEEZE_PART_TIME + UNSQUEEZE_PART_TIME

    def __init__(self, points, angles, radius_callable, diameter_callable, push_callable, push_grasp_callable):
        super().__init__(points, angles)

        self.gripper_push_dist = [[] for _ in self.rows]
        self.gripper_squeeze_dist = [[] for _ in self.rows]

        self.stop_push_angle = [[] for _ in self.rows]
        self.stop_squeeze_angle = [[] for _ in self.rows]

        # PivotJoint constraint to simulate part rotating after being pushed by one plate
        self.polygon_pins = [[] for _ in self.rows]

        self.radius_callable = radius_callable
        self.diameter_callable = diameter_callable
        self.push_callable = push_callable
        self.push_grasp_callable = push_grasp_callable

    def step(self, dt):
        dt = dt % PushGraspDisplay.TOTAL_TIME
        if dt == 0:
            for row in self.grippers:
                for g in row:
                    g.stop()
            # init move phase
            # create a new box; start moving all boxes to next one
            self.add_polygon()

            for row in self.polygons:
                for p in row:
                    p.reset_vel_func()
                    p.move()
        elif 0 + 50 < dt < 200:
            # move phase

            # stop polygons when they are in between a gripper
            for r in self.polygons:
                if r and (b := r[-1].body).position.x >= self.del_pos:
                    self.space.remove(b, *b.shapes)
                    r.pop()
                for p in r:
                    if any(np.abs(p.body.position.x - self.gripper_pos) < 3) or \
                            abs(p.body.position.x - self.display_pos) < 3:
                        p.body.velocity = 0, 0
        elif dt == 200:
            # init push phase
            self.space.damping = 0

            # distance to stop pushing. fallback
            self.gripper_push_dist = [[50] * len(self.grippers[0]) for _ in self.rows]

            # distance to stop squeezing. fallback
            self.gripper_squeeze_dist = [[50] * len(self.grippers[0]) for _ in self.rows]

            # angle to stop pushing
            self.stop_push_angle = [[0] * len(self.polygons[0]) for _ in self.rows]

            # angle to stop squeezing
            self.stop_squeeze_angle = [[0] * len(self.polygons[0]) for _ in self.rows]

            self.polygon_pins = [[] for _ in self.rows]
            for row_idx, row in enumerate(self.polygons):
                for i, p in enumerate(row[:len(self.grippers[0])]):
                    p.squeeze()

                    # angle of gripper relative to polygon
                    rel_angle = (self.grippers[row_idx][i].angle % (2 * np.pi) - p.body.angle % (2 * np.pi)) % (
                                2 * np.pi)

                    rel_angle_2 = (p.body.angle % (2 * np.pi) - self.grippers[row_idx][i].angle % (2 * np.pi)) % (
                            2 * np.pi)

                    rel_push_output_angle = self.push_callable(rel_angle)
                    push_output_angle = (self.grippers[row_idx][i].angle % (2 * np.pi) - rel_push_output_angle) % (
                                2 * np.pi)

                    rel_pg_output_angle = self.push_grasp_callable(rel_angle)
                    pg_output_angle = (self.grippers[row_idx][i].angle % (2 * np.pi) - rel_pg_output_angle) % (
                                2 * np.pi)

                    # minimum pushing distance
                    self.gripper_push_dist[row_idx][i] = self.radius_callable(rel_push_output_angle)
                    # minimum squeezing distance
                    self.gripper_squeeze_dist[row_idx][i] = self.diameter_callable(rel_pg_output_angle)

                    # angle to stop pushing
                    self.stop_push_angle[row_idx][i] = push_output_angle

                    # angle to stop squeezing (push grasp)
                    self.stop_squeeze_angle[row_idx][i] = pg_output_angle

                    # lock polygon in place for push action
                    pin = pymunk.PivotJoint(p.body, self.space.static_body, p.body.position)
                    self.space.add(pin)
                    self.polygon_pins[row_idx].append(pin)

            for row in self.grippers:
                for g in row:
                    g.push()
        elif 200 < dt < 350:
            # push phase
            for row_idx, row in enumerate(self.grippers):
                for i, g in enumerate(row):
                    stop = False
                    if i < len(self.polygons[0]):
                        # stop if polygon is at the push angle or as a fallback at the push distance
                        distance = g.bot.position.get_distance(self.polygons[row_idx][i].body.position)
                        if abs(self.polygons[row_idx][i].body.angle % (2 * np.pi) - self.stop_push_angle[row_idx][i]) \
                                < 0.05 or abs(distance - self.gripper_push_dist[row_idx][i]) < 3:
                            stop = True
                            # self.polygons[row_idx][i].body.angle = self.stop_push_angle[row_idx][i]
                            # self.polygons[row_idx][i].poly.filter = Polygon.move_filter

                    elif abs(g.bot.position.get_distance(g.bot_pos) - self.gripper_push_dist[row_idx][i]) < 3:
                        stop = True
                    if stop:
                        g.stop()

        elif dt == 350:
            # init squeeze phase
            for row in self.polygons:
                for p in row:
                    p.squeeze()

            # remove locking pivotjoint constraints and start sqeeze/grasping phase
            for row_polygon_pins, row_grippers in zip(self.polygon_pins, self.grippers):
                for p in row_polygon_pins:
                    self.space.remove(p)
                for g in row_grippers:
                    g.grasp()
        elif 350 < dt < 500:
            # squeeze phase
            for row_idx, row in enumerate(self.grippers):
                for i, g in enumerate(row):
                    distance = g.distance()
                    stop = False
                    if i < len(self.polygons[0]):
                        if abs(self.polygons[row_idx][i].body.angle % (2 * np.pi) - self.stop_squeeze_angle[row_idx][i]) \
                                < 0.05:
                            stop = True
                    if abs(distance - self.gripper_squeeze_dist[row_idx][i]) < 3 or distance < 3:
                        stop = True

                    if stop:
                        g.stop()
                        if i < len(self.polygons[0]):
                            self.polygons[row_idx][i].poly.filter = Polygon.move_filter
                            self.polygons[row_idx][i].body.angle = self.stop_squeeze_angle[row_idx][i]

        elif dt == 500:
            # init unsqueeze phase
            self.space.damping = 1
            for row_polygons, row_grippers in zip(self.polygons, self.grippers):
                for p in row_polygons:
                    p.body.velocity_func = Polygon.zero_velocity

                for g in row_grippers:
                    g.unsqueeze()
        elif dt > 500:
            # unsqueeze phase
            for row in self.grippers:
                for g in row:
                    g.limit_unsqueeze()

        super().step(dt)
