"""
This module animates the squeeze plan to orient the part up to symmetry. Uses the 2D physics simulator
pymunk to simulate rigid polygons being squeezed and oriented up to symmetry.
"""
import numpy as np

import pymunk
from pymunk.vec2d import Vec2d

import math, random

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
        self.orth_slope = -1/self.slope if not np.isclose(0, self.slope) else -math.inf
        length, distance = length / 2, distance / 2
        
        if math.isfinite(self.orth_slope):
            # starting position offsets
            self.x_offset = distance/math.sqrt(self.orth_slope**2+1)
            self.y_offset = (self.orth_slope*distance)/math.sqrt(self.orth_slope**2+1)

            # squeeze velocity components, magnitude of velocity vector = velocity
            self.vel_x = velocity/math.sqrt(self.orth_slope**2+1)
            self.vel_y = velocity*self.orth_slope/math.sqrt(self.orth_slope**2+1)
        else:
            self.x_offset = 0
            self.y_offset = distance
            
            self.vel_x = 0
            self.vel_y = velocity
        
        # bottom gripper
        self.bot_vel = Vec2d(self.vel_x, self.vel_y) # squeeze velocity vector
        self.bot_pos = Vec2d(x-self.x_offset, y-self.y_offset) # starting position vector
        self.bot, self.bot_seg = Gripper.make_gripper(self.bot_pos, length, angle)
        
        # top gripper
        self.top_vel = -Vec2d(self.vel_x, self.vel_y) # squeeze velocity vector
        self.top_pos = Vec2d(x+self.x_offset, y+self.y_offset) # starting position vector 
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
        eps = 3
        
        if (self.top.position - self.top_pos).length < eps:
            self.top.velocity = 0, 0
        if (self.bot.position - self.bot_pos).length < eps:
            self.bot.velocity = 0, 0
        
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
            self.body.angle = random.uniform(0, 2*math.pi)
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
    
    def __init__(self, points, angles):
        self.angles = angles
        self.points = points
        
        self.gripper_pos = np.array([i * 250 for i in range(1, len(angles)+1)])
        self.start_pos = 0
        self.del_pos = (len(angles) + 1) * 250      ### Keep one display location?
        
        self.xlim = (self.start_pos, self.del_pos)
        self.ylim = (-200, 200)
        
        self.space = pymunk.Space(threaded=True)
        self.space.threads = 2
        self.space.gravity = 0, 0
        self.space.damping = 1
        self.step_size = 1/50       # time between each frame
        
        self.init_grippers()    
        self.polygons = []

        self.init_draw_points()

    def init_draw_points(self):
        """Initializes the draw points"""
        self.draw_points = np.vstack((self.points, self.points[0]))
        self.draw_points = self.draw_points.T

        # thick line for alignment purposes
        self.thick_line = self.draw_points[:, :2]
        for i in range(1, len(self.draw_points)-1):
            if math.dist(self.draw_points[:, i], self.draw_points[:, i+1]) > \
                math.dist(self.thick_line[:, 0], self.thick_line[:, 1]):
                self.thick_line = self.draw_points[:, i:i+2]
        
    def init_grippers(self):
        """Initializes the grippers according to their angles and adds them to the space."""
        self.grippers = []
        
        for angle, xpos in zip(self.angles, self.gripper_pos):
            g = Gripper(xpos, 0, angle)
            
            self.grippers.append(g)
            self.space.add(g.top, g.top_seg)
            self.space.add(g.bot, g.bot_seg)
    
    def add_polygon(self):
        """Add a polygon to the space. Polygon angles are randomly sampled from [0, 2*pi)."""
        p = Polygon(self.start_pos, 0, self.points)
        self.polygons.insert(0, p)
        self.space.add(p.body, p.poly)
      
    def step(self, dt):
        """
        This method should be overridden by subclasses. Squeeze plans and Push-grasp plans have
        different animation timing and steps.
        """
        for x in range(10):
            self.space.step(self.step_size/10)
        # raise NotImplementedError

    def step_draw(self, loops=1):
        """Steps the environment and returns a frame for each step for several seconds"""
        figs = []
        # steps = round(seconds/self.step_size)
        steps = type(self).TOTAL_TIME

        for i in range(steps):
            self.step(i)
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
                'xaxis': {'range': self.xlim},
                'yaxis': {
                    'scaleanchor': 'x',
                    'range': self.ylim
                },
                'showlegend': False
            }
        }

        return fig

    def __get_traces(self):
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
                        'width': 2*shape.radius
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
                    'line': {
                        'color': 'blue',
                    }
                }

                line = {
                    'type': 'scatter',
                    'x': rotated_line[0].tolist(),
                    'y': rotated_line[1].tolist(),
                    'mode': 'lines',
                    'line': {
                        'color': 'red',
                        'width': 4
                    }
                }

                traces.append(poly)
                traces.append(line)

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

        self.grippers_min_dist = []
        self.polygon_rotate_dist = []

        self.diameter_callable = diameter_callable
        self.squeeze_callable = squeeze_callable

    def step(self, dt):
        dt = dt % 500
        if dt == 0:
            # init move phase
            # create a new box; start moving all boxes to next one
            self.add_polygon()

            for p in self.polygons:
                p.reset_vel_func()
                p.move()
        elif 0 + 50 < dt < 200:
            # move phase
            
            # stop polygons when they are in between a gripper
            if self.polygons and (b := self.polygons[-1].body).position.x >= self.del_pos:
                self.space.remove(b, *b.shapes)
                self.polygons.pop()
            for p in self.polygons:
                if any(np.abs(p.body.position.x-self.gripper_pos) < 3):
                    p.body.velocity = 0, 0
        elif dt == 200: 
            # init squeeze phase
            self.space.damping = 0
            
            self.grippers_min_dist = [50] * len(self.grippers)
            self.polygon_rotate_dist = [0] * len(self.polygons)
            for i, p in enumerate(self.polygons[:len(self.grippers)]):
                p.squeeze()
                p.body.moment = math.inf
                rel_angle = (self.grippers[i].angle % (2*np.pi) - p.body.angle % (2*np.pi)) % (2*np.pi)
                
                self.polygon_rotate_dist[i] = self.diameter_callable(rel_angle)
                
                ###### TODO
                self.grippers_min_dist[i] = self.diameter_callable(self.squeeze_callable(rel_angle))

            for g in self.grippers:
                g.squeeze()      
        elif 200 < dt < 350:
            # squeeze phase  
            for i, g in enumerate(self.grippers): 
                distance = g.distance()
                
                ###### TODO
                if i < len(self.polygons) and abs(distance - self.polygon_rotate_dist[i]) < 10:
                    self.polygons[i].body.moment = 1e6
                ######
                
                if abs(distance - self.grippers_min_dist[i]) < 1:
                    g.stop()
                    if i < len(self.polygons):
                        self.polygons[i].poly.filter = Polygon.move_filter
                
        elif dt == 350:
            # init unsqueeze phase
            self.space.damping = 1
            for p in self.polygons:
                p.body.velocity_func = Polygon.zero_velocity

            for g in self.grippers:
                g.unsqueeze()
        elif dt > 350 :
            # unsqueeze phase
            for g in self.grippers:
                g.limit_unsqueeze()

        super().step(dt)

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

        self.gripper_push_dist = []
        self.gripper_squeeze_dist = []

        self.polygon_pins = []      # PivotJoint constraint to simulate part rotating after being
                                    # pushed by one plate

        self.radius_callable = radius_callable
        self.diameter_callable = diameter_callable
        self.push_callable = push_callable
        self.push_grasp_callable = push_grasp_callable

    def step(self, dt):
        dt = dt % 650
        if dt == 0:
            # init move phase
            # create a new box; start moving all boxes to next one
            self.add_polygon()

            for p in self.polygons:
                p.reset_vel_func()
                p.move()
        elif 0 + 50 < dt < 200:
            # move phase
            
            # stop polygons when they are in between a gripper
            if self.polygons and (b := self.polygons[-1].body).position.x >= self.del_pos:
                self.space.remove(b, *b.shapes)
                self.polygons.pop()
            for p in self.polygons:
                if any(np.abs(p.body.position.x-self.gripper_pos) < 3):
                    p.body.velocity = 0, 0
        elif dt == 200: 
            # init push phase
            self.space.damping = 0
            
            self.gripper_push_dist = [50] * len(self.grippers)
            self.gripper_squeeze_dist = [50] * len(self.grippers)
            self.polygon_pins = []
            for i, p in enumerate(self.polygons[:len(self.grippers)]):
                p.squeeze()
                rel_angle = (self.grippers[i].angle % (2*np.pi) - p.body.angle % (2*np.pi)) % (2*np.pi)
                
                # minimum pushing distance
                self.gripper_push_dist[i] = self.radius_callable(self.push_callable(rel_angle))

                # minimum squeezing distance
                self.gripper_squeeze_dist[i] = self.diameter_callable(self.push_grasp_callable(rel_angle))

                # lock polygon in place for push action
                pin = pymunk.PivotJoint(p.body, self.space.static_body, p.body.position)
                self.space.add(pin)
                self.polygon_pins.append(pin)

            for g in self.grippers:
                g.push()      
        elif 200 < dt < 350:
            # push phase  
            for i, g in enumerate(self.grippers): 
                if i < len(self.polygons):
                    distance = g.bot.position.get_distance(self.polygons[i].body.position)
                else:
                    distance = g.bot.position.get_distance(g.bot_pos)

                if abs(distance - self.gripper_push_dist[i]) < 2:
                    g.stop()

        elif dt == 350:
            # init squeeze phase
            # remove locking pivotjoint constraints and start sqeeze/grasping phase
            for p in self.polygon_pins:
                self.space.remove(p)
            for g in self.grippers:
                g.grasp()
        elif 350 < dt < 500  :
            # squeeze phase
            for i, g in enumerate(self.grippers):
                distance = g.distance()
                if abs(distance - self.gripper_squeeze_dist[i]) < 2:
                    g.stop()
                    if i < len(self.polygons):
                        self.polygons[i].poly.filter = Polygon.move_filter

        elif dt == 500:
            # init unsqueeze phase
            self.space.damping = 1
            for p in self.polygons:
                p.body.velocity_func = Polygon.zero_velocity

            for g in self.grippers:
                g.unsqueeze()
        elif dt > 500:
            # unsqueeze phase
            for g in self.grippers:
                g.limit_unsqueeze()

        super().step(dt)