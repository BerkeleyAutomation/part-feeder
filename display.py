class Gripper:
    
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
        self.bot_limiter = Gripper.make_unsqueeze_limiter(self.bot_pos)
        
        self.bot = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.bot.position = self.bot_pos
        self.bot_seg = pymunk.Segment(self.bot, a=(-length, 0), b=(length, 0), radius=1)
        self.bot_seg.friction = 0
        self.bot_seg.filter = Gripper.filter
        self.bot.angle = angle
        
        # top gripper
        self.top_vel = -Vec2d(self.vel_x, self.vel_y) # squeeze velocity vector
        self.top_pos = Vec2d(x+self.x_offset, y+self.y_offset) # starting position vector
        self.top_limiter = Gripper.make_unsqueeze_limiter(self.top_pos)
        
        self.top = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.top.position = self.top_pos
        self.top_seg = pymunk.Segment(self.top, a=(-length, 0), b=(length, 0), radius=1)
        self.top_seg.friction = 0
        self.top_seg.filter = Gripper.filter
        self.top.angle = angle
    
    @staticmethod
    def make_unsqueeze_limiter(pos, eps=1):
        def position_func(body, dt):
            if (body.position - pos).length < eps:
                body.velocity = 0, 0
            pymunk.Body.update_position(body, dt)
                
        return position_func
    
    def make_squeeze_limiter(self, polygon):
        rel_angle = (self.angle % (2*np.pi) - polygon.body.angle % (2*np.pi)) % (2*np.pi)
            
        for i in range(len(ranges)-1):
            if ranges[i] <= rel_angle < ranges[i+1]:
                assert ranges[i] <= minima[i] <= ranges[i+1]
                min_dist = diameter_func(minima[i])
                
        def position_func(body, dt):
            if self.distance() < min_dist:
                self.stop()
                
            pymunk.Body.update_position(body, dt)
        
        return position_func
                
    def squeeze(self):
        self.bot.velocity = self.bot_vel
        self.top.velocity = self.top_vel
        
    def unsqueeze(self):
        self.bot.velocity = -self.bot_vel
        self.top.velocity = -self.top_vel
    
    def stop(self):
        self.bot.velocity = 0, 0
        self.top.velocity = 0, 0
        
    def limit_unsqueeze(self):
        self.bot.position_func = self.bot_limiter
        self.top.position_func = self.top_limiter
        
    def limit_squeeze(self, polygon):
        squeeze_limiter = self.make_squeeze_limiter(polygon)
        self.bot.position_func = squeeze_limiter
        self.top.position_func = squeeze_limiter
        
    def reset_pos_func(self):
        self.bot.position_func = pymunk.Body.update_position
        self.top.position_func = pymunk.Body.update_position
        
    def reset_vel_func(self):
        self.bot.velocity_func = pymunk.Body.update_velocity
        self.top.velocity_func = pymunk.Body.update_velocity
        
    def distance(self):
        return self.bot.position.get_distance(self.top.position)


class Polygon:
    
    squeeze_filter = pymunk.ShapeFilter(categories=0b10, mask=pymunk.ShapeFilter.ALL_MASKS())
    move_filter = pymunk.ShapeFilter(categories=0b10, mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b01)
    
    gripper_pos = []
    del_pos = math.inf
    state = 0
    
    def __init__(self, x, y, points, display, angle=None):
        self.points = list(map(tuple, points))
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.position = x, y
        
        self.poly = pymunk.Poly(self.body, self.points, radius=.5)
        self.poly.mass = 1e3
        self.poly.friction = 0
        self.poly.elasticity = 0
        
        if angle is None:
            self.body.angle = random.uniform(0, 2*math.pi)
            
        self.display = display
        self.pos_limiter = self.make_pos_limiter()
        
    def reset_pos_func(self):
        self.body.position_func = pymunk.Body.update_position
        
    def reset_vel_func(self):
        self.body.velocity_func = pymunk.Body.update_velocity
        
    def move(self):
        self.body.velocity = 100, 0
        self.poly.filter = Polygon.move_filter
        # self.body.velocity_func = Polygon.constant_angular_velocity
        self.body.position_func = self.pos_limiter
        
    def squeeze(self, gripper):
        self.poly.filter = Polygon.squeeze_filter
        # self.reset_vel_func()
        
        def velocity_func(body, gravity, damping, dt):
            top = gripper.top_seg
            bot = gripper.bot_seg
            top_collide_shape = self.poly.segment_query(top.a + top.body.position, 
                                                        top.b + top.body.position, top.radius).shape
            bot_collide_shape = self.poly.segment_query(bot.a + bot.body.position, 
                                                        bot.b + bot.body.position, bot.radius).shape
            
            if top_collide_shape is None or bot_collide_shape is None:
                body.moment = 1e100 # math.inf
            else:
                body.moment = 1e6
            
            pymunk.Body.update_velocity(body, gravity, damping, dt)
            
        self.body.velocity_func = velocity_func

    def make_pos_limiter(self):
        def position_func(body, dt):
            if int(body.position.x) in Polygon.gripper_pos and Polygon.state >= 50:
                body.velocity = body.velocity * 0
            if body.position.x >= Polygon.del_pos:
                self.display.space.remove(body, *body.shapes)
                self.display.polygons.pop()

            pymunk.Body.update_position(body, dt)
            
        return position_func
        
    @staticmethod
    def constant_angular_velocity(body, gravity, damping, dt):
        pymunk.Body.update_velocity(body, gravity, damping, dt)
        body.angular_velocity = 0
        body.velocity = 0, 0


class Display:
    
    MOVE_PART_TIME = 200
    SQUEEZE_PART_TIME = 150
    UNSQUEEZE_PART_TIME = 150
    TOTAL_TIME = MOVE_PART_TIME + SQUEEZE_PART_TIME + UNSQUEEZE_PART_TIME
    
    def __init__(self, points, angles):
        self.angles = angles
        self.points = points
        
        self.fig = plt.figure(figsize=(8, 5), tight_layout=True)
        
        self.gripper_pos = [i * 250 for i in range(1, len(angles)+1)]
        self.start_pos = 0
        self.del_pos = (len(angles) + 1) * 250
        
        self.xlim = (self.start_pos, self.del_pos)
        self.ylim = (-200, 200)
        
        self.ax = plt.axes(xlim=self.xlim, ylim=self.ylim)
        self.ax.set_aspect('equal')
        
        self.space = pymunk.Space(threaded=True)
        self.space.threads = 2
        self.space.gravity = 0, 0
        self.space.damping = 1
        
        self.init_grippers()    
        self.polygons = []
        self.grippers_min_dist = []
    
        self.do = pymunk.matplotlib_util.DrawOptions(self.ax)
        
        Polygon.gripper_pos = self.gripper_pos
        Polygon.del_pos = self.del_pos
        
    def init_grippers(self):
        self.grippers = []
        
        for angle, xpos in zip(self.angles, self.gripper_pos):
            g = Gripper(xpos, 0, angle)
            
            self.grippers.append(g)
            self.space.add(g.top, g.top_seg)
            self.space.add(g.bot, g.bot_seg)
    
    def add_polygon(self):
        p = Polygon(self.start_pos, 0, self.points, self)
        self.polygons.insert(0, p)
        self.space.add(p.body, p.poly)
        
    def make_animation(self, frames=None):
        animate = self.make_animate_func()
        init = lambda: self.space.debug_draw(self.do)
        if frames is None:
            frames = Display.TOTAL_TIME
        self.anim = animation.FuncAnimation(self.fig, animate, init_func=init, 
                                            frames=frames, interval=20, blit=False)    
        
        return self.anim
    
    @print_errors_to_stdout
    def make_animate_func(self):     
        def animate(dt):
            dt = dt % 500
            Polygon.state = dt
            if dt == 0:
                # init move phase
                # create a new box; start moving all boxes to next one
                self.add_polygon()

                for p in self.polygons:
                    p.reset_vel_func()
                    p.move()
            elif 0 < dt < 200:
                # move phase
                pass
            elif dt == 200: 
                # init squeeze phase
                self.grippers_min_dist = [50] * len(self.grippers)
                for i, p in enumerate(self.polygons[:len(self.grippers)]):
                    p.squeeze(self.grippers[i])
                    rel_angle = (self.grippers[i].angle % (2*np.pi) - p.body.angle % (2*np.pi)) % (2*np.pi)
            
                    for j in range(len(ranges)-1):
                        if ranges[j] <= rel_angle < ranges[j+1]:
                            assert ranges[j] <= minima[j] <= ranges[j+1]
                            self.grippers_min_dist[i] = diameter_func(minima[j])

                for g in self.grippers:  
                    g.reset_pos_func()
                    g.squeeze()          
            elif 200 < dt < 350:
                # squeeze phase
                self.space.damping = 0
                for i, g in enumerate(self.grippers): 
                    distance = g.distance()
                    if distance < self.grippers_min_dist[i]:
                        g.stop()
                    
            elif dt == 350:
                # init unsqueeze phase
                self.space.damping = 1
                for p in self.polygons:
                    p.body.velocity_func = Polygon.constant_angular_velocity

                for g in self.grippers:
                    g.unsqueeze()
                    g.limit_unsqueeze()
            elif dt > 350:
                # unsqueeze phase
                pass

            for x in range(10):
                self.space.step(1/50/10)
            self.ax.clear()
            self.ax.set_xlim(*self.xlim)
            self.ax.set_ylim(*self.ylim)
            self.space.debug_draw(self.do)

            self.ax.set_title(f'{dt}|{[(round(p.body.position.x, 2), round(p.body.position.y, 2)) for p in self.polygons]}')
#             points_top = self.polygons[0].poly.shapes_collide(self.grippers[0].top_seg).points
#             points_bot = self.polygons[0].poly.shapes_collide(self.grippers[0].bot_seg).points
#             self.ax.set_title(f'{len(points_top)} | {len(points_bot)}')
            
#             top = self.grippers[0].top_seg
#             bot = self.grippers[0].bot_seg
#             points_top = self.polygons[0].poly.segment_query(top.a + top.body.position, top.b + top.body.position, top.radius).shape
#             points_bot = self.polygons[0].poly.segment_query(bot.a + bot.body.position, bot.b + bot.body.position, bot.radius).shape
#             self.ax.set_title(f'{points_top} | {points_bot}')
            self.ax.set_title(f'{self.polygons[0].body.moment} | {self.polygons[0].body.center_of_gravity}')
            
        return animate