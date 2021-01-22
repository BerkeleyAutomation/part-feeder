import numpy as np

import traceback, functools, pprint, math, random

def signed_area(p1, p2, p3):
    """
    Returns the twice the signed area of a triangle defined by the points (p1, p2, p3).
    The sign is positive if and only if (p1, p2, p3) form a counterclockwise cycle
    (a left turn). If the points are colinear, then this returns 0. If the points form
    a clockwise cycle, this returns a negative value.
    
    This method is described in further detail in Preparata and Shamos (1985). 
    """
    mat = np.hstack((np.vstack((p1, p2, p3)), np.ones((3, 1)))).astype('int32')
    return round(np.linalg.det(mat)) # since matrix only has integers, determinant should itself be an integer

def convex_hull(points):
    """
    Returns the convex hull of a set of points, which defines a convex polygon. 
    The returned points form a counterclockwise sequence.
    
    This is an implementation of Jarvis's march algorithm, which runs in O(nh) time.
    """
    assert len(points) >= 3, "Polygon must have at least 3 points"

    points = np.array(points)
    if all(points[0] == points[len(points)-1]):
        points = points[:-1]
    
    l_idx = np.argmin(points, axis=0)[0]
    l = points[l_idx]
    
    result = [l]
    start = 0
    
    p, q = l_idx, None
    while True:
        q = (p + 1) % len(points)
        
        for i in range(len(points)):
            if i == p:
                continue
            v1, v2 = points[i]-points[p], points[q]-points[i]
            d = signed_area(points[p], points[i], points[q])
            if d > 0 or (d == 0 and np.linalg.norm(v1) > np.linalg.norm(v2)):
                q = i
                
        p = q
        if p == l_idx:
            break
        result.append(points[q])
        
    return np.array(result)

def antipodal_pairs(points):
    """
    Returns the antipodal pairs of a convex polygon. The points must be in
    a counterclockwise sequence.
    
    This procedure is described in further detail in Preparata and Shamos (1985).
    """
    res = []
    n = len(points)
    def _next(i):
        return (i + 1) % n
    
    p = n - 1
    q = _next(p)
    while signed_area(points[p], points[_next(p)], points[_next(q)]) > \
          signed_area(points[p], points[_next(p)], points[q]):
        q = _next(q)
        
    p0, q0 = 0, q

    while q != p0:
        # print(res)
        p = _next(p)
        res.append([p, q])
        while signed_area(points[p], points[_next(p)], points[_next(q)]) > \
              signed_area(points[p], points[_next(p)], points[q]):
            q = _next(q)
            if (p, q) != (q0, p0): # and sorted([p, q]) not in res:
                res.append([p, q])
            else:
                break
        if signed_area(points[p], points[_next(p)], points[_next(q)]) == \
           signed_area(points[p], points[_next(p)], points[q]):
            if (p, q) != (q0, n-1): # and sorted([p, q]) not in res:
                res.append([p,_next(q)])
            else:
                break
                
    return np.array(res)

def make_diameter_function(convex_hull):
    """Returns a diameter function of a convex polygon"""
    pairs = antipodal_pairs(convex_hull)
    
    pieces = []
    n = len(convex_hull)
    for p1, p2 in pairs:
        v = convex_hull[p2] - convex_hull[p1]  # vector from p1 to p2
        
        v1, v2 = convex_hull[p1] - convex_hull[(p1+1)%n], convex_hull[(p1-1)%n] - convex_hull[p1]
        v3, v4 = convex_hull[(p2+1)%n] - convex_hull[p2], convex_hull[p2] - convex_hull[(p2-1)%n]

        a1, a2 = np.arctan2(*v1[::-1]) % (2*np.pi), np.arctan2(*v2[::-1]) 
        a3, a4 = np.arctan2(*v3[::-1]) % (2*np.pi), np.arctan2(*v4[::-1]) 
        
        max_angle = min(a1, a3) % np.pi
        length = np.linalg.norm(v)  # length of the vector (chord)
        initial_angle = np.arctan(v[1]/v[0] if v[0] != 0 else np.sign(v[1]) * np.inf) + np.pi/2  # angle of the parallel supporting 
                                                        # lines that are perpendicular to the chord
        pieces.append((max_angle, length, initial_angle))
        
    piecewise_func = []
    for i in range(len(pieces)):
        min_ = pieces[(i-1) % len(pieces)][0]
        max_ = pieces[i][0]
        if max_ < min_:
            # wrapped around
            piecewise_func.append((min_, np.pi) + pieces[i][1:])
            piecewise_func.append((0, max_) + pieces[i][1:])
        else:
            piecewise_func.append((min_, max_) + pieces[i][1:])
        
    # remove parallel edges
    piecewise_func = sorted([p for p in piecewise_func if not np.isclose(p[0], p[1])], key=lambda x: x[0])
    
    def diameter_func(theta):
        theta = theta % np.pi
        for p in piecewise_func:
            min_, max_, l, i = p
            if min_ <= theta < max_:
                return l * np.abs(np.cos(theta-i))
        
    return np.array(piecewise_func), diameter_func

def find_extrema(piecewise_func, diameter_func):
    """Returns the extrema (minima and maxima) of the diameter function. Used to compute the squeeze function."""
    maxima = []
    for p in piecewise_func:
        min_, max_, l, i = p
        if min_ <= i < max_:
            maxima.append(i)
            maxima.append(i+np.pi)
    maxima.sort()
    ranges = [0] + maxima + [2*np.pi]
    
    minima = []
    discont = np.append(piecewise_func[:, :2].flatten(), piecewise_func[:, :2].flatten()+np.pi)
    for i in range(len(ranges)-1):
        valid_points = discont[np.logical_and(ranges[i] <= discont, discont <= ranges[i+1])]
        minimum = min(valid_points, key=diameter_func)
        
        minima.append(minimum)
        # minima.append(minimum+np.pi)
        
    return np.array(maxima), np.array(minima)

def make_squeeze_func(piecewise_func, diameter_func):
    maxima, minima = find_extrema(piecewise_func, diameter_func)
    
    ranges = np.concatenate(([0], maxima, [2*np.pi]))
    def squeeze_func(theta):
        theta = theta % (2*np.pi)
        for i in range(len(ranges)-1):
            if ranges[i] <= theta < ranges[i+1]:
                assert ranges[i] <= minima[i] <= ranges[i+1]
                return minima[i]
            
        return 2*np.pi
    
    return squeeze_func

class SInterval:
    def __init__(self, a, b, image_min, image_max):
        assert a <= b
        assert image_min <= image_max or np.isclose(image_min, image_max)
        
        self.a = a
        self.b = b
        self.image_min = image_min
        self.image_max = image_max
        
        self.interval_m = self.b - self.a
        self.image_m = self.image_max - self.image_min
        
    def __repr__(self):
        return f'SInterval({round(self.a, 3)}, {round(self.b, 3)}, {round(self.image_min, 3)}, {round(self.image_max, 3)}, ' + \
               f'{round(self.interval_m, 3)}, {round(self.image_m, 3)})'

def period_from_r_fold(r):
    """
    Returns the period of a polygon's squeeze function given the n-fold (called r-fold in the paper) 
    rotational symmetry of the polygon
    """
    return 2*np.pi/(r*(1+r%2))


def detect_squeeze_periodicity(max_r=8):
    assert max_r >= 2, "All polygons have at least 1-fold rotational symmetry"
    
    res_r, res_T = 1, np.pi
    
    x = np.linspace(0, 2*np.pi, 1000)
    y = np.array([squeeze_func(t) for t in x])
    for r in range(2, max_r+1):
        T = period_from_r_fold(r)
        
        x_shift = x + T
        y_shift = np.array([squeeze_func(t) for t in x_shift]) % (2*np.pi)
        
        if all(np.isclose((y+T)%(2*np.pi), y_shift)):
            res_r, res_T = r, T
            
    return res_r, res_T

def centroid(points):
    """Returns the centroid of a polygon"""
    n = len(points)
    def next(i):
        return (i + 1) % n
    shoelace = [points[i, 0]*points[next(i), 1] - points[next(i), 0]*points[i, 1] for i in range(n)]
    list_x = [(points[i, 0] + points[next(i), 0])*shoelace[i] for i in range(n)]
    list_y = [(points[i, 1] + points[next(i), 1])*shoelace[i] for i in range(n)]
    
    const = 1/(6*polygon_signed_area(points))
    C_x = const * sum(list_x)
    C_y = const * sum(list_y)
    
    return C_x, C_y
    
def polygon_signed_area(points):
    """Returns the signed area of a polygon as described by the shoelace formula"""
    n = len(points)
    def next(i):
        return (i + 1) % n
    
    res = sum([points[i, 0]*points[next(i), 1] - points[next(i), 0]*points[i, 1] for i in range(n)])
    return res/2

def scale_polygon(points):
    """Scales the polygon so that the largest dimension is approximately 100 units"""
    width = max(points[:, 0]) - min(points[:, 0])
    height = max(points[:, 1]) - min(points[:, 1])
    
    larger = max(width, height)
    scale = 100/larger
    
    return np.around(points * scale).astype(np.int32)

def scale_and_center_polygon(points):
    points = scale_polygon(points)

    C_x, C_y = centroid(points)

    points[:, 0] -= round(C_x)
    points[:, 1] -= round(C_y)
    
    return points