"""
This contains the main parts of the part feeder algorithm
"""

import numpy as np

class Polygon:

    """Polygon class represents a polygon through points on a cartesian coordinate system. This
    class assumes all vertices are integers."""

    def __init__(self, points, scale_and_center=True):
        assert type(points) in [list, np.ndarray]

        points = np.array(points).astype(np.int32)

        assert len(points.shape) == 2 and points.shape[1] == 2

        if all(points[0] == points[len(points)-1]):
            points = points[:-1]

        if scale_and_center:
            points = Polygon.scale_polygon(points)

            C_x, C_y = Polygon.centroid(Polygon.convex_hull(points))

            points[:, 0] -= round(C_x)
            points[:, 1] -= round(C_y)

        self.points = points
        self.n = len(points)
        self.convex_hull = Polygon.convex_hull(points)

        # By default, the centroid given is the centroid of the convex hull
        self.C_x, self.C_y = Polygon.centroid(self.convex_hull)

    @staticmethod
    def centroid(points):
        """Returns the centroid of a polygon"""
        n = len(points)
        def next(i):
            return (i + 1) % n

        shoelace = [points[i, 0]*points[next(i), 1] - points[next(i), 0]*points[i, 1] for i in range(n)]
        list_x = [(points[i, 0] + points[next(i), 0])*shoelace[i] for i in range(n)]
        list_y = [(points[i, 1] + points[next(i), 1])*shoelace[i] for i in range(n)]
        
        const = 1/(6*Polygon.signed_area(points))
        C_x = const * sum(list_x)
        C_y = const * sum(list_y)
        
        return C_x, C_y
    
    @staticmethod
    def signed_area(points):
        """Returns the signed area of a polygon as described by the shoelace formula"""
        n = len(points)
        def next(i):
            return (i + 1) % n
        
        res = sum([points[i, 0]*points[next(i), 1] - points[next(i), 0]*points[i, 1] for i in range(n)])
        return res/2

    @staticmethod
    def scale_polygon(points, max_dim=100):
        """
        Scales the polygon so that the largest dimension is approximately max_dim units
        
        Units are retained as integers, so there is chance for slight deviations in the points.
        """
        width = max(points[:, 0]) - min(points[:, 0])
        height = max(points[:, 1]) - min(points[:, 1])
        
        scale = max_dim/max(width, height)
        
        return np.around(points * scale).astype(np.int32)

    @staticmethod
    def convex_hull(points):
        """
        Returns the convex hull of a set of points, which defines a convex polygon. 
        The returned points form a counterclockwise sequence.
        
        This is an implementation of Jarvis's march algorithm, which runs in O(nh) time.
        """
        assert len(points) >= 3
        
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
                d = Polygon.triangle_signed_area(points[p], points[i], points[q])
                if d > 0 or (d == 0 and np.linalg.norm(v1) > np.linalg.norm(v2)):
                    q = i
                    
            p = q
            if p == l_idx:
                break
            result.append(points[q])
            
        return np.array(result)

    @staticmethod
    def triangle_signed_area(p1, p2, p3):
        """
        Returns the twice the signed area of a triangle defined by the integer points (p1, p2, p3).
        The sign is positive if and only if (p1, p2, p3) form a counterclockwise cycle
        (a left turn). If the points are colinear, then this returns 0. If the points form
        a clockwise cycle, this returns a negative value.
        
        This method is described in further detail in Preparata and Shamos (1985). 
        """
        mat = np.hstack((np.vstack((p1, p2, p3)), np.ones((3, 1)))).astype('int32')

        # since matrix only has integers, determinant should itself be an integer
        return round(np.linalg.det(mat)) 

    @staticmethod
    def antipodal_pairs(points):
        """
        Returns the antipodal pairs of a convex polygon. The points must be in
        a counterclockwise sequence.
        
        This procedure is described in further detail in Preparata and Shamos (1985).
        """
        res = []
        n = len(points)
        def next(i):
            return (i + 1) % n
        
        p = n - 1
        q = next(p)
        while Polygon.triangle_signed_area(points[p], points[next(p)], points[next(q)]) > \
              Polygon.triangle_signed_area(points[p], points[next(p)], points[q]):
            q = next(q)
            
        p0, q0 = 0, q

        while q != p0:
            # print(res)
            p = next(p)
            res.append([p, q])
            while Polygon.triangle_signed_area(points[p], points[next(p)], points[next(q)]) > \
                  Polygon.triangle_signed_area(points[p], points[next(p)], points[q]):
                q = next(q)
                if (p, q) != (q0, p0): # and sorted([p, q]) not in res:
                    res.append([p, q])
                else:
                    break
            if Polygon.triangle_signed_area(points[p], points[next(p)], points[next(q)]) == \
               Polygon.triangle_signed_area(points[p], points[next(p)], points[q]):
                if (p, q) != (q0, n-1): # and sorted([p, q]) not in res:
                    res.append([p,next(q)])
                else:
                    break
                    
        return np.array(res)

if __name__ == '__main__':
    # a few tests
    # points = np.array([(-42, -41), (48, -41), (39, 25), (-34, 59)]) # 4gon in paper
    points = np.array([[233, 360], [131, 268], [147, 164], [243, 123], [318, 300], [245, 292], [233, 360]]) # another test shape

    poly = Polygon(points)

    print(poly.points)
    print(poly.convex_hull)
    print(Polygon.centroid(poly.convex_hull))
    print(Polygon.antipodal_pairs(poly.convex_hull))
    print(Polygon.convex_hull(poly.points))