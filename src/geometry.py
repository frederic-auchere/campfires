import numpy as np


class Point:

    def __init__(self, *args, rect=True):
        if not rect:
            self.x, self.y, self.z = self.to_rect(*args)
        else:
            self.x, self.y, self.z = args

    @staticmethod
    def to_rect(lon, lat, r):
        x = r*np.cos(lat)*np.cos(lon)
        y = r*np.cos(lat)*np.sin(lon)
        z = r*np.sin(lat)
        return x, y, z

    def mean_with(self, pt):
        return Point((self.x + pt.x)/2, (self.y + pt.y)/2, (self.z + pt.z)/2)

    def dist_from(self, pt):
        dx = self.x - pt.x
        dy = self.y - pt.y
        dz = self.z - pt.z
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def __add__(self, p):
        return Point(self.x + p.x, self.y + p.y, self.z + p.z)

    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y, self.z - p.z)


class Vector:

    def __init__(self, *args):
        if len(args) == 2:
            self.x, self.y, self.z = self.unit(*args)
        else:
            self.x, self.y, self.z = args

    def __mul__(self, s):
        return Vector(self.x*s, self.y*s, self.z*s)

    @staticmethod
    def unit(*args):
        pt1, pt2 = args
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        dz = pt2.z - pt1.z
        length = np.sqrt(dx**2 + dy**2 + dz**2)
        return dx/length, dy/length, dz/length


class Line:

    def __init__(self, pt1, pt2):
        self.p = pt1
        self.v = Vector(pt1, pt2)

    def find_closest_points(self, line):
        n = np.cross((self.v.x, self.v.y, self.v.z), (line.v.x, line.v.y, line.v.z))
        n2 = np.cross((line.v.x, line.v.y, line.v.z), n)
        n1 = np.cross((self.v.x, self.v.y, self.v.z), n)
        dp = line.p - self.p
        c1 = self.p + self.v*(np.dot((dp.x, dp.y, dp.z), n2)/np.dot((self.v.x, self.v.y, self.v.z),  n2))
        dp = self.p - line.p
        c2 = line.p + line.v*(np.dot((dp.x, dp.y, dp.z), n1)/np.dot((line.v.x, line.v.y, line.v.z),  n1))

        return c1, c2, c1.mean_with(c2), c1.dist_from(c2)
