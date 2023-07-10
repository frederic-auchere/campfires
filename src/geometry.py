import numpy as np


class Point:
    """
    Represents a point in 3D space.
    """

    def __init__(self, *args, rect=True):
        """
        :param args: input coordinates (3 values).
        :param rect: if True (default), the input coordinates are assumed to be cartesian, otherwise they are assumed to
         be polar and converted to cartesian.
        """
        if not rect:
            self.x, self.y, self.z = self.to_rect(*args)
        else:
            self.x, self.y, self.z = args

    @staticmethod
    def to_rect(lon, lat, r):
        """
        Converts the point coordinates from polar to rectangular.

        :param lon: longitude (in radians)
        :param lat: latitude (in radins)
        :param r: radius
        :return: 3-tuple of the rectangular coordinates
        """
        x = r*np.cos(lat)*np.cos(lon)
        y = r*np.cos(lat)*np.sin(lon)
        z = r*np.sin(lat)
        return x, y, z

    def mean_with(self, pt):
        """
        Computes the mean of the point with another one. The mean point is defined as the one whose
        cartesian coordinates are each the average of those of the two points.

        :param pt: other Point object
        :return: new Point object of averaged cartesian coordinates
        """
        return Point((self.x + pt.x)/2, (self.y + pt.y)/2, (self.z + pt.z)/2)

    def dist_from(self, pt):
        """
        Computes the distance from another point.

        :param pt:  other Point object.
        :return: distance
        """
        dx = self.x - pt.x
        dy = self.y - pt.y
        dz = self.z - pt.z
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def __add__(self, p):
        """
        Adds another point. The results is defined as the point whose rectangular coordinates are the sum
        of those of the two points.

        :param p: other point.
        :return: new point
        """
        return Point(self.x + p.x, self.y + p.y, self.z + p.z)

    def __sub__(self, p):
        """
        Subtracts another point. The results is defined as the point whose rectangular coordinates are the subtraction
        of those of the two points.

        :param p: other Point object.
        :return: new Point object.
        """
        return Point(self.x - p.x, self.y - p.y, self.z - p.z)


class Vector:
    """
    Represents a vector in 3D space. A vector is defined by either its cartesian coordinates, or two points.
    """

    def __init__(self, *args):
        """
        :param args: either two Point objects, or coordinates (3 values)
        """
        if len(args) == 2:
            self.x, self.y, self.z = self.unit(*args)
        else:
            self.x, self.y, self.z = args

    def __mul__(self, s):
        """
        Multiplies the vector coordinates by a given constant.

        :param s: scaling constant.
        :return:  new Vector object.
        """
        return Vector(self.x*s, self.y*s, self.z*s)

    @staticmethod
    def unit(*args):
        """
        Computes a unit vector from two points.

        :param args: two Point objects
        :return: 3-tuple of the vector cartesian coordinates
        """
        pt1, pt2 = args
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        dz = pt2.z - pt1.z
        length = np.sqrt(dx**2 + dy**2 + dz**2)
        return dx/length, dy/length, dz/length


class Line:
    """
    Represents a line in 3D space. A line is defined by two points.
    """

    def __init__(self, pt1, pt2):
        """
        :param pt1: first Point object
        :param pt2: second Point object
        """
        self.p = pt1
        self.v = Vector(pt1, pt2)

    def find_closest_points(self, line):
        """
        Computes the properties of the shortest segment jointing the line to another line.

        :param line: other Line object.
        :return: 4-tuple containting:
            - Point at one end of the shortest segment (belong to first line).
            - Point at other end of the shortest segment (belong to second line).
            - middle Point (average of the two ends)
            - distance between the two ends.
        """
        n = np.cross((self.v.x, self.v.y, self.v.z), (line.v.x, line.v.y, line.v.z))
        n2 = np.cross((line.v.x, line.v.y, line.v.z), n)
        n1 = np.cross((self.v.x, self.v.y, self.v.z), n)
        dp = line.p - self.p
        c1 = self.p + self.v*(np.dot((dp.x, dp.y, dp.z), n2)/np.dot((self.v.x, self.v.y, self.v.z),  n2))
        dp = self.p - line.p
        c2 = line.p + line.v*(np.dot((dp.x, dp.y, dp.z), n1)/np.dot((line.v.x, line.v.y, line.v.z),  n1))

        return c1, c2, c1.mean_with(c2), c1.dist_from(c2)
