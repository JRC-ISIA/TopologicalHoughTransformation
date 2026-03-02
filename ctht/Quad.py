import logging
import numpy as np

class Quad(object):

    def __init__(self, depth, bounding_box=None):
        self.depth = depth
        self.bounding_box = bounding_box
        self.neighbors = []

        self._center_function_value = None
        self._index = None
        self.active = True
        self.diag = None

        self.get_quad_diag()

    @property
    def t0(self):
        return self.bounding_box[0]

    @property
    def t1(self):
        return self.bounding_box[1]

    @property
    def r0(self):
        return self.bounding_box[2]

    @property
    def r1(self):
        return self.bounding_box[3]

    @property
    def index(self):
        if self._index is not None:
            return self._index  # Getter
        raise ValueError("Index not set.")

    @index.setter
    def index(self, index):
        if self._index is None:
            self._index = index  # Setter
        else:
            raise ValueError("Index already set.")

    def add_neighbor(self, neighbor):
        while neighbor not in self.neighbors:
            self.neighbors.append(neighbor)

    def remove_neighbor(self, neighbor):
        while neighbor in self.neighbors:
            self.neighbors.remove(neighbor)

    @property
    def center_function_value(self):
        if self._center_function_value is not None:
            return self._center_function_value
        raise ValueError("Center value not set.")

    @center_function_value.setter
    def center_function_value(self, value):
        if self._center_function_value is None:
            self._center_function_value = value
        else:
            raise ValueError("Center value already set.")

    def get_center(self):
        t_center = (self.t0 + self.t1) / 2
        r_center = (self.r0 + self.r1) / 2
        return t_center, r_center

    def get_quad_diag(self):
        self.diag = np.hypot((self.t1 - self.t0), (self.r1 - self.r0))

    def subdivide(self):
        self.active = False
        t_mid, r_mid = self.get_center()

        # weights for being more generic, however redundant here

        new_bb = [
            [self.t0, t_mid, self.r0, r_mid],
            [t_mid, self.t1, self.r0, r_mid],
            [self.t0, t_mid, r_mid, self.r1],
            [t_mid, self.t1, r_mid, self.r1]
        ]

        ddepth = self.depth + 1
        quads = [
            Quad(ddepth, bounding_box=new_bb[0]),
            Quad(ddepth, bounding_box=new_bb[1]),
            Quad(ddepth, bounding_box=new_bb[2]),
            Quad(ddepth, bounding_box=new_bb[3])
        ]

        # work on setting neighbors or other properties if needed
        for q in quads:
            q.neighbors = [quad for quad in quads if quad != q]
        
        # work on the neighbors inheritated from parent if needed
        for neighbor in self.neighbors[:]:
            # remove me from the neighbor's
            neighbor.remove_neighbor(self)
            # remove neighbor from my neighbors
            self.remove_neighbor(neighbor)

            # add new quads as a neighbor to the old neighbor and vice versa
            for q in quads:
                if q.is_quad_neighbors(neighbor):
                    neighbor.add_neighbor(q)
                    q.add_neighbor(neighbor)

        return quads
    

    def is_quad_neighbors(self, other):
        """Check if two quads are neighbors (share an edge)."""

        #  (x-condition) and not (y-condition)
        horizontally_touching = (self.t1 == other.t0 or self.t0 == other.t1) and \
                                not (self.r1 <= other.r0 or other.r1 <= self.r0)

        vertically_touching = (self.r1 == other.r0 or other.r1 == self.r0) and \
                              not (self.t1 <= other.t0 or other.t1 <= self.t0)

        diagonally_touching = (self.t1 == other.t0 or self.t0 == other.t1) and \
                              (self.r0 == other.r1 or self.r1 == other.r0)

        # moebius strip condition
        moebious_x_condition = (self.t1 == np.pi / 2 and other.t0 == -np.pi / 2) or \
                               (self.t0 == -np.pi / 2 and other.t1 == np.pi / 2)
        moebius_y_condition = (self.r1 == -other.r0 or other.r1 == -self.r0)
        moebius_touching = moebious_x_condition and moebius_y_condition

        return horizontally_touching or vertically_touching or \
            diagonally_touching or moebius_touching