import numpy as np

class DPCPhantom:
    """
    Polyenergetic two-dimensional x-ray phantom.  It knows its pixel coordinates and can add ellipses.
    """
    def __init__(self, xs, ys, keV):
        nx = len(xs)
        ny = len(ys)
        self.x = xs
        self.y = ys
        self.xx, self.yy = np.meshgrid(xs,ys)

        self.keV = np.atleast_1d(keV)

        self.mu = np.zeros((len(self.keV), nx, ny))
        self.delta = np.zeros_like(self.mu)
        self.scattering = np.zeros_like(self.mu)
        self.density = np.zeros((nx,ny))
    
    def max_ellipse_mask(self):
        """Returns largest inset elliptical mask.
        """
        rx = (self.x[-1] - self.x[0])/2
        ry = (self.y[-1] - self.y[0])/2
        mask = self.ellipse_mask([self.x.mean(), self.y.mean()], [rx, ry])
        return mask

    def ellipse_mask(self, center_xy, radius_xy):
        return (self.xx-center_xy[0])**2/radius_xy[0]**2 + (self.yy-center_xy[1])**2/radius_xy[1]**2 <= 1.0

    def rectangle_mask(self, center_xy, half_width_xy):
        return (np.abs(self.xx-center_xy[0]) < half_width_xy[0]) * (np.abs(self.yy-center_xy[1]) < half_width_xy[1])
    
    def _check_center_radius(self, center, radius):
        if np.alen(center) != 2:
            raise Exception("center must have two elements (x0, y0)")
        if np.alen(radius) != 2:
            raise Exception("radius must have two elements (rx, ry)")

    def apply_circle_mask(self):
        mask = np.logical_not(self.max_ellipse_mask())
        self.mu[:,mask] = 0.0
        self.delta[:,mask] = 0.0
        self.scattering[:,mask] = 0.0
        self.density[mask] = 0.0
    
    def _add_masked(self, mask, material):
        self.mu[:,mask] = material.mu(self.keV)[:,np.newaxis]
        self.delta[:,mask] = material.delta(self.keV)[:,np.newaxis]
        self.density[mask] = material.density

    def add_ellipse(self, center, radius, material):
        center = np.atleast_1d(center)
        radius = np.atleast_1d(radius)
        self._check_center_radius(center, radius)
        self._add_masked(self.ellipse_mask(center,radius), material)

    def add_rectangle(self, center, half_width, material):
        center = np.atleast_1d(center)
        half_width = np.atleast_1d(half_width)
        self._check_center_radius(center, half_width)
        self._add_masked(self.rectangle_mask(center, half_width), material)
    

        