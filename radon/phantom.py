import numpy as np

class Phantom:
    """
    Polyenergetic two-dimensional x-ray phantom.  It knows its pixel coordinates and can add ellipses.
    """
    def __init__(self, xs, ys, num_energies):
        nx = len(xs)
        ny = len(ys)
        self.x = xs
        self.y = ys
        self.num_energies = num_energies
        self.xx, self.yy = np.meshgrid(xs,ys)
        self.img = np.zeros((num_energies, nx, ny))
        
    def ellipse_mask(self, center_xy, radius_xy):
        return (self.xx-center_xy[0])**2/radius_xy[0]**2 + (self.yy-center_xy[1])**2/radius_xy[1]**2 <= 1.0
    
    def _standardize_mu(self, mu):
        if np.isscalar(mu):
            mu = np.full((num_energies), mu)
        elif len(mu) != self.num_energies:
            raise Exception(f"mu must be a scalar or {self.num_energies}-element array for this {self.num_energies}-energy phantom.")
        return mu
    
    def _check_center_radius(self, center, radius):
        if np.alen(center) != 2:
            raise Exception("center must have two elements (x0, y0)")
        if np.alen(radius) != 2:
            raise Exception("radius must have two elements (rx, ry)")
        
    def add_ellipse(self, center, radius, mu):
        mu = self._standardize_mu(mu)
        self._check_center_radius(center, radius)
        self.img[:,self.ellipse_mask(center, radius)] = mu[:,np.newaxis]
    

        