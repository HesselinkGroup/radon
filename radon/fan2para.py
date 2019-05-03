import numpy as np

# Coordinate transformation

def map_fan2para(phi_fan, theta_src, d_src_iso):
    """
    Convert fan beam coordinates (phi, theta_src) to parallel beam coordinates (r, theta).
    """
    return -d_src_iso*np.sin(phi_fan), theta_src + phi_fan

def map_para2fan(r, theta, d_src_iso):
    """
    Convert parallel beam coordinates (r, theta) to fan beam coordinates (phi, theta_src).
    """
    phi = np.arcsin(-r / d_src_iso)
    return phi, theta - phi


