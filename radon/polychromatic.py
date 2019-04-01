import numpy as np
from . import ctsimulation


def radon(img, angle_rad, center=None):

    if img.ndim < 3:
        raise Exception("img must have shape (num_energies, n, n)")

    num_energies = img.shape[0]
    num_angles = len(angle_rad)
    n = img.shape[1]

    sino_shape = (num_energies, n, num_angles)
    sinogram = np.zeros(sino_shape)

    for ii in range(num_energies):
        sinogram[ii,:,:] = ctsimulation.radon(img[ii,:,:], angle_rad, center)

    return sinogram


def _general_backproject(fun, sinogram, angle_rad, center=None):

    if sinogram.ndim < 3:
        raise Exception("sinogram must have shape (num_energies, n, num_angles)")

    num_energies = sinogram.shape[0]
    num_angles = sinogram.shape[2]
    n = sinogram.shape[1]

    img_shape = (num_energies, n, n)
    img = np.zeros(img_shape)

    for ii in range(num_energies):
        img[ii,:,:] = fun(sinogram[ii,:,:], angle_rad, center)

    return img

def backproject(sinogram, angle_rad, center=None):
    return _general_backproject(ctsimulation.backproject, sinogram, angle_rad, center)

def filtered_backproject(sinogram, angle_rad, center=None):
    return _general_backproject(ctsimulation.filtered_backproject, sinogram, angle_rad, center)


