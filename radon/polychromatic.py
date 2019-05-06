import scipy.stats
import numpy as np
from . import ctsimulation


def radon(img, angle_rad, center=None):
    """
    Calculate Radon transform of stack of images.
    """
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

def poly_sinogram(sinograms, src_intensity_per_bin_keV, src_keV, poisson=False, quantization_bits=None, min_energy_keV=None):
    """Calculate polyenergetic sinogram from monoenergetic sinograms in several energy intervals.
    
    Args:
        sinograms: sinogram for each energy bin
        src_intensity_per_bin_keV: source intensity per pixel for each energy bin (total in bin)
        src_keV: center of each energy bin
        poisson (bool): whether to generate photon Poisson noise
        quantization_bits: (optional) number of bits for simulated detector quantization (default: no quantization)
        min_energy_keV: (optional) clipping value for sinogram (default: 1e-3*src_keV[0], i.e. a thousandth of a photon)
    
    Returns:
        np.ndarray: polyenergetic sinogram (unitless)
    """
    
    if len(sinograms) != len(src_intensity_per_bin_keV) or len(sinograms) != len(src_keV):
        raise Exception("API error: len(sinograms), len(src_intensity_per_bin_keV) and len(src_keV) should be equal.")
    
    if quantization_bits is not None:
        raise Exception("Detector quantization is not implemented.")
    
    transmission = np.exp(-sinograms)
    src_num_photons_per_bin = src_intensity_per_bin_keV / src_keV # YES
    
    observed_photons = transmission * src_num_photons_per_bin[:,None,None] # YES
    if poisson:
        observed_photons = scipy.stats.poisson.rvs(observed_photons)
    
    observed_total_energy_keV = np.sum(observed_photons*src_keV[:,None,None], axis=0) # keV per pixel
    # YES
    
    flood_total_energy_keV = np.sum(src_num_photons_per_bin * src_keV) # YES
    
    if min_energy_keV is None:
        min_energy_keV = 1e-3*src_keV[0]
    poly_sinogram = np.log(flood_total_energy_keV) - np.log(np.maximum(observed_total_energy_keV, min_energy_keV))
    
    return poly_sinogram