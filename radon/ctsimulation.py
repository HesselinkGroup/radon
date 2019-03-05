import numpy as np
import skimage

def radon(img, angle_rad, center=None):

    # Rotation of zero means no rotation.
    # The projection direction then is along columns, i.e. axis=0.
    # So, the size of the sinogram is (img.shape[1], len(angle_rad)).
    
    if img.shape[0] != img.shape[1]:
        raise Exception("Image is not square as will be required for the inverse transform.")
    
    sinogram = np.empty((img.shape[1], len(angle_rad)))
    
    if np.isscalar(angle_rad):
        angle_rad = [angle_rad]
        
    for ii, aa in enumerate(angle_rad):
        sinogram[:,ii] = np.sum(skimage.transform.rotate(img, aa, center=center), axis=0)
    
    return sinogram

def backproject(sinogram, angle_rad, center=None):
    
    img_shape = (sinogram.shape[0], sinogram.shape[0])

    backprop = np.zeros(img_shape)
    
    for ii, aa in enumerate(angle_rad):
        backprop += skimage.transform.rotate(np.broadcast_to(sinogram[:,ii], img_shape), -aa, center=center)

    return backprop


def filtered_backproject(sinogram, angles, center=None):
    fradon = np.fft.fft(sinogram, axis=0)
    freqs = np.fft.fftfreq(sinogram.shape[0])
    sinogram_sharp = np.fft.ifft(np.abs(freqs)[:,np.newaxis] * fradon, axis=0)
    zz = backproject(sinogram_sharp.real, angles, center=center) * np.pi / len(angles)

    # Chop off the corners (pie are round, cornbread are square)
    xs = np.arange(zz.shape[0])
    xx,yy = np.meshgrid(xs,xs)
    x0 = zz.shape[0]/2.0
    zz[(xx-x0)**2 + (yy-x0)**2 > x0**2] = 0.0
    
    # Volume correction:
    
    dc_sinogram = np.mean(sinogram.sum(0))
    dc_zz = zz.sum()
    zz += (dc_sinogram - dc_zz) / zz.size
    
    return zz



def detector_value(num_photons, num_bits, peak_num_photons, clip=True):
    """
    Simulation of detector value.
    
    Parameters:
        num_photons: array-like
            actual number of photons hitting detector
        num_bits: integer
            bit depth of detector output
        peak_num_photons: scalar
            number of photons corresponding to highest detector output (2**num_bits - 1).
    
    Returns:
        x: array-like
            detector output, quantized in [0, 2**num_bits-1], clipped at the top
        
    TODO: Dark current (should it go here?)
    """
    
    x_max = 2**num_bits - 1

    x = (num_photons/(peak_num_photons+1)*(x_max+1)).astype(int)
    
    if clip:
        x = np.minimum(x_max, x)
    
    return x


def log_transform_quantized(x, num_bits):
    """
    Calculate log((2**num_bits-1) / max(x,1)).

    Parameters:
        x:        number(s) in [0, 2**num_bits-1]
        num_bits: number of bits!

    Returns: log((2**num_bits-1) / max(x,1))
    """
    x_clipped = np.maximum(1, x)
    
    ks = 1.0 # "system constant" in Mori 2013
    x_max = 2**num_bits - 1
    p_max = ks*np.log(x_max)
    p = p_max - ks*np.log(x_clipped)
    return p



