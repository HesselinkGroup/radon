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


def filtered_backproject(radon_tx, angles, center=None):
    fradon = np.fft.fft(radon_tx, axis=0)
    freqs = np.fft.fftfreq(radon_tx.shape[0])
    sinogram_sharp = np.fft.ifft(np.abs(freqs)[:,np.newaxis] * fradon, axis=0)
    zz = backproject(sinogram_sharp.real, angles, center=center) * np.pi / len(angles)
    
    # Volume correction:
    
    dc_radon = radon_tx.ravel().sum() / radon_tx.shape[0]
    dc_zz = zz.ravel().sum()
    zz += (dc_radon - dc_zz) / radon_tx.shape[0]**2
    
    return zz

