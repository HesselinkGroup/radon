"""
Simulate X-ray spectrum with beam hardening layers.
"""

import numpy as np
from xraymaterials import Material


class Spectrum:
    """
    X-ray spectrum filtering and binning class.
    
    The spectrum is a list of photon energies and corresponding source intensities.
    The source intensities are in units of power but unnormalized.
    The intensities are thus NOT photon counts: for this, divide intensity by photon energy!
    """
    def __init__(self, keV, intensity):
        """
        Create a spectrum from tabulated energies and intensities.
        """
        self.keV = keV
        self.src_intensity = intensity
        self.attenuation = np.ones_like(self.src_intensity)
        
    def add_layer(self, material, thickness_cm):
        """
        Add an attenuating layer of material.
        
        Parameters:
            material:     xraymaterials.Material to place in the beam path
            thickness_cm: thickness of material layer (cm)
        """
        atten = np.exp(-thickness_cm * material.mu(self.keV))
        self.attenuation *= atten
    
    def add_grating(self, materials, fill_factors, thickness_cm):
        """
        Add an attenuating square grating.  Its attenuation factor will be calculated
        by summing the energy transmitted through the thick and thin parts of the
        grating, assuming a uniform incident wave.
        
        Parameters:
            materials:    xraymaterials.Materials to place in the beam path
            fill_factors: fraction from 0 to 1 of materials
            thickness_cm: thickness of layer (cm)
        """
        
        f_total = np.sum(fill_factors) # total fill factor; remainder (vacuum) is 1-f_total
        mus = np.array([mat.mu(self.keV) for mat in materials])
        attens = np.sum(np.asarray(fill_factors) * np.exp(-thickness_cm * mus)) + (1-f_total)
        
        self.attenuation *= atten
        
    
#     def add_grating(self, material, fill_factor, thick_cm, thin_cm=0):
#         """
#         Add an attenuating square grating.  Its attenuation factor will be calculated
#         by summing the energy transmitted through the thick and thin parts of the
#         grating, assuming a uniform incident wave.
        
#         Parameters:
#             material:     xraymaterials.Material to place in the beam path
#             fill_factor:  fraction from 0 to 1 of each grating period that is "thick"
#             thick_cm:     thickness of material in "thick" part of the grating (cm)
#             thin_cm:      thickness of material in "thin" part of the grating (cm)
#         """
#         mu = material.mu(self.keV)
#         atten = fill_factor*np.exp(-thick_cm*mu) + (1-fill_factor)*np.exp(-thin_cm*mu)
#         self.attenuation *= atten
    
    @classmethod
    def from_photon_file(cls, filename, skiprows=3):
        """
        Create a Spectrum from a CSV file containing raw spectrum in PHOTON COUNT.
        
        The CSV file should have two columns: energy in keV, and number of photons (per something).
        
        Parameters:
            filename:    name of CSV file
            skiprows:    number of rows to skip before data begins (optional, default=3)
        
        Returns:
            new Spectrum object
        """
        spectrum_file = np.loadtxt(filename, skiprows=skiprows)
        keV = spectrum_file[:,0]
        num_photons = spectrum_file[:,1]
        intensity = num_photons * keV
        return cls(keV, intensity)
    
    def intensity(self, energy_keV=None):
        """
        Calculate intensity after transmission through material layers.
        
        Parameters:
            energy_keV: array of energies (keV) (optional)
        
        Returns:
            transmitted intensity
        """
        intensity = self.src_intensity * self.attenuation
        
        if energy_keV is not None:
            intensity = np.interp(energy_keV, self.keV, intensity)
        
        return intensity
    
    def binned_intensity(self, min_keV, max_keV, num_bins, op=None):
        """
        Calculate intensity after transmission through material layers, summed into bins by energy.
        
        Parameters:
            min_keV:   lowest energy of interest
            max_keV:   highest energy of interest
            num_bins:  number of equal-length energy bins
            op:        (optional) operation to perform on each bin.
                       Default is summation with np.sum.
                       Consider also trying np.mean.
        
        Returns:
            intensity_binned: summed intensity in each bin
            center_keV:       energy at the center of each bin
        """
        
        if op is None:
            op = np.sum
        
        intensity = self.src_intensity * self.attenuation
        
        bins = np.linspace(min_keV, max_keV, num_bins+1)
        inds = np.digitize(self.keV, bins)
        intensity_binned = np.array([op(intensity[inds==idx]) for idx in range(1,num_bins+1)])
        
        center_keV = 0.5*(bins[1:] + bins[:-1])
        
        return intensity_binned, center_keV
        

def make_standard_source():
    """
    Create Spectrum object for 90 keV DPC design, including 1 mm copper beam hardening
    layer and G0, G1 and G2 gratings.
    
    Yao-Te's description of G0 (gold pitch p0 = 2.538 um, silicon pitch = p0/2):
    1. Uniform thickness of bottom silicon thickness, 20 um
    2. Uniform thickness of gold with thickness p0/2
    3. 50% coverage of gold with thickness t_G0 (115 um)
    4. 25% coverage of silicon with thickness t_G0 (115 um)
    
    G1 and G2 are similar.  The G2 design is somewhat in flux.
    """
    
    # Nominal grating thickness (the height of the silicon teeth):
    t_g0 = 115e-4
    t_g1 = 19e-4
    t_g2 = 325e-4
    
    # Nominal grating pitches (pitch of the gold grating = thickness of gold layer)
    p_g0 = 2.538e-4
    p_g1 = 4.342e-4
    p_g2 = 15.013e-4
    
    # Uniform silicon thickness
    t_Si = 20e-4 # G0, G1
    t_Si_g2 = 150e-4
    
    # Beam hardening layer, 1 mm copper
    t_Cu = 0.1
    
    # Beryllium window, .8 mm
    t_Be = 0.08
    
    # All gratings have 50% fill factor.
    # I feel like something is fishy about the effective transmission calculation.
    # It seems to rely on having no phase relationship between G0, G1 and G2.
    
    gold = Material.from_element("Au")
    copper = Material.from_element("Cu")
    silicon = Material.from_element("Si")
    beryllium = Material.from_element("Be")
    
    src = Spectrum.from_photon_file("180kVp_02042018.txt")
    
    # I have to do the gratings by hand... I didn't make a good enough API yet.
    mu_au = gold.mu(src.keV)
    mu_si = silicon.mu(src.keV)
    
    def grating_attenuation(p, t):
        # Column 1: lots of vacuum, thin layer of gold; f = 0.25
        column_1 = np.exp(-(p/2)*mu_au)
        
        # Column 2: thin layer of gold, lots of silicon; f = 0.25
        column_2 = np.exp(-(p/2)*mu_au - t*mu_si)
        
        # Column 3: thick layer of gold; f = 0.5
        column_3 = np.exp(-(t+p/2)*mu_au)
        
        atten_tot = 0.25*column_1 + 0.25*column_2 + 0.5*column_3
        
        return atten_tot
    
    src.attenuation *= grating_attenuation(p_g0, t_g0)
    src.attenuation *= grating_attenuation(p_g1, t_g1)
    src.attenuation *= grating_attenuation(p_g2, t_g2)
    
    # Uniform silicon layers (all three together)
    src.add_layer(silicon, t_Si + t_Si + t_Si_g2)
    
    # Copper beam-hardening layer
    src.add_layer(copper, t_Cu)
    
    # Beryllium window
    src.add_layer(beryllium, t_Be)
    
    return src


def create_linearization_table(material, src_keV, src_intensity, design_keV):
    mu = material.mu(src_keV)
    design_mu = material.mu(design_keV)
    
    complete_attenuation_level = 1e-6
    max_thickness_cm = -np.log(complete_attenuation_level)/np.min(mu)
    thickness_cm = np.linspace(max_thickness_cm, 0, 100)
    
    atten = np.exp(-mu[None,:]*thickness_cm[:,None])
    penetrating_intensity = src_intensity[None,:]*atten
    atten_poly = penetrating_intensity.sum(axis=1)/np.sum(src_intensity)
    atten_poly[0] = 0.0   # Little cheat for the table lookup
    
    atten_mono = np.exp(-design_mu*thickness_cm)
    
    return atten_poly, atten_mono, thickness_cm



















