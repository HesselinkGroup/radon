import numpy as np
import scipy


class DPCBeamHardeningCorrection:
    
    def __init__(self, d12, p2, material, poly_keV, poly_intensity, poly_visibility, mono_keV):
        """Beam hardening corrector.
        
        Perform beam hardening correction (the "water correction") for amplitude and phase signals in DPC.
        
        Amplitude correction:
            Use poly intensity to look up mass thickness
            Use mass thickness to look up mono intensity.
        
        Phase correction:
            Use poly intensity to look up mass thickness
            Use poly phase shift and mass thickness to look up mass slope
            Use mass slope to look up mono phase shift.
        """
        self.d12 = d12
        self.p2 = p2
        self.material = material
        self.poly_keV = poly_keV
        self.poly_intensity = poly_intensity
        self.poly_visibility = poly_visibility
        self.mono_keV = mono_keV
        
        # Lookup tables
        
        max_thickness = 100.0
        self.thickness_list = np.linspace(0.0, max_thickness)
        print("Plz calculate max_thickness rigorously")
        
        self.phi_list = np.linspace(-np.pi, np.pi, 100)
        self.poly_phi_table = None
        self.init_poly_phi_table()
        
        self.init_poly_transmission_table()
    
    # ==== BEAM HARDENING CORRECTION FUNCTION
    
    def correct_sinograms(self, intensity_poly, phase_poly):
        """Perform beam-hardening correction.
        
        Args:
            intensity_poly (np.array): polyenergetic intensity measurements e.g. sinogram
            phase_poly (np.array): polyenergetic Talbot phase shift measurements e.g. sinogram
        
        Returns:
            np.array: monoenergetic intensity, same shape as intensity_poly
            np.array: monoenergetic Talbot phase shift, same shape as phase_poly
        """
        mass_thickness = np.interp(intensity_poly, self.poly_transmission_table[::-1], self.thickness_list[::-1])

        intensity_mono = np.interp(mass_thickness, self.thickness_list, self.mono_transmission_table.ravel())
        mass_slope = self.lookup_mass_slope(phase_poly, mass_thickness)
        phase_mono = self.calc_phi(mass_slope, self.mono_keV)[0]
        
        
        return self._impl.correct_sinograms_implementation(intensity_poly, phase_poly)

        return intensity_mono, phase_mono


    # ==== Implementation-related stuff
    
    def calc_transmission(self, mass_thickness, keV):
        mass_thickness = np.atleast_1d(mass_thickness)
        keV = np.atleast_1d(keV)
        
        mu_rho = self.material.mu(keV) / self.material.density
        
        tx = np.exp(-mu_rho[:,None]*mass_thickness[None,:])
        return tx

    def calc_poly_transmission(self, mass_thickness):
        poly_tx = np.sum(self.poly_intensity[:,None]*self.calc_transmission(mass_thickness, self.poly_keV), axis=0)/np.sum(self.poly_intensity)
        return poly_tx
    
    def calc_phi(self, mass_slope, keV):
        keV = np.atleast_1d(keV)
        delta_rho = self.material.delta(keV) / self.material.density

        mass_slope = np.atleast_1d(mass_slope)
        x = (2*np.pi*self.d12/self.p2)*mass_slope[None,:]*delta_rho[:,None]
        return x

    def calc_mass_slope(self, phi, keV):
        delta_rho = self.material.delta(keV) / self.material.density
        mass_slope = phi / (2*np.pi*self.d12/self.p2) / delta_rho
        return mass_slope
    
    def calc_poly_phi(self, mass_thickness, mass_slope):
        delta_rho = self.material.delta(self.poly_keV) / self.material.density

        mass_thickness = np.atleast_1d(mass_thickness)
        mass_slope = np.atleast_1d(mass_slope)
        
        phi = self.calc_phi(mass_slope, self.poly_keV)
        
        weight = self.poly_intensity[:,None,None] * self.poly_visibility[:,None,None]
        transmission = self.calc_transmission(mass_thickness, self.poly_keV)[:,:,None]
        phase_factor = np.exp(1j*phi[:,None,:])
        
        poly_phi = np.angle( np.sum(weight*transmission*phase_factor, axis=0) )
        return poly_phi

    def init_poly_phi_table(self):
        
        self.poly_phi_table = np.zeros((len(self.phi_list), len(self.thickness_list)))

        # For each mass_thickness I want to invert the calculation of phi so I can
        # map mass_thickness,phi -> mass_slope.  I'm going to invert it the easiest
        # way: linear interpolation (since the functions are monotone).
        #
        # First, for each mass_thickness, make a mass_slope -> phi map.
        # I need to pick a range of mass_slope values that will give me all phi
        # between -pi and pi.  I'll calculate the largest mass slope I could
        # possibly want by working at 180 keV...

        max_slope = self.calc_mass_slope(np.pi, 180.0) # this is the biggest mass slope we would ever need to consider.
        initial_mass_slopes = np.linspace(0.0, max_slope, 1000)

        for col, thickness in enumerate(self.thickness_list):
            phis = self.calc_poly_phi(thickness, initial_mass_slopes).ravel()
            idx = np.where(np.diff(phis) < 0)[0][0]

            phis = np.concatenate((-phis[idx::-1], phis[1:idx+1]))
            mass_slopes = np.concatenate((-initial_mass_slopes[idx::-1], initial_mass_slopes[1:idx+1]))

            self.poly_phi_table[:,col] = np.interp(self.phi_list, phis, mass_slopes)

    def init_poly_transmission_table(self):
        self.poly_transmission_table = self.calc_poly_transmission(self.thickness_list)
        self.mono_transmission_table = self.calc_transmission(self.thickness_list, self.mono_keV)
    
    def lookup_mass_slope(self, phi, mass_thickness):
        """Infer mass slope from mass thickness and polyenergetic phi (Talbot phase shift from 0 to 2*pi).
        
        Uses bilinear interpolation into the polyenergetic phi table.
        
        Args:
            phi: polyenergetic phi (Talbot phase shift)
            mass_thickness: line integral of density
        
        Returns:
            array: mass slope (lateral derivative of mass thickness)
        """
        
        if np.isscalar(phi):
            phi = np.atleast_1d(phi)
        if np.isscalar(mass_thickness):
            mass_thickness = np.atleast_1d(mass_thickness)

        mass_thickness = np.atleast_1d(mass_thickness)
        mass_slope = scipy.interpolate.interpn((self.phi_list, self.thickness_list), self.poly_phi_table, np.column_stack((phi.ravel(), mass_thickness.ravel())))
        mass_slope.shape = phi.shape
        return mass_slope


    