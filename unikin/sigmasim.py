# This file is part of unikin.
#
# Copyright (C) 2021 ETH Zurich, Eno Paenurk
#
# unikin is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# unikin is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with unikin. If not, see <https://www.gnu.org/licenses/>.

"""
Reaction cross-section (sigma_R) simulation
This code is part of unikin, available at:
https://gitlab.ethz.ch/paenurke/unikin
"""

import sys
import warnings
import numpy as np
from numpy.random import uniform, normal
from scipy.signal import savgol_filter
# local modules
from . import export, constants, misc, timing, statecount


@export
class SigmaSim():
    """Simulates the CID cross section
    """

    # scale emax up to generate the energy grid beyond emax
    # otherwise distributions with the new code get cut at emax
    emax_scale = 1.2

    def __init__(self, M_ion, gas = 'Ar', fwhm = 1.5, temp = 313.15, tau = 6e-5, n_ions = 1000,
                emax=20000, egrain=100):
        """
        Args:
            M_ion (float): molar mass of the ion
            gas (str, optional): Collision gas (Ne, Ar or Xe). Defaults to 'Ar'.
            fwhm (float, optional): KED FWHM in eV (from the experiment). Defaults to 1.5.
            temp (int, optional): Temperature in K. Defaults to 313.15.
            tau (float, optional): Residence time of the ions in s. Defaults to 6e-5.
            n_ions (int, optional): Number of ions in the simulation. Defaults to 1000.
            emax (float, optional): Maximum energy of the simulation (in cm-1). Defaults to 20000.
            egrain (float, optional): Energy grain in the simulation (in cm-1). Defaults to 10.
        """
        self.n_ions = n_ions
        self.tau = tau
        self.kBT = temp * constants.kB
        self.egrain = egrain
        self.emax = emax
        # generate energy grid until emax_scale * emax (to include distributions beyond emax)
        # not advisable to start with values <1 (low E values blow up p_b)
        self.energies = np.arange(1, self.emax_scale*emax, egrain, dtype=float)
        # Auxiliary constants
        # Conversion of KED FWHM to sigma^2
        self.ked_s2 = (1/(8.0 * np.log(2.0))) * (fwhm * constants.ev2cm  * (constants.gas_molmass[gas] / (constants.gas_molmass[gas] + M_ion)))**2
        # conversion of Doppler FHWM to sigma^2
        self.doppler_s2 =  (1/(8.0 * np.log(2.0))) * (11.1 * (M_ion / (constants.gas_molmass[gas] + M_ion)) * self.kBT)
        # conversion factor for the impact parameter (includes the conversion from cm^3 to m^3)
        self.impact_constant = 1e-8 * 2 / (4 * np.pi * constants.vac_permittivity) * constants.polarizability[gas]
        # Savitzky-Golay smoothing window length (approx 1000 cm-1)
        self.smooth_win_len = int(np.ceil(1000/self.egrain) // 2 * 2 + 1) # has to be odd


    def calc_sigma(self, rate_calculator):
        """Cross section simulation

        Args:
            rate_calculator (object): Rate model class object

        Returns:
            np array: Average sigma
        """
        # Warn in case max energy is above the one in the rate model
        if max(rate_calculator.energies) < max(self.energies):
            warnings.warn('Maximum energy in the rate model is '+str(max(rate_calculator.energies))+
                          ', maximum energy for the simulation is '+str(max(self.energies))+
                          '. Results may be unreliable.' )
        # Extract data from the calculator
        rate_calculator.calc_rate()
        idxs = misc.take_closest_np(rate_calculator.energies, self.energies)
        dos = np.nan_to_num(rate_calculator.dos[idxs])
        rate = np.nan_to_num(rate_calculator.rate[idxs])
        # Calculate the Boltzmann distribution
        p_boltz = self.gen_p_boltz(dos)
        # Calculate sigma lookup array
        rate_sigmoid = 1 - np.exp(-rate*self.tau)
        ###
        # Simulate CID
        ###
        # Energy distribution from KED
        p_e = self.gen_p_ked_dopp()
        # Impact parameter
        p_e, p_b = self.gen_impact_param(p_e)
        # Convolution with Boltzmann distribution
        p_e += np.tile(p_boltz, (p_e.shape[0],1))
        # Deconvolution with KERD
        p_kerd = self.gen_p_kerd(p_e, dos)
        p_e -= p_kerd
        ###
        # Average sigma calculation (Equation S14 in http://dx.doi.org/10.1021/acs.jpca.1c00183)
        ###
        # Get the indices and scale negative ones to 0 and the ones above max index to the max index value
        sigma_idxs = ((p_e)/self.egrain).astype('i')
        sigma_idxs[sigma_idxs < 0] = 0
        sigma_idxs[sigma_idxs > len(rate_sigmoid)-1] = len(rate_sigmoid)-1
        # Get the rate curve based on p_e (relies on rate_sigmoid corresponding to self.energies)
        sigma = rate_sigmoid[sigma_idxs]
        # Multiply by pi and impact parameter
        sigma *= np.pi * p_b ** 2
        # Average for each energy and rescale until emax
        sigma_mean = np.mean(sigma, axis = 1)[:int(len(self.energies)/self.emax_scale)]
        energies = self.energies[:len(sigma_mean)]
        # Apply Savitzky-Golay smoothing
        sigma_mean = savgol_filter(sigma_mean,self.smooth_win_len,1)
        return sigma_mean, energies
    
    """
    Distributions
    """
    
    def gen_p_boltz(self, dos):
        """
        Generate Boltzmann distribution of vibrational energies
        Equation S9 in http://dx.doi.org/10.1021/acs.jpca.1c00183 
        """
        # Calculate distribution on the energy grid and normalize
        boltz_ref = dos * np.exp(-self.energies / self.kBT)
        boltz_ref /= sum(boltz_ref)
        # Generate a random energy according to boltz_ref probability for each ion
        p_boltz = np.random.choice(self.energies, p = boltz_ref, size = self.n_ions)
        return p_boltz


    def gen_p_ked_dopp(self):
        """Calculate KED
           Related to equation S10 in http://dx.doi.org/10.1021/acs.jpca.1c00183 
           Note: Using random.choice instead of random.normal to avoid dealing with negative values.
           random.normal gives a shorter code, but cannot easily avoid negative values at low E
        """
        # Calculate normal distributions on the energy grid (shifted by the lab frame kinetic energy)
        x = np.linspace(min(self.energies), max(self.energies), self.n_ions)
        # Calculate total Gaussian sigma^2 for KED and Doppler
        sigma2 = self.ked_s2 + self.doppler_s2*x
        # Calculate combined distribution function for reference and normalize
        ref_dist = 1/np.sqrt(sigma2*np.pi*2) * np.exp(-0.5*(x - self.energies[: , None])**2 / sigma2)
        ref_dist = (ref_dist.T / np.sum(ref_dist, axis = 1)).T
        # Generate a random energy according to boltz_ref probability for each ion
        p_e = np.zeros_like(ref_dist)
        for i in np.arange(len(self.energies)):
            p_e[i] = np.random.choice(x, p = ref_dist[i], size = self.n_ions)
        return p_e


    def gen_impact_param(self, p_e):
        """Calculate impact parameter

        Args:
            p_e (np array): energy distribution
        """
        # Calculate triangular distributions for the b parameter (equation S12 in http://dx.doi.org/10.1021/acs.jpca.1c00183)
        b_dist = np.random.triangular(left=0, mode=1, right=1, size = p_e.shape)
        # Convert b_dist into b values (equation S6 in the SI of https://doi.org/10.1021/jp072092l)
        p_b = b_dist * (self.impact_constant / p_e) ** 0.25
        # Convert p_e from E_coll to E_loc (equation S11 in http://dx.doi.org/10.1021/acs.jpca.1c00183)
        p_e *= (1 - b_dist ** 2)
        return p_e, p_b


    def gen_p_kerd(self, p_e, dos):
        """Calculate KERD

        Args:
            p_e (np array): energy distribution
            dos (np array): density of states
        """
        # Calculate approximate combined DOS with the collision gas
        # (for large systems has a negligible effect; can comment out for speed, and change kerd_ref equation to just dos[::-1])
        gas_freq = np.asarray([20,20])
        gas_dos = statecount.beyer_swinehart(gas_freq, self.egrain, max(self.energies), count='dos')
        comb_dos = np.convolve(dos, gas_dos)[:len(dos)]
        comb_dos[1:] *= self.egrain
        # Calculate KERD distribution on the energy grid (equation S13 in http://dx.doi.org/10.1021/acs.jpca.1c00183)
        kerd_ref = np.sqrt(self.energies) * comb_dos[::-1]
        kerd_ref /= np.sum(kerd_ref)
        p_kerd = np.random.choice(self.energies, p = kerd_ref, size = p_e.shape)
        return p_kerd
