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
Optimization of the rate or cross-section
This code is part of unikin, available at:
https://gitlab.ethz.ch/paenurke/unikin
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar
from scipy.signal import savgol_filter
# local modules
from . import export, timing, constants, misc, sigmasim

@export
class Optimizer():
    """ Optimization class for fitting the rate or cross section data
    """

    def __init__(self, calculator, simulator = None, ncore = 1, verbose = True):
        self.calculator = calculator # rate model object
        self.simulator = simulator # sigmasigm object
        self.ncore = ncore # number of cores for running differential evolution
        self.verbose = verbose # print data for each step
    

    def optimize(self, ref_energies, ref_data, opt_type = 'rate', e_guess = 55, e_win = 45,
                min_alpha = 5000, max_alpha = 6500):

        # Define opt type for the class
        self.opt_type = opt_type
        # Define reference energies and cross section or rate
        self.ref_energies = np.asarray(ref_energies)
        if opt_type == 'rate':
            self.ref_rate = np.asarray(ref_data)
        elif opt_type == 'sigma':
            smooth_win_len = int(np.ceil(1000/(self.ref_energies[1]-self.ref_energies[0])) // 2 * 2 + 1) # calculate smoothing window
            self.ref_sigma = savgol_filter(np.asarray(ref_data),smooth_win_len,1) # smooth the data
            self.ref_sigma = (self.ref_sigma - min(self.ref_sigma)) / (max(self.ref_sigma) - min(self.ref_sigma)) # normalize
        else:
            raise NameError('The selected optimization \"'+str(opt_type)+'\" does not exist.')
        
        # Define E0 bounds
        e_upper = e_guess + e_win
        e_lower = e_guess - e_win
        # Check and adjust bounds to the energy range if necessary (helps avoid convergence issues)
        if e_upper * constants.kcal2cm > max(ref_energies):
            e_upper = max(ref_energies)/constants.kcal2cm
        if e_lower < 0:
            e_lower = 1.0

        # Optimize the parameters depending on the method
        if type(self.calculator).__name__ == 'SSACM':
            # The bounds are for E0 (in cm^-1) and ssacm_c (in eV)
            result = differential_evolution(self._ssacm_rmsd, bounds=[(e_lower,e_upper),(0,5)], workers=self.ncore)
            return {'e0': self.calculator.e0,
                    'ssacm_c': self.calculator.ssacm_c,
                    'rmsd': result.fun}
        elif type(self.calculator).__name__ == 'LCID':
            # The bounds are for E0, v_eff, and alpha_prime - in that order, all in cm^-1
            result = differential_evolution(self._lcid_rmsd, bounds=[(e_lower,e_upper),(500,1000),(min_alpha, max_alpha)], workers=self.ncore)
            return {'e0': self.calculator.e0, 
                    'v_eff': self.calculator.v_eff,
                    'alpha_prime': self.calculator.alpha_prime,
                    'rmsd': result.fun}
        else:
            #result = differential_evolution(self._rmsd, bounds=[(e_lower,e_upper)])
            # scalar minimization is faster than evolution; can uncomment the line above and comment the one below if want to use DE
            result = minimize_scalar(self._rmsd, method = 'bounded', bracket=(e_lower, e_upper), bounds=([e_lower, e_upper]))
            return {'e0': self.calculator.e0, 
                    'rmsd': result.fun}

    def _rmsd(self, e0):
        """Calculate RMSD between rate or cross section data
        """
        # Scale all energies if it's VTST
        # Else just redefine the E0
        if type(self.calculator).__name__ == 'VTST':
            self.calculator.e0 = [float(e*(e0/max(self.calculator.e0)))*constants.kcal2cm for e in self.calculator.e0]
        else:
            if isinstance(e0, np.ndarray): # just to avoid an error
                e0 = float(e0[0])
            self.calculator.e0 = e0 * constants.kcal2cm
        # Recalcuate SOS for SSACM (because c changes it)
        if type(self.calculator).__name__ == 'SSACM':
            self.calculator.calc_sos(f_rigid = True)
        # Calculate phase space for LCID
        if type(self.calculator).__name__ == 'LCID':
            self.calculator.calc_phasespace()
        
        # Optimize the parameters to fit the rate
        if self.opt_type == 'rate':
            # Calculate the rate
            self.calculator.calc_rate()
            rate = np.nan_to_num(self.calculator.rate)
            # Get the rate values at the reference energies
            compare_rate = rate[misc.take_closest_np(self.calculator.energies, self.ref_energies)]
            # Compute the RMSD of log10(k)
            rmsd = np.sqrt(np.sum((np.log10(self.ref_rate)-np.log10(compare_rate))**2)/len(self.ref_rate))
        
        # Optimize the parameters to fit the cross section
        elif self.opt_type == 'sigma':
            sigma, energies = self.simulator.calc_sigma(self.calculator)
            # Get the cross section values at the reference energies
            compare_sigma = sigma[misc.take_closest_np(energies, self.ref_energies)]
            compare_sigma = (compare_sigma - min(compare_sigma)) / (max(compare_sigma) - min(compare_sigma)) # normalize
            # Compute the RMSD of sigma
            rmsd = np.sqrt(np.sum((self.ref_sigma-compare_sigma)**2)/len(self.ref_sigma))
            # if rmsd is nan, convert it to a high value instead
            if np.isnan(rmsd):
                rmsd = 100.0

        # Print if verbose
        if self.verbose:
            if type(self.calculator).__name__ == 'LCID':
                print("%5.2f %9.2f %9.2f %8.4f" % (e0, self.calculator.v_eff, self.calculator.alpha_prime, rmsd))
            elif type(self.calculator).__name__ == 'SSACM':
                print("%5.2f %9.2f %8.4f" % (e0, self.calculator.ssacm_c, rmsd))
            else:
                print("%5.2f %8.4f" % (e0, rmsd))
        return rmsd

    def _ssacm_rmsd(self, params):
        """Function for SSACM optimization
        """
        e0, c = params
        self.calculator.ssacm_c = c * constants.ev2cm
        return self._rmsd(e0)

    def _lcid_rmsd(self, params):
        """Function for LCID optimization
        """
        e0, v_eff, alpha_prime = params
        self.calculator.v_eff = v_eff
        self.calculator.alpha_prime = alpha_prime
        self.calculator.calc_param()
        self.calculator.calc_phasespace()
        return self._rmsd(e0)
