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
Statistical rate theory models
This code is part of unikin, available at:
https://gitlab.ethz.ch/paenurke/unikin
"""

import numpy as np
from scipy import integrate
# local modules
from . import export, timing, constants, misc, statecount
from .rate import rrkm_rate, j_averaged_rate


"""
L-CID model
"""

@export
class LCID:
    """Calculate L-CID rates
    """

    # Fitted parameters (from https://doi.org/10.1021/jp072092l)
    beta_slope = 0.5190
    alpha_slope = 0.1593
    alpha_intercept = 1.546

    def __init__(self, dof, rotors, e0, v_eff, alpha_prime, energies=None, emax=50010, egrain=10):
        """       
        Arguments:
            dof {int} -- number of degrees of freedom
            rotors {int} -- number of rotors
            energies {list} -- list of energy values to evaluate the rate at (in cm-1)
            e0 {float} -- E0
            v_eff {float} -- effective frequency
            alpha_prime {float} -- alpha prime as defined for LCID
        """
        # Define variables
        self.dof = dof
        self.rotors = rotors
        self.v_eff = v_eff
        if energies == None:
            self.energies = np.arange(0,emax,egrain)
        else:
            self.energies = np.asarray(energies)
        self.e0 = e0
        # Set up constants
        self.alpha_prime = alpha_prime
        self.calc_param()
        # Calculate DOS and SOS
        self.calc_phasespace()

    def calc_dos(self, ene, ts = False):
        """Density of States calculator
        Equations 9 and 16 in the SI of https://doi.org/10.1021/jp072092l
        Arguments:
            ene {float or np.array} -- energy to eval DOS at (in cm-1)
            ts {bool} -- True for calculating TS DOS
        
        Returns:
            [float or np.array] -- density of states
        """
        if ts: # for the TS
            dos = np.exp(self.beta_ts * (np.sqrt((ene * (ene + self.alpha_prime)) / (ene + self.alpha))))
        else: # for the reactant
            dos = np.exp((self.beta * ene) / np.sqrt(ene + self.alpha))
        return dos

    def calc_sos(self):
        """Sum of States calculator
        
        Arguments:
            ene {float or np.array} -- energy to eval SOS at (in cm-1)
        
        Returns:
            [float or np.array] -- sum of states
        """
        ene = self.energies - self.e0
        ene[ene < 0] = 0
        dos_ts = lambda x: self.calc_dos(x, ts = True)
        sos = np.zeros_like(ene)
        for i in range(len(sos)):
            sos[i] = integrate.quad(dos_ts, 0, ene[i], epsabs=0.0, epsrel=1.0e-6)[0]
        return sos
    
    def calc_param(self):
        """Calculates the LCID parameters
        Based on the SI of https://doi.org/10.1021/jp072092l
        """
        self.alpha = self.v_eff * np.exp(self.alpha_intercept - self.alpha_slope * self.rotors)
        self.beta = np.sqrt(self.dof / (self.beta_slope * self.v_eff))
        self.beta_ts = np.sqrt((self.dof - 1) / (self.beta_slope * self.v_eff))

    def calc_phasespace(self):
        """Calculates the phase space variables
        """
        self.dos = self.calc_dos(self.energies, ts = False)
        self.sos = self.calc_sos()

    def calc_rate(self):
        """Calculates the rate
        """
        self.rate = rrkm_rate(self.sos,self.dos)

    def calc_all(self):
        """Calculate the parameters, phase space functions, and the rate
        """
        self.calc_param()
        self.calc_phasespace()
        self.calc_rate()


"""
Conventional statistical rate models
"""

class ParentRate:
    """Instantiates the parent class for rate calculators
    """

    def __init__(self, reactant_file, ts_file, e0, intype='densum', 
                sym = 1, egrain=10, emax=50010, B=None, B_ts=None):
        """        
        Arguments:
            reactant_file {string} -- Reactant file
            ts_file {string or list} -- TS file(s)
            e0 {float} -- E0 value

        Keyword Arguments:
            s {int} -- reaction degeneracy (default: {1})
            intype {str} -- type of input (Options: densum, freq, txt)
        """

        self.e0 = e0 
        self.sym = sym # degeneracy / symmetry number
        self.B = B # rot constant for reactant
        self.B_ts = B_ts # rot constant for TS
        self.j_rate = False # Toggle for J-averaging, currently handled externally

        # Read from densum output files:
        if intype == 'densum':
            self.energies = misc.read_densum(reactant_file)['ene']
            self.dos = misc.read_densum(reactant_file)['dos']
            if isinstance(ts_file, str):
                self.sos = misc.read_densum(ts_file)['sos']
            elif isinstance(ts_file, list):
                self.sos = misc.read_densum(ts_file[0])['sos']

        # Read frequencies from a file (for Beyer-Swinehart)
        elif intype == 'freq':
            self.energies = np.arange(0, emax, egrain)
            # Reactant: check if file (string); load file if file
            if isinstance(reactant_file, str):
                freq_temp = np.loadtxt(reactant_file)
                self.dos = statecount.beyer_swinehart(freq_temp, egrain, emax, count='dos')
            else:
                self.dos = statecount.beyer_swinehart(reactant_file, egrain, emax, count='dos')
            # TS: Check if string or list, deal accordingly
            if isinstance(ts_file, str):
                freq_temp = np.loadtxt(ts_file)
                self.sos = statecount.beyer_swinehart(freq_temp, egrain, emax, count='sos')
            elif isinstance(ts_file, list):
                if isinstance(ts_file[0], str):
                    freq_temp = np.loadtxt(ts_file[0])
                    self.sos = statecount.beyer_swinehart(freq_temp, egrain, emax, count='sos')
                else:
                    self.sos = statecount.beyer_swinehart(ts_file[0], egrain, emax, count='sos')
            elif isinstance(ts_file, np.ndarray):
                self.sos = statecount.beyer_swinehart(ts_file, egrain, emax, count='sos')

        # Read from regular two-column text files
        elif intype == 'txt':
            self.energies = np.loadtxt(reactant_file).T[0]
            self.dos = np.loadtxt(reactant_file).T[1]
            if isinstance(ts_file, str):
                self.sos = np.loadtxt(ts_file).T[1]
            elif isinstance(ts_file, list):
                self.sos = np.loadtxt(ts_file[0]).T[1]
        
        # Define energy step (assume uniform for whole data)
        self.estep = self.energies[1] - self.energies[0]


    def calc_rate(self):
        """Calculates the rate
        """
   
        if not self.j_rate:
            # Shift the sos array by E0 (in terms of energy steps)
            shift_length = int(round(self.e0/self.estep))
            sos = misc.shift_array_w_nan(self.sos, shift_length)
            # Calculate the rate
            self.rate = rrkm_rate(sos, self.dos, s = self.sym)
       
        else:
            """ Calculates the J-averaged rate
            """
            # Setup variables for J averaging
            g_J, E_prefactor = misc.setup_j_ave(self.B, self.energies[-1])
            E_J = E_prefactor * self.B
            E_ts_J = E_prefactor * self.B_ts

            # Calculate the J-averaged rate
            self.rate = j_averaged_rate(self, self.e0, self.sos, E_J, E_ts_J, g_J)


@export
class RACRRKM(ParentRate):
    """RAC-RRKM rate calculator
    """
    def __init__(self, reactant_file, ts_file, e0, intype='densum', 
                sym = 1, egrain=10, emax=50010, B=None, B_ts=None):
        super().__init__(reactant_file, ts_file, e0, intype, sym, egrain, emax, B, B_ts)

@export
class VTST(ParentRate):
    """ VTST calculator
    """
    def __init__(self, reactant_file, ts_file, e0, intype='densum', sym = 1, egrain=10, emax=50010, B=None, B_ts=None):
        super().__init__(reactant_file, ts_file, e0, intype, sym, egrain, emax, B, B_ts)

        # Make a list of TS B values if given
        if B_ts:
            self.B_list = [b for b in B_ts]
        
        # SOS data for all the TS files
        self.sos_vtst = [[] for i in range(len(ts_file))]
        if intype == 'densum':
            for i in range(0,len(ts_file)):
                self.sos_vtst[i] = misc.read_densum(ts_file[i])['sos']
        elif intype == 'freq':
            for i in range(0,len(ts_file)):
                if isinstance(ts_file[i], str):
                    freq_temp = np.loadtxt(ts_file[i])
                    self.sos_vtst[i] = statecount.beyer_swinehart(freq_temp, egrain, emax, count='sos')
                else:
                    self.sos_vtst[i] = statecount.beyer_swinehart(ts_file[i], egrain, emax, count='sos')
        elif intype == 'txt':
            for i in range(0,len(ts_file)):
                self.sos_vtst[i] = np.loadtxt(ts_file[i]).T[1]
    #@timing
    def calc_rate(self):
        """ VTST rate calculator
        """
        # Initalize data containers
        all_rates = []

        if not self.j_rate:
            # Loop over all the energies, generate sos and dos arrays
            for x, e0 in enumerate(self.e0):
                # Shift the sos array by E0 (in terms of energy steps)
                shift_length = int(round(e0/self.estep))
                sos = misc.shift_array_w_nan(self.sos_vtst[x], shift_length)
                # Calculate and append the rate
                all_rates.append(rrkm_rate(sos, self.dos, s = self.sym))

        else:
            """ VTST J-averaged rate calculator
            """
            # Setup variables for J averaging
            g_J, E_prefactor = misc.setup_j_ave(self.B, self.energies[-1])
            E_J = E_prefactor * self.B
            E_ts_J = [E_prefactor * b for b in self.B_list]
            # Loop over all the energies
            for x, e0 in enumerate(self.e0):
                # Calculate the J-averaged rate
                rate = j_averaged_rate(self, e0, self.sos_vtst[x], E_J, E_ts_J[x], g_J)
                all_rates.append(rate)

        # Find the final rate by taking the minimum rate along each energy axis
        self.rate = np.amin(all_rates, axis = 0)


class Pst(ParentRate):
    """Instantiates Phase Space Theory
    """
    def __init__(self, reactant_file, ts_file, e0, intype='densum', 
                sym = 1, egrain=10, emax=50010, B=None, B_ts=None):
        super().__init__(reactant_file, ts_file, e0, intype, sym, egrain, emax, B, B_ts)

        # Auxiliary DOS data for convolution with many TS files
        self.sos_rot = self.sos # the rotational SOS has to be the SOS for SSACM
        self.dos_aux = [[] for i in range(1,len(ts_file))]
        if intype == 'densum':
            for i in range(0,len(ts_file)-1):
                self.dos_aux[i] = misc.read_densum(ts_file[i+1])['dos']
        elif intype == 'freq':
            for i in range(0,len(ts_file)-1):
                if isinstance(ts_file[i+1], str):
                    freq_temp = np.loadtxt(ts_file[i+1])
                    self.dos_aux[i] = statecount.beyer_swinehart(freq_temp, egrain, emax, count='dos')
                else:
                    self.dos_aux[i] = statecount.beyer_swinehart(ts_file[i+1], egrain, emax, count='dos')
        elif intype == 'txt':
            for i in range(0,len(ts_file)-1):
                self.dos_aux[i] = np.loadtxt(ts_file[i+1]).T[1]


    def calc_sos(self, f_rigid = False):
        """Calculate Sum of States by convolution of different TS data
        """
        # Convolve the functions (SOS and first DOS should be rotations for SSACM)
        # If 1 auxiliary DOS files given, assume it to be vibrational
        if len(self.dos_aux) == 1: 
            temp_sos = self.sos_rot.copy()
            # Correct the rotations with f_rigid if SSACM
            if f_rigid:
                self.calc_frigid()
                temp_sos *= self.f_rigid
            self.sos = np.convolve(temp_sos, self.dos_aux[0])[:len(self.energies)]
            self.sos[1:] *= self.estep
        # Otherwise expect 3 auxiliary DOS files: first rotational, second two vibrational
        elif len(self.dos_aux) == 3:
            temp_sos = np.convolve(self.sos_rot, self.dos_aux[0])[:len(self.energies)]
            temp_sos[1:] *= self.estep
            if f_rigid:
                self.calc_frigid()
                temp_sos *= self.f_rigid
        # Convolve all further DOS files
            temp_dos = np.convolve(self.dos_aux[1], self.dos_aux[2])[:len(self.energies)]
            temp_dos[1:] *= self.estep
            self.sos = np.convolve(temp_sos, temp_dos)[:len(self.energies)]
            self.sos[1:] *= self.estep

        else:
            print('PSL and SSACM expect either 2 -t flags (rot SOS, vib DOS) or 4 -t flags (rot SOS, rot DOS, 2 vib DOS).')
            print('Exiting.')
            exit(1)

@export
class PSL(Pst):
    """ Phase Space Limit calculator
    """
    def __init__(self, reactant_file, ts_file, e0, intype='densum', 
                sym = 1, egrain=10, emax=50010, B=None, B_ts=None):
        super().__init__(reactant_file, ts_file, e0, intype, sym, egrain, emax)
        self.calc_sos()

@export
class SSACM(Pst):
    """Simplified SACM calculator
    """

    def __init__(self, reactant_file, ts_file, e0, intype='densum', 
                sym = 1, egrain=10, emax=50010, B=None, B_ts=None,
                ssacm_c = 1, disstype='ion-mol'):
        """        
        Arguments:
            ssacm_c {float} -- coefficient for f_rigid scaling
        """
        super().__init__(reactant_file, ts_file, e0, intype, sym, egrain, emax, B, B_ts)
        self.ssacm_c = ssacm_c
        self.disstype = disstype
        self.calc_sos(f_rigid = True)

    def calc_frigid(self):
        """Calculate the rigidity factor
        Equations S1 and S2 in http://dx.doi.org/10.1021/acs.jpca.1c00183
        """
        if self.disstype == 'ion-atom':
            self.f_rigid = np.exp(-self.energies/self.ssacm_c)
        elif self.disstype == 'ion-mol':
            self.f_rigid = (1 + ((self.energies)/self.ssacm_c)**2 )**(-2/3)
            
    
