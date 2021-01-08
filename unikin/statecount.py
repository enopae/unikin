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
Functions for counting density/sum of states
This code is part of unikin, available at:
https://gitlab.ethz.ch/paenurke/unikin
"""

import numpy as np
from scipy.interpolate import interp1d
# local modules
from . import export, timing

@export
def beyer_swinehart(freq, egrain, emax, count='dos', interp=False):
    """Beyer-Swinehart state count

    Args:
        freq (array): array of frequencies
        egrain (int): energy grain (same unit as freq)
        emax (int): maximum energy (same unit as freq)
        count (str, optional): Count type - dos or sos. Defaults to 'dos'.
        interp (bool, optional): Interpolate to make smoother. Defaults to False.

    Returns:
        array: density or sum of states
    """
    # Initialize values for counting (reduce to integers divided by egrain)
    emax = int(round(emax/egrain))
    freq = abs(np.array(freq)) # make all values positive
    if type(freq) == np.float64: # to deal with only one provided frequency value
        freq = [freq]
    # For high egrain values, should use a multiplier for correcting the initial SOS array
    ini_multiplier = egrain//freq[0]
    if ini_multiplier == 0:
        ini_multiplier = 1
    # Set up frequencies
    freq = [1 if f<egrain else int(round(f/egrain)) for f in freq]
    # Set up DOS or SOS count
    if count == 'dos':
        bs = np.zeros(emax)
        bs[0] = 1
    elif count == 'sos':
        bs = np.ones(emax) * ini_multiplier
    # Count the states
    for i in range(0,len(freq)):
        for j in range(freq[i],emax):
            bs[j] += bs[j-freq[i]]
    # Interpolate if toggled
    if interp:
        bs[bs < 1] = 0
        idx = np.nonzero(bs)
        x = np.arange(len(bs))
        interp = interp1d(x[idx],bs[idx],fill_value='extrapolate')
        bs = interp(x)
    # Divide DOS by egrain to get DOS per 1 energy unit
    if count == 'dos':
        bs[1:] /= egrain
    return bs

@export
def rot1D(B,energies):
    """Calculates dos and sos of a 1D classical rotor
    According to formulas from Hase's book p. 176

    Arguments:
        B {float} -- rotational constant in cm-1
        energies {list} -- energies in cm-1
    """
    energies = energies.astype(float)
    energies[energies == 0] = 1e-1 # avoid divison by 0 error
    dos = np.sqrt(1/(B*energies))
    sos = 2*np.sqrt(energies/B)
    return {'dos': dos, 'sos': sos}

@export
def rot2D(B,energies):
    """Calculates dos and sos of a 2D classical rotor
    According to formulas from Hase's book p. 176

    Arguments:
        B {float} -- rotational constant in cm-1
        energies {list} -- energies in cm-1
    """
    dos = 1/B*np.ones_like(energies)
    sos = energies/B
    return {'dos': dos, 'sos': sos}