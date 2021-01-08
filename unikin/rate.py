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
Rate equations
This code is part of unikin, available at:
https://gitlab.ethz.ch/paenurke/unikin
"""

import numpy as np
# local modules
from . import export, timing
from . import constants, misc

@export
def rrkm_rate(sos_ts, dos, s = 1):
    """Calculate the RRKM rate

    Arguments:
        sos_ts {array} -- Sum of States at the TS
        dos {array} -- Density of States of the reactant

    Keyword Arguments:
        s {int} -- reaction symmetry number / degeneracy (default: {1})

    Returns:
        array -- rates
    """
    dos = np.asarray(dos)
    sos_ts = np.asarray(sos_ts)
    dos[dos == 0] = 1
    rate = (s * sos_ts / (dos * constants.h))
    return rate

#@timing
def j_averaged_rate(model, e0, sos_in, E_J, E_ts_J, g_J):
    """Calculate the J-averaged RRKM rate (Eq 5 in the paper)
        (this function is not really well-suited for using separately
        outside the methods.py module)

    Args:
        model (class object): rate model object
        e0 (float): E0 value (cm-1)
        sos_in (array): SOS array from the class object (separately for
                        compatibility with VTST)
        E_J (float): Rotational energy of the reactant
        E_ts_J (float): Rotational energy of the TS
        g_J (int): rotational degeneracy

    Returns:
        array: J-averaged rate
    """
   
    # Initialize containers
    dos = []
    sos = []
    # Length of one data array
    data_len = len(model.energies)
    # Energy step in the data (should be uniform)
    e_step = model.energies[1] - model.energies[0]

    # Prepare the sos array for each J value
    for e in E_ts_J:
        # Calculate length of the energy shift
        shift_length = int(round((e0 + e)/e_step))
        # If it's more than the data, then sos is empty
        if shift_length >= data_len:
            sos.append(np.full_like(sos_in, np.nan))
        # Otherwise shift the sos array by shift_length
        else:
            sos.append(misc.shift_array_w_nan(sos_in, shift_length))

    # Prepare the dos array for each J value
    for e in E_J:
        # Calculate length of the energy shift
        shift_length = int(round(e/e_step))
        # Shift the dos array by shift_length
        dos.append(misc.shift_array_w_nan(model.dos, shift_length))

    # Calculate the rates
    j_rates = rrkm_rate(sos, dos, s = model.sym)

    # Change nan in dos to 0 for weighting
    dos = np.nan_to_num(dos)
    # Calc the weights and the weighted rates
    weights = np.array([g_J] * data_len).T * dos
    weighted_rates = np.asarray(j_rates) * weights
    # Calculate the final rate by summing over all J-s and weights
    rate = np.sum(np.nan_to_num(weighted_rates), axis = 0) / np.sum(weights, axis = 0)
    return rate
