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
Miscellaneous utilities
(not all are used in the main code)
This code is part of unikin, available at:
https://gitlab.ethz.ch/paenurke/unikin
"""

import numpy as np
from bisect import bisect_left
import re
# local modules
from . import export, constants

@export
def shift_array_w_nan(array, shift_length):
    """Shift data in 1D array by inserting np.nan in the beginning
        (the array length remains the same, so last values are lost)

    Args:
        array (ndarray): 1D array to be shifted
        shift_length (int): length of the shift
    """
    ini_length = len(array)
    filler = np.full(shift_length, np.nan)
    new_array = np.concatenate((filler, array[:ini_length - shift_length]))
    return new_array

@export
def setup_j_ave(B, energy):
    """Calculates some parameters for J-averaging calculations

    Args:
        B (float): rotational constant in cm-1
        energy (list or float): energy value or list in cm-1

    Returns:
        [list]: parameters used in J-average rate calculations
    """
    # Max J value possible (Armentrout 1997)
    J_max = int(((1 + 4*energy / B)**0.5 -1 ) / 2)
    # Precalculate J-related data
    J_vals = np.arange(0, J_max + 1)
    g_J = 2 * J_vals + 1
    E_prefactor = constants.h * constants.c * J_vals * (J_vals + 1)
    return g_J, E_prefactor

@export
def jk_rotors(B_vals):
    """Identify J and K rotor from 3 B values

    Args:
        B_vals (array): array of three B values

    Returns:
        dict: the K rotor and J rotor B values
    """
    diff1 = B_vals[0] - B_vals[1]
    diff2 = B_vals[1] - B_vals[2]
    if diff1 > diff2:
        k_rot = B_vals[0]
        j_rot = (B_vals[1]*B_vals[2])**(1/2)
    else:
        k_rot = B_vals[2]
        j_rot = (B_vals[0]*B_vals[1])**(1/2)
    return {'k': k_rot,'j': j_rot}

@export
def convert2cm(value, unit):
    """Convert energy values to cm-1

    Args:
        inval (float): energy value 
        unit (unit): energy unit (kcal, ev, or kj)

    Returns:
        float: energy value in cm-1
    """
    val = float(value)
    if 'kcal' in unit:
        val *= constants.kcal2cm
    elif 'ev' in unit:
        val *= constants.ev2cm
    elif 'kj' in unit:
        val *= constants.kj2cm
    return val

@export
def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after
    else:
       return before

@export
def take_closest_np(array, values):
    """Takes closest values between two numpy arrays

    Arguments:
        array {array} -- reference 1D array
        values {array} -- values to convert

    Returns:
        array -- converted values
    """
    #make sure array is a numpy array
    array = np.array(array)

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = ((idxs == len(array))|(np.fabs(values - array[np.maximum(idxs-1, 0)]) < np.fabs(values - array[np.minimum(idxs, len(array)-1)])))
    idxs[prev_idx_is_less] -= 1

    return idxs

@export
def write_2col_rate_result(name,calculator,func_name):
    """Writes a 2 column file from a rate calculator

    Args:
        name (str): basename for the files
        calculator (object): calculator object
        func_name (str): data to be written
    """
    column1 = calculator.energies
    column2 = getattr(calculator, func_name)
    with open(name + '_' + func_name + '.dat','w',encoding='utf8') as f:
            for i in np.arange(0,len(column1)):
                    f.write("%.1f %.5e\n" % (column1[i], column2[i]))
            f.close()

@export
def write_2col_result(name, energies, data, func_name):
    """Writes a 2 column file (meant for dos or sigma calculation results)

    Args:
        name (str): basename for the files
        energies (array): energies corresponding to the data
        data (array): data to be written
        func_name (str): name of the function to be written
    """
    with open(name + '_' + func_name + '.dat','w',encoding='utf8') as f:
            for i in np.arange(0,len(data)):
                    f.write("%.1f %.5e\n" % (energies[i], data[i]))
            f.close()

@export
def write_opt_result(name, opt_result):
    """Writes the result file of the rate optimization

    Args:
        name (str): basename for the files
        opt_result (dict): optimization results
    """
    with open(name + '_opt.log','w',encoding='utf8') as f:
            for i in opt_result:
                try:
                    f.write("%-14s %.6f\n" % (i, opt_result[i]))
                except: # hack for VTST
                    f.write("%-14s %.6f\n" % (i, max(opt_result[i])))
            f.close()

@export
def read_file(file):
    """
    Reads the content of a file and splits the line content into a list
    """
    content = []
    with open(file, encoding='utf8') as f:
        for line in f:
                content.append(line.split())
        f.close()
    return content

@export
def read_densum(file):
    """
    Reads a densum output file
    Returns a list with energy, DOS and SOS
    Reads the length of the lists from the densum file
    """
    with open(file, encoding='utf8') as f:
        content = f.readlines()
        for i, line in enumerate(content):
            if 'Density' in line.split():
                n_lines = int(content[i-1].split()[1])
                line_start = i + 1
                break
        data = {'ene': np.zeros(n_lines),
                'dos': np.zeros(n_lines),
                'sos': np.zeros(n_lines)
                }
        for i in range(line_start,line_start + n_lines):
            for j, label in enumerate(data):
                data[label][i-line_start] = float(content[i].split()[j+1])
        f.close()
    return data

def atoi(text):
    return int(text) if text.isdigit() else text

@export
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

