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
Physical constants
This code is part of unikin, available at:
https://gitlab.ethz.ch/paenurke/unikin
"""
# From CRC Handbook of Chemistry and Physics, 101st Edition
# http://hbcponline.com/faces/documents/01_01/01_01_0001.xhtml
# http://hbcponline.com/faces/documents/01_10/01_10_0001.xhtml 
# (accessed August 2020)
# http://hbcponline.com/faces/documents/10_04/10_04_0001.xhtml
# http://hbcponline.com/faces/documents/04_01/04_01_0001.xhtml
# (accessed March 2021)

h_si = 6.62607015e-34  # Planck's constant in J*s (SI units)
nA = 6.02214076e23 # Avogadro's constant
c = 299792458*100 # speed of light cm/s
kj2cm = 83.59350 # kJ/mol to cm-1 conversion
ev2cm = 8065.54 # eV to cm-1 conversion
ghz2cm = 0.0333564 # GHz to cm-1 (for rotations)
ev2kcal = 23.06054  # eV to kcal/mol conversion
kcal2cm = 349.7552 # kcal/mol to cm-1 conversion
h = h_si * (kj2cm / 1000) * nA # Planck's constant in cm-1 * s

# For sigmasim
# convert permittivity to e^2/cm-1
vac_permittivity = 8.8541878128e-12 / 1.602176634e-19 / 100 / ev2cm
# polarizabilities (in angstrom^3)
polarizability = {'Ne': 0.39432,
                  'Ar': 1.6411,
                  'Xe': 4.044}
polarizability = {i:polarizability[i] * 1e-24 for i in polarizability} # convert to cm^3
# molar masses
gas_molmass = {'Ne': 20.180,
               'Ar': 39.948,
               'Xe': 131.293}
kB_si = 1.380649e-23 # Boltzmann constant in J/K (SI units)
kB = kB_si * (kj2cm / 1000) * nA # Boltzmann constant in cm-1/K
