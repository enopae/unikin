#!/bin/env python3

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
unikin
Python code for unimolecular kinetics by statistical rate theory
Version: 0.1.0
Written by: Eno Paenurk
Available at: https://gitlab.ethz.ch/paenurke/unikin
"""

import argparse as ap
import numpy as np
import pathlib as pl
import json
# local modules
from . import models, opt, misc, statecount, sigmasim

def main():

    """
    Argument parsing
    """

    parser = ap.ArgumentParser()
    
    # Set up parent parser for common arguments
    parent_parser = ap.ArgumentParser(add_help=False)
    parent_group = parent_parser.add_argument_group('General options', 'Options common to all submodules.')
    parent_group.add_argument("-n", "--name", help="Basename for the output files", type=str, default='dummy')
    parent_group.add_argument('--emax', help='If applicable, maximum energy for the calculation (in cm-1)', type=int, default=50010)
    parent_group.add_argument('--egrain', help='If applicable, energy step for the calculation (in cm-1)', type=int, default=10)
    parent_group.add_argument('-w', '--write', help = 'Functions to write into files',
                                choices=['rate','dos','sos','sigma'], action='append', type=str)

    # Set up subparsers for methods
    method_parsers = parser.add_subparsers(help = 'Available methods.')
    parser.add_argument('--json', help = 'Input file in json format', type = str, default = None)
    parser.add_argument('--ncore', help = 'Number of cores for running differential evolution (only affects L-CID and SSACM).', type=int, default=1)

    # Parser for rate models
    rates_parent = ap.ArgumentParser(add_help=False)
    rates_parent.add_argument("-e", "--e0", 
                                help = "E0 value (specify unit if not cm-1: value,unit with kcal for kcal/mol, kj for kj/mol, ev for eV)",
                                type=str, default=str(20000.0)) # string to enable unit specification
    rates_parent.add_argument('--json', help = 'Input file in json format', type = str, default = None)
    opt_parser = rates_parent.add_argument_group('Optimization options', 'General options for parameter optimization for rate or cross section fitting.')
    opt_parser.add_argument('--opt', help = 'Activate optimization. Options are listed below', type=bool, default=False)
    opt_parser.add_argument('--e_guess', help = 'E0 guess for optimization (in kcal/mol)', type=float, default=50.0)
    opt_parser.add_argument('--e_win', help = 'Energy window for opt around the E0 guess (in kcal/mol)', type=float, default=45.0)
    opt_parser.add_argument('--ref_file', help = 'Reference file (2 columns - energy vs rate)', type=str)
    opt_parser.add_argument('--verbose', help = 'Toggle verbose printing for opt.', type=bool, default=False)

    # Parser for CID cross section fitting
    cid_parser_parent = ap.ArgumentParser(add_help=False)
    cid_parser = cid_parser_parent.add_argument_group('CID simulations.', 'Options for CID cross section simulation.')
    cid_parser.add_argument('--cid', help = 'Activate CID simulation. Specific options are listed below.', type=bool, default=False)
    cid_parser.add_argument('-m', '--mass', help = 'Molecular mass of the parent ion.', type=float)
    cid_parser.add_argument('-g', '--gas', help = 'Collision gas.', choices=['Ne','Ar','Xe'], type=str, default='Ar')
    cid_parser.add_argument('--n_ions', help = 'Number of ions for the simulation', type=int, default=1000)
    cid_parser.add_argument('--fwhm', help = 'KED FWHM (usually experimental) (in eV)', type=float, default=1.5)
    cid_parser.add_argument('--tau', help = 'Residence time of the ions (in seconds).', type=float, default=6e-5)
    cid_parser.add_argument('--temp', help = 'Simulation temperature (in K)', type=float, default=313.15)

    # Parser for L-CID
    lcid_parser = method_parsers.add_parser('lcid', help = 'Calculate the L-CID rate.', 
                                            parents=[parent_parser, rates_parent, cid_parser_parent],
                                            formatter_class=ap.ArgumentDefaultsHelpFormatter)
    lcid_parser.set_defaults(method=models.LCID)
    lcid_parser.add_argument("--dof", help = "Degrees of freedom", type=int, default=30)
    lcid_parser.add_argument("--rotors", help = "Number of rotors", type=int, default=0)
    lcid_parser.add_argument("-v", "--v_eff", help = "Effective frequency in cm-1", type=float, default=1000.0)
    lcid_parser.add_argument("-a", "--alpha_prime", help = "Alpha prime in cm-1", type=float, default=6000.0)
    lcid_parser.add_argument('--min_alpha', help = 'Minimum alpha_prime value in opt', type=float, default=5000.0)
    lcid_parser.add_argument('--max_alpha', help = 'Maximum alpha_prime value in opt', type=float, default=6500.0)
    #lcid_parser.add_argument('--energies', help = '(optional) List of energies to calculate the rate at', default=None)
    # maybe will implement energy list option later

    # Set up parser for non-LCID rate models
    nonlcid_parser = ap.ArgumentParser(add_help=False)
    nonlcid_parser.add_argument("--sym", help="Reaction symmetry factor", type=int, default=1)
    nonlcid_parser.add_argument("-i", "--intype", help="Type of the input file", 
                                choices=['densum','freq','txt'], type=str, default='txt')
    nonlcid_parser.add_argument('-r', "--reactant_file", help = "Reactant file", type=str)
    nonlcid_parser.add_argument('-t', "--ts_file", help = "TS file(s)", action='append', type=str)
    nonlcid_parser.add_argument('--B', help = '(optional) Rotational constant for the reactant (in cm-1)', default=None)
    nonlcid_parser.add_argument('--B_ts', help = '(optional) Rotational constant for the TS (in cm-1)', default=None)

    # Parser for RAC-RRKM
    rac_parser = method_parsers.add_parser('rac', help='Calculate the RAC-RRKM rate.', 
                                            parents=[parent_parser, nonlcid_parser, rates_parent, cid_parser_parent],
                                            formatter_class=ap.ArgumentDefaultsHelpFormatter)
    rac_parser.set_defaults(method=models.RACRRKM)
    
    # Parser for VTST
    vtst_parser = method_parsers.add_parser('vtst', help='Calculate the VTST rate.',
                                            parents=[parent_parser, nonlcid_parser, rates_parent, cid_parser_parent],
                                            formatter_class=ap.ArgumentDefaultsHelpFormatter)
    vtst_parser.set_defaults(method=models.VTST)

    # Parser for PSL
    psl_parser = method_parsers.add_parser('psl', help='Calculate the PSL rate.', 
                                            parents=[parent_parser, nonlcid_parser, rates_parent, cid_parser_parent],
                                            formatter_class=ap.ArgumentDefaultsHelpFormatter)
    psl_parser.set_defaults(method=models.PSL)

    # Parser for SSACM
    ssacm_parser = method_parsers.add_parser('ssacm', help='Calculate the SSACM rate.', 
                                            parents=[parent_parser, nonlcid_parser, rates_parent, cid_parser_parent],
                                            formatter_class=ap.ArgumentDefaultsHelpFormatter)
    ssacm_parser.set_defaults(method=models.SSACM)
    ssacm_parser.add_argument('-c', '--ssacm_c', 
                            help = 'The c parameter for f_rigid calculation (specify unit if not cm-1: value,kcal for kcal/mol, kj for kj/mol, ev for eV)', 
                                type=str, default=str(5000.0))
    ssacm_parser.add_argument('-d', '--disstype', help = 'Type of dissociation', 
                            choices=['ion-mol', 'ion-atom'], type=str, default='ion-mol')

    # Parser for DOS and SOS calculator
    dos_parser = method_parsers.add_parser('dos', help='Calculate DOS and/or SOS from various data.', 
                                            parents=[parent_parser],
                                            formatter_class=ap.ArgumentDefaultsHelpFormatter)
    dos_parser.add_argument('--bs', help = 'File with frequencies (in cm-1) for Beyer-Swinehart count', type=str)
    dos_parser.add_argument('--rot1d', help = 'B value (in cm-1) for calculating 1D rotor DOS and/or SOS', type=float)
    dos_parser.add_argument('--rot2d', help = 'B value (in cm-1) for calculating 2D rotor DOS and/or SOS', type=float)
    dos_parser.add_argument('--convolve', help = 'Convolve the data from two files (file names comma-separated; should have the same length)', type=str)
    dos_parser.add_argument('--dens2dat', help = 'Densum output file to convert data from into a 2-column text file', type=str)


    """
    Parse the arguments and run the code
    """

    # Collect the arguments
    args = parser.parse_args()

    # If given, read in json input file and overwrite the arguments
    if args.json:
        # Make temp argument dict
        temp_args = vars(args)
        with open(args.json, 'r') as jf:
            inp = json.load(jf)
            for i in inp:
                # Read out the method
                if i == 'method':
                    if inp[i] == 'rac':
                        temp_args[i] = models.RACRRKM
                    elif inp[i] == 'psl':
                        temp_args[i] = models.PSL
                    elif inp[i] == 'ssacm':
                        temp_args[i] = models.SSACM
                    elif inp[i] == 'vtst':
                        temp_args[i] = models.VTST
                    elif inp[i] == 'lcid':
                        temp_args[i] = models.LCID
                    else:
                        print('Unknown method "'+str(inp[i])+'" selected.')
                        print('Available options: rac, psl, ssacm, vtst, lcid.')
                        exit(1)
                # Read out other options
                else:
                    temp_args[i] = inp[i]
            jf.close()
        # Convert the changed arguments into ap namespace
        args = ap.Namespace(**temp_args)


    ### For rate models (defined by the existence of the e0 variable)
    if 'e0' in args:

        """
        Set up the variables for calculations
        """

        # Make a new arguments dict, exclude method-irrelevant keywords
        # with more options coming in, this is no longer convenient, should try to replace. ##
        non_method_args = ['method', 'name', 'write', 'jobtype', 'e_guess', 'e_win', 'min_alpha',
                           'max_alpha', 'ref_file', 'opt', 'cid', 'mass', 'gas', 'fwhm', 'tau', 
                           'n_ions', 'temp', 'verbose', 'json', 'ncore']
        method_args = {i:vars(args)[i] for i in vars(args) if i not in non_method_args}


        # VTST special care
        if args.method.__name__ == 'VTST':
            # Read e0 file names from the e0 argument
            method_args['e0'] = misc.read_file(args.e0)
            # Read TS file names from the ts argument (the files must be in the same directory as the file list)
            ts_filepath = pl.Path(args.ts_file[0]).parents[0]
            method_args['ts_file'] = [str(ts_filepath)+'/'+i[0] for i in misc.read_file(args.ts_file[0])]
            if args.B_ts:
                # Read TS B values from the B_ts argument
                method_args['B_ts'] = [float(i[0]) for i in misc.read_file(args.B_ts)]
        
        # Somewhat annoying workaround to have input compatible with VTST and the rest
        # To be maybe changed later
        if hasattr(args, 'B'): # to avoid LCID crash
            if args.B and args.B_ts:
                if not args.method.__name__ == 'VTST':
                    method_args['B_ts'] = float(args.B_ts)
                method_args['B'] = float(args.B)

        # Convert units if necessary
        def _convert(ene):
            if ',' in ene:
                value, unit = ene.split(',')
                return misc.convert2cm(value, unit)
            else:
                return float(ene)
        
        # e0 converter
        if args.method.__name__ == 'VTST': # loop for vtst
            for i, e in enumerate(method_args['e0']):
                method_args['e0'][i] = _convert(e[0])
        else:
            method_args['e0'] = _convert(method_args['e0'])
        
        # SSACM c converter
        if hasattr(args, 'ssacm_c'):
            if ',' in method_args['ssacm_c']:
                value, unit = method_args['ssacm_c'].split(',')
                method_args['ssacm_c'] = misc.convert2cm(value, unit)
            else:
                method_args['ssacm_c'] = float(method_args['ssacm_c'])

        # Set up the rate calculator
        calculator = args.method(**method_args)

        # Toggle j-averaged rate calculation if B and B_ts given
        if hasattr(args, 'B'):
            if args.B and args.B_ts:
                calculator.j_rate = True

        # Set up the simulator if CID is requested
        if args.cid:
            simulator = sigmasim.SigmaSim(args.mass, args.gas,
                        args.fwhm, args.temp, args.tau, args.n_ions,
                        args.emax, args.egrain)
            # Rescale energies for LCID so that sigma simulator can draw rate data from higher energies
            if args.method.__name__ == 'LCID':
                calculator.energies = np.arange(0, 1.01*simulator.emax_scale * args.emax, args.egrain)
                calculator.calc_phasespace()


        """
        Perform the requested calculations
        """

        # Optimize the parameters if toggled
        if args.opt:
            # Read in the reference
            ref = np.loadtxt(args.ref_file).T
            ref_ene = ref[0]
            ref_data = ref[1]
            # Define parameters for L-CID
            if hasattr(args, 'min_alpha'):
                min_alpha = args.min_alpha
                max_alpha = args.max_alpha
            else:
                min_alpha = max_alpha = 0 # just so they're defined
            
            # Optimize either cross section or rate
            if args.cid:
                # Set up optimizer
                optimizer = opt.Optimizer(calculator, simulator, ncore = args.ncore, verbose = args.verbose)
                opt_result = optimizer.optimize(ref_ene, ref_data, opt_type = 'sigma',
                        e_guess = args.e_guess,
                        e_win = args.e_win,
                        min_alpha = min_alpha,
                        max_alpha = max_alpha)
            else:
                # Set up optimizer
                optimizer = opt.Optimizer(calculator, ncore = args.ncore, verbose = args.verbose)
                opt_result = optimizer.optimize(ref_ene, ref_data, opt_type = 'rate',
                        e_guess = args.e_guess,
                        e_win = args.e_win,
                        min_alpha = min_alpha,
                        max_alpha = max_alpha)

            # Write the optimization results into a file
            for r in opt_result:
                if hasattr(calculator, r):
                    setattr(calculator, r, opt_result[r])
            misc.write_opt_result(args.name, opt_result)

        # Calculate the cross section
        if args.cid:
            sigma, energies = simulator.calc_sigma(calculator)
        # Calculate the rate
        calculator.calc_rate()

        # Write the selected result files
        if args.write:
            for w in args.write:
                if not w == 'sigma':
                    misc.write_2col_rate_result(args.name, calculator, w)
                elif w == 'sigma':
                    misc.write_2col_result(args.name, energies, sigma, 'sigma')


    # If E0 not present in the keywords, then perform a density/sum of states calculation
    # (possibly temporary implementation)
    else:
        # Only perform convolution
        if args.convolve:
            file1, file2 = args.convolve.split(',')
            data1 = np.loadtxt(file1).T[1]
            data2 = np.loadtxt(file2).T[1]
            energies = np.loadtxt(file1).T[0]
            data = np.convolve(data1, data2)[:len(energies)]
            data[1:] *= (energies[1]-energies[0]) # multiply by egrain
            misc.write_2col_result(args.name, energies, data, 'conv')

        # Calculate the density/sum of states with the specified method
        else:
            if args.write:
                for w in args.write:
                    energies = np.arange(0,args.emax, args.egrain)
                    if args.bs:
                        freq = np.loadtxt(args.bs)
                        data = {w: statecount.beyer_swinehart(freq, args.egrain, args.emax, count = w)}
                    elif args.rot1d:
                        data = statecount.rot1D(args.rot1d, energies)
                    elif args.rot2d:
                        data = statecount.rot2D(args.rot2d, energies)
                    elif args.dens2dat:
                        data = misc.read_densum(args.dens2dat)
                        energies = data['ene']
                    
                    misc.write_2col_result(args.name, energies, data[w], w)
        
