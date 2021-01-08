# UniKin

Python code for unimolecular kinetics by statistical rate theory. Available models include: RAC-RRKM, (microcanonical) VTST, PSL, SSACM, and L-CID. All these models can be used for either microcanonical rate or CID reaction cross-section calculations.

## Basic usage
Call python on the `unikin.py` code with `-h` flag to see the options. 
```
python unikin.py -h
```

For each option, you can see the respective flags by invoking the option and calling the help feature again. For example, for SSACM calculation options:
```
python unikin.py ssacm -h
```

#### Usage recommendation
Generate a soft link to the python code in your `bin` directory:
```
ln -s path/to/unikin.py unikin
chmod 755 unikin
```

You can now call `unikin` in any directory as just:
```
unikin -h
```

Instead of specifying keyword flags on the command line, the calculations can also be performed by using an input file in `.json` format. In that case, the code should be called as:
```
unikin --json input.json
```
A template input file for each of the rate models is provided in the input_templates directory. The keywords in the input files are the same as the ones used on the command line, so the description of them can also be found by calling `unikin <method> -h`. Note that it is advised not to remove unnecessary keywords (e.g., the CID simulation keywords when cid is set to false), as this has not been extensively tested.


### Calculation job types
The code can be used to either calculate the rate or the reaction-cross section based on input data, to optimize the variables in the rate model with respect to reference data, or to calculate/manipulate density of states (DOS) and sum of states (SOS). CID cross-section simulation can be performed by invoking `--cid True` and optimization by invoking `--opt True` with any of the rate models. The DOS/SOS calculations can be accessed by calling `unikin dos`.

#### Notes on opt
The parameter optimization method requires a file with reference rates, specified by `--ref_file`. The reference file should include energies in cm-1 in the first column and the respective rates, or cross-sections for CID simulations, in the second column. The energies for the CID simulations have to be the center of mass energies. Additionally, only the reactant and TS files (or degrees of freedom and number of rotors for L-CID) are required as input.

The parameters that are optimized for each model are:

| Model | Keyword | Optimized parameters |
| --- | --- | --- |
| RAC-RRKM | rac | e0 |
| PSL | psl | e0 |
| SSACM | ssacm | e0, ssacm_c |
| VTST | vtst | e0 |
| L-CID | lcid | e0, v_eff, alpha_prime |

These parameters do not have to be specified if opt is selected, except for VTST (it requires the energies for the PES, which it then scales during the optimization).

The range for e0 optimization can be changed by specifying `--e_guess` as the middle value (default 50 kcal/mol) and `--e_win` for the value range around e_guess (default 45 kcal/mol). The ssacm_c parameter is optimized in the value range of 0 to 5 eV. The ranges for L-CID parameters are described below.

If `--verbose True` is invoked, then each of the fitting parameter values is printed at each step to stdout.

### Writing out the results
All the result filenames include the name given by the `-n` flag. The optimization automatically writes a log file including the optimized parameters (in cm-1) and the RMSD between the reference data and calculated data. Additionally, the rate, the cross-section (called sigma), the DOS, and/or the SOS can be written into a two-column file by invoking the respective keywords with the `-w` flag (can be specified several times).

In the `dos` method, convolution will automatically write an output file. For other options, `-w` should be invoked to specify whether DOS or SOS is required (or both by invoking the flag twice with either keyword).

### Parallelization

The differential evolution optimization for L-CID and SSACM can be run with more than one core (tested with up to 12). The number of cores can be specified with the `--ncore` flag. This can be invoked together with the `--json` flag, too. For example, for optimization with four cores:

```
unikin --json lcid_or_ssacm_opt_input.json --ncore 4
```

## Simulations

### Input file types
The input files for the rate models can be given in three formats, defined by the `-i` or `--intype` flag.

| intype | Description |
| --- | --- |
| txt | Two-column text file with the data; first column energies in cm-1, second column data |
| densum | Multiwell output file from the densum or paradensum module |
| freq | One-column text file with vibrational frequencies in cm-1 (will invoke Beyer-Swinehart count) |

For `densum` and `freq`, the code will either extract/calculate DOS or SOS, depending on the model and whether it's for the reactant or TS.

#### Input files for different models
Each model requires one reactant input file, specified with the `-r` or `--reactant_file` flag. The input file(s) for the transition state are specified with the `-t` or `--ts_file` flag, and depend on the method as described below. Multiple TS files can be entered by invoking the `-t` flag several times.

| Model | Keyword | `-t` input |
| --- | --- | --- |
| RAC-RRKM | rac | One file |
| PSL | psl | Multiple files (rot SOS/DOS and vib DOS) |
| SSACM | ssacm | Like PSL, rot SOS/DOS should be entered first |
| VTST | vtst | One file containing the filenames for each TS |

### Energy input
For most energy input variables, the energy unit is predefined in the current version. For E0 and SSACM c value, the unit can be specified with value, unless the value is in cm-1. For example, to use kcal/mol for E0 and eV for c:
```
unikin ssacm -r <reactant_file> -t <TS_rot_file> -t <TS_vib_file> -e 35,kcal -c 0.07,ev
```
Other unit definitions can be found by calling `--help`.

#### Energy ranges
If `txt` or `densum` files are used as an input, the energy range for the rate calculation is defined by the energies in the reactant DOS file (should be the same for TS files). For `freq`, the energy range should be set by specific `--emax` for the highest energy and `--egrain` for the energy step (in cm-1). The `emax` and `egrain` parameters should also be specified for `dos` calculations, except convolution.

The `emax` and `egrain` parameters must be specified for CID simulations, regardless of the input type; in addition, the input data should cover at least approximately 20% higher energies than `emax` for the CID simulation (because CID involves distributions that can have energy values above `emax`.

### J-averaged rates
To calculate J-averaged rates, `--B` and `--B_ts` parameters have to be defined (then J-averaging is invoked automatically), corresponding to the rotational constant of the reactant and the TS in cm-1, respectively. Note: this is considerably slower than the regular rate calculation.

#### Special notes about VTST
VTST requires several TS instances to be defined. For each of the parameters that is otherwise only specified per one TS in other models, VTST expects one-column text files that list these parameters. Specifically, `-t` expects a file with a list of TS data files (path relative to the list file), `-e` expects a file with a list of TS E0 values, `--B_ts` expects a file with a list of B values.

#### Special notes about L-CID
L-CID requires no data files for DOS and SOS, therefore `-r` and `-t` flags are not available. DOS and SOS are calculated based on degrees of freedom `--dof`, number of rotors `--rotors`, effective frequency `--v_eff`, and alpha prime `--alpha_prime`. In optimization jobs, the effective frequency is optimized in the range of 0 to 2000 cm-1, and alpha prime optimization range can be specified by `--min_alpha` and `--max_alpha`.

### Misc
The code does not currently include error and exception handling; the user might experience hard-to-trace Python errors if a mistake in the usage is made.

SSACM and L-CID fitting can be parallelized by setting a different number of workers for differential evolution in the source code of `opt.py`.
