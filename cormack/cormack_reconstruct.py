"""NAME
cormack_reconstruct.py
PURPOSE
Reconstruct a 3D image from a set of 2D projections using the Cormack method

CALLING SEQUENCE
result = cormack_reconstruct(x, y, [emiss, par=par], [St], bright, 
                             coords=coords, crea_mat, time=time, status=status)

INPUTS
x: 1D array of x coordinates of the image (radii) on which to reconstruct the 
emissivity distribution. If BRIGTH is set, x is the 1D array of the p vectors
y: 1D array of y coordinates of the image (angles) on which to reconstruct the
emissivity distribution. If BRIGTH is set, y is the 1D array of the phi angles
N.B. x and y must be in the same units (meters)

OPTIONAL INPUTS
emiss: structre containing the coefficients output of the routine 
CORMACK_INVERT (see CORMACK_INVERT for details). If this is provided also the 
par parameter must be provided. Alternatively, the St structure can be given 
in input and leave the routine to extract the par and emiss parameters from it.
par: structure containing the parameters output of the routine CORMACK_SETUP
(see CORMACK_SETUP for details). This has to be provided only if the emiss
parameter is given.
St: structure containing the data, par and emiss. 
coords: 'XY' if the points are specified in the x-y plane, 'RT' if the points
are specified in polar coordinates. Default is 'XY'.
time: time to consider when running the routine. 

KEYWORDS
bright: if set, the routine will reconstruct the brightness distribution.
Otherwise the routine will compute the emissivity distribution. 
crea_mat: if set, the routine will create a matrix starting from the x,y 
vectors, which must be specified (valid only for the emissivity reconstruction). 
OUTPUTS
result: 2D array containing the reconstructed emissivity or brightness
OPTIONAL OUTPUTS
status: status of the routine. If 1, the routine has been executed correctly.
If 0, the routine has not been executed correctly.  

MODIFICATION HISTORY
rewriting the IDL routine in Python - 2024/05/16 
"""

import numpy as np
from scipy import interpolate
from scipy import integrate

import numpy as np

def cormack_reconstr(xin, yin, parametro3, par=None, time=None, coords=None, 
                     status=None, bright=None, crea_mat=None, debug=None):
    if debug:
        ksd = True
    else:
        ksd = False

    status = True

    if not parametro3:
        print("Provide EMISS, PAR, or ST.")
        res = False
        status = False
        return None
    else:
        if 'BRIGHT' in parametro3.keys():
            emiss = parametro3['emiss']
            par = parametro3['par']
            time_array = parametro3['bright']['time']
        else:
            emiss = parametro3
            time_array = parametro3['time']

    if par is not None:
        parn = par.keys()
        idl_set = True
    else:
        idl_set = False

    if 'P_REF' not in parn:
        P_ref = [0.0, 0.0, 0.0]
    else:
        P_ref = par['P_ref']

    if debug:
        top = time.time()

    if par is None:
        if crea_mat:
            nx = len(xin)
            ny = len(yin)

            cc = np.ones((1, ny))
            x2 = np.reshape((xin - P_ref[0]) * cc, nx * ny)

            rr = np.ones(nx)
            y2 = np.reshape((yin - P_ref[2]) * rr, nx * ny)
        else:
            x2 = xin - P_ref[0]
            y2 = yin - P_ref[2]
    else:
        if xin[0] < 0.0:
            x2 = par['p']
            y2 = par['phi']
        else:
            prel_dum = [xin, yin]
            # Assuming the function shift_geom exists
            # shift_geom(xin, yin, prel_dum, radius=par['radius'], P_ref=P_ref, g=g)
            x2 = g['p']
            y2 = g['phi']

    n_pnt = len(x2)

    if debug:
        print("Define x2, y2:", (time.time() - top) * 1.0e3, "ms")

    mc = int(par['mc'])
    ms = int(par['ms'])
    ls = int(par['ls'])

    dummy = min(abs(time[0] - time_array), ind)

    coeff_d = emiss['coeff'][:, ind]
    n_coeff = len(coeff_d)

    if debug:
        top = time.time()

    met = par['base'].upper()

    x2 = x2 / par['null_radius']

    if bright:
        # Assuming RebuildFunction exists and takes appropriate arguments
        status_IDL = RebuildFunction(
            base=met, coord_sys='PF',
            mc=mc, ms=ms, ls=ls, coeff_d=coeff_d, x2=x2, y2=y2, res=res,
            debug=debug
        )

        if not status_IDL:
            res = False
            status = False
            return None

        if debug:
            print("Reconstruction of brightness:", (time.time() - top) * 1.0e3, "ms")
    else:
        coords = coords.upper()
        if coords == 'XY':
            y2 = y2 / par['null_radius']

        status_IDL = RebuildFunction(
            base=met, coord_sys=coords,
            mc=mc, ms=ms, ls=ls, coeff_d=coeff_d, x2=x2, y2=y2, res=res,
            debug=debug
        )

        if not status_IDL:
            res = False
            status = False
            return None

        if crea_mat:
            res = np.reshape(res, (nx, ny))

        res = res / par['null_radius']

        if debug:
            print("Reconstruction of emissivity:", (time.time() - top) * 1.0e3, "ms")

    return res
