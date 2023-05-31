import cython
import sys
import math
from cython.parallel import prange, parallel

from libc.math cimport exp as exp
from libc.math cimport sin
from libc.math cimport cos
from libc.math cimport abs as c_abs
from libc.math cimport M_PI

cimport numpy as np
import numpy as np

cdef extern from "complex.h" nogil:
    double cabs(double complex)
cdef extern from "complex.h" nogil:
    double complex cexp(double complex)

cdef void progress( long count, long total, suffix=''):
    cdef int bar_len     = 60
    cdef int filled_len  = int(round(bar_len * count / float(total)))
    cdef double percents = round(100.0 * count / float(total), 1)
    bar                  = '*' * filled_len + ' ' * (bar_len - filled_len)

    sys.stdout.write('%s [%s] %s%s\r' % (suffix, bar, percents, '%'))
    sys.stdout.flush()  # As suggested by Rom Ruben

cdef double complex pressure1D ( double r, double f, double Phase, double Q, double dAlpha, double c, double P0 = 101300 ) nogil: 
    """
    Computes the complex pressure for a certain distance r
    """
    cdef double lamda     = c / f
    cdef double t         = r / c 
    cdef double omega     = 2*M_PI*f 

    cdef double a         = Q * cos(omega * t + Phase ) * exp(-1*dAlpha*r) * ((lamda/4) / r)
    cdef double b         = Q * sin(omega * t + Phase ) * exp(-1*dAlpha*r) * ((lamda/4) / r)
    return a + 1j*b

def calculateAlpha(f, To=20, Po=101300):
    """
        Computation of the acoustic attenuation in air (or any other type of gas) using the frequency f, the temperature To and the atmospheric pressure.
        The parameters needed for computation are from literature.
    """
    Cpluft                 = 1003.7                         # [J/kg*K] bei 0°C
    Cvluft                 = 717.3                          # [J/kg*K] bei 0°C
    Roluft                 = Po*28.97/(8314*(To+273.15))
    Schallgeschwindigkeit  = calculateC(To)                 # bei gegebenen Lufttemperatur [°C]
    Lamb                   = 0.0242;                        # Wärmeleitfähigkeit der Luft Bei 0°C
    Lambraum               = Lamb + (7.21e-5)*To            # Wärmeleitfähigkeit der Luft bei gegebenen Lufttemperatur [°C]
    Viskositat_dynamisch   = (17.3 + 0.05*(To))*1e-6        # [kg/m*s]  [Pa*s] der Luft bei gegebenen Lufttemperatur [°C]
    Viskositat_kinematisch = (13.52 + 0.1*(To))*1e-6        # [kg/m*s]  [Pa*s]der Luft bei gegebenen Lufttemperatur [°C]

    alfa_air = (np.power((2*np.pi*f),2.0))*(((4/3)*Viskositat_dynamisch)+Viskositat_kinematisch+(Lambraum*((1/Cvluft)-(1/Cpluft))))/(2*Roluft*(np.power(Schallgeschwindigkeit,3.0)))

    return alfa_air

def currentProfile( double I, double r0, r):
    return I * np.exp ( -r**2/r0**2 ) / ( np.pi * r0**2 )

def calculateC(To = 20 ):
    return 343.2*np.sqrt((To+273.15)/293.15)    # bei gegebenen Lufttemperatur [°C]

@cython.boundscheck(False)
@cython.wraparound(False)
def gpss_calculation3D ( double[:] Xs, double[:] Ys, double[:] Zs, double[:] Phase, double[:] Q, double f, double [:, :, : ] Xmesh, double [:, :, :] Ymesh, double [:, :, :] Zmesh, double complex [:, :, :] p ):
    cdef long nX = Xmesh.shape[0]
    cdef long nY = Xmesh.shape[1]
    cdef long nZ = Xmesh.shape[2]
    cdef long nSteps = Xs.size
    cdef long iStep, iX, iY, iZ
    cdef double r = 0
    cdef double c = calculateC(25)
    cdef double complex dcScaling = 0
    cdef double dAlpha = calculateAlpha( f )

    with nogil, parallel(num_threads=4):
        for iStep in prange( nSteps , schedule='dynamic'):
            for iX in range ( nX ):
                for iY in range ( nY ):
                    for iZ in range ( nZ ):
                        r = ((Xmesh[iX, iY, iZ]-Xs[iStep])**2.0 + (Ymesh[iX, iY, iZ]-Ys[iStep])**2.0 + (Zmesh[iX, iY, iZ]-Zs[iStep])**2.0 )**.5
                        p[iX, iY, iZ] += pressure1D( r, f, Phase[iStep], Q[iStep] , dAlpha , c  )

#                    with gil:
#                        progress ( iStep, nSteps, 'Calculating 3D soundfield' )


@cython.boundscheck(False)
@cython.wraparound(False)
def gpss_calculation2D ( double[:] Xs, double[:] Ys, double[:] Zs, double[:] Phase, double[:] Q, double f, double [:, : ] Xmesh, double [:, :] Ymesh, double [:, :] Zmesh, double complex [:, :] p ):
    cdef long nX = Xmesh.shape[0]
    cdef long nY = Xmesh.shape[1]
    cdef long nSteps = Xs.size
    cdef long iStep, iX, iY
    cdef double r = 0 
    cdef double c = calculateC(20)
    cdef double complex dcScaling = 0
    cdef double dAlpha = calculateAlpha( f )
 
    with nogil, parallel(num_threads=4):
        for iStep in prange( nSteps , schedule='dynamic'):
            for iX in range ( nX ):
                for iY in range ( nY ):
                    r = ((Xmesh[iX, iY]-Xs[iStep])**2.0 + (Ymesh[iX, iY]-Ys[iStep])**2.0 + (Zmesh[iX, iY]-Zs[iStep])**2.0 )**.5
                    p[iX, iY] += pressure1D( r, f , Phase[iStep], Q[iStep], dAlpha, c  )
# commented because, printing progress will cost computation time for waiting until stdout is finished
#                with gil:
#                    progress ( iStep, nSteps, 'Calculating 2D soundfield' )
