import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt 

from pss import GPSS
from GPSSPlot import GPSSPlot


df = 25e3 
fmax = 250e3
c = 343 
ds = (c/fmax)/10 

Curv = [23e-3, 69e-3, 114e-3 ]

X = np.arange ( -20e-3, 20e-3, ds)
Y = np.arange ( -20e-3, 20e-3, ds)
Z = np.arange ( 0, 100e-3, ds )

Ymesh, Zmesh = np.meshgrid(Y , Z )
Xmesh = np.zeros ( Ymesh.shape )

p = np.zeros ( shape=Ymesh.shape, dtype=np.cdouble )
fileName = 'SphericallyCurvedRectSourceFieldData_Rc_' + str(int(Curv[0]*1e3)) + '_mm_YZ_Complex'

for iFreq in np.arange ( df, fmax, df ):
    X = np.arange ( -20e-3, 20e-3, ds)
    Y = np.arange ( -20e-3, 20e-3, ds)
    Z = np.arange ( 0, 100e-3, ds )

    Ymesh, Zmesh = np.meshgrid(Y , Z )
    Xmesh = np.zeros ( Ymesh.shape )

    print ( 'Calculating soundfield @ %3.1f kHz\n' % ( iFreq * 1e-3 ) )
    # build acoustical source 
    Xs, Ys, Zs = GPSS.BuildSphericallyRectSource(ds, 17.54e-3, 17.25e-3, 12.5e-3, Curv[0] )
    I0 = 1.0 / Xs.size 
    # Calculating the resulting two-dimensional complex field
    p += GPSS.RunCalculation2DComplex(Xs, Ys, Zs, iFreq, Xmesh, Ymesh, Zmesh, I0 )

pp = np.sqrt(p.real*p.real + p.imag*p.imag)
GPSSPlot.PlotFieldData( fileName, pp, Ymesh, Zmesh)
