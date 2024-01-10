import numpy as np
#import matplotlib
#matplotlib.use('Agg')

#import matplotlib.pyplot as plt 
from datetime import datetime  
from GPSS import GPSS
from GPSSPlot import GPSSPlot
import os
import h5py as hd

# centre frequency of the pulse source 
fm =250e3
c = 343 
# spatial resolution (should be at least a fifth of the wavelength)
ds = (c/fm)/5

X = np.arange ( -20e-3, 20e-3, ds)
Y = np.arange ( -20e-3, 20e-3, ds)
Z = np.arange ( 0, 200e-3, ds )

# build mesh
Ymesh, Zmesh = np.meshgrid(Y , Z )
Xmesh = np.zeros ( Ymesh.shape )

p = np.zeros ( shape=Ymesh.shape, dtype=np.double )

now = datetime.now()
prefix = 'simdata/' + now.strftime("%Y%m%d-%H%M")
print ( 'Data ist stored under : %s.' % (prefix) )

fileName = prefix + '/SimpleRect_YZ'

def SaveData ( fileName, Xmesh, Ymesh, Zmesh, pdata ):
    with hd.File( fileName + '.mat', 'w') as fd:
        fd.create_dataset('Xmesh', data=Xmesh,compression="gzip", compression_opts=9)
        fd.create_dataset('Ymesh', data=Ymesh,compression="gzip", compression_opts=9)
        fd.create_dataset('Zmesh', data=Zmesh,compression="gzip", compression_opts=9)
        fd.create_dataset('p', data=pdata,compression="gzip", compression_opts=9)

## create path for storing the simulation data (if not exists)
if (os.path.exists ( prefix ) == False ):
    os.mkdir ( prefix )

print ( 'Calculating soundfield @ %3.1f kHz\n' % ( fm*1e-3 ) )
# build acoustical source 
Xs, Ys, Zs = GPSS.BuildRectangularSource(ds, 5e-3, 10e-3  )
# amplitude weighting 
I0 = 1 / Ys.size 
# Calculating the resulting two-dimensional complex field
Is = np.ones ( Xs.shape ) * I0
# Phase shifting of each simulation source is zero
Phs = np.zeros ( Ys.shape )
# run the calculation
p = GPSS.run_calc_2d(Xs, Ys, Zs, Phs, fm, Xmesh, Ymesh, Zmesh, Is )
# Plot soundfield 
GPSSPlot.PlotFieldData( fileName, p, Ymesh, Zmesh)
# save the data
SaveData(fileName, Xmesh, Ymesh, Zmesh, p )
