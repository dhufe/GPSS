import numpy as np
from datetime import datetime  
from GPSS import GPSS
from GPSSPlot import GPSSPlot
import os
import h5py as hd

# centre frequency of the pulse source 
fm = 100e3
c = 343 
# spatial resolution (should be at least a fifth of the wavelength)
ds = (c/fm)/10

SourceWidth = 20e-3
SourchHeigth = 10e-3

X = np.arange ( -15e-3, 15e-3, ds)
Y = np.arange ( -15e-3, 15e-3, ds)
Z = np.arange ( 0, 150e-3, ds )

# build 3D mesh
# indexing type 'ij' is necessary ?
#Xmesh, Ymesh, Zmesh = np.meshgrid(X, Y, Z, indexing='ij')
Xmesh, Ymesh, Zmesh = np.meshgrid(X, Y, Z )

p = np.zeros ( shape=Xmesh.shape, dtype=np.double )

now = datetime.now()
prefix = 'simdata/' + now.strftime("%Y%m%d-%H%M")
print ( 'Data ist stored under : %s.' % (prefix) )

fileName = prefix + '/SimpleRect3D'

def SaveData ( fileName, Xmesh, Ymesh, Zmesh, pdata, paramDict = None ):
    with hd.File( fileName + '.mat', 'w') as fd:

        if paramDict:
            for key in paramDict.keys():
                fd.attrs[key] = paramDict[key]

        fd.create_dataset('Xmesh', data=Xmesh,compression="gzip", compression_opts=9)
        fd.create_dataset('Ymesh', data=Ymesh,compression="gzip", compression_opts=9)
        fd.create_dataset('Zmesh', data=Zmesh,compression="gzip", compression_opts=9)
        fd.create_dataset('p', data=pdata,compression="gzip", compression_opts=9)

## create path for storing the simulation data (if not exists)
if (os.path.exists ( prefix ) == False ):
    os.mkdir ( prefix )

print ( 'Calculating soundfield @ %3.1f kHz\n' % ( fm*1e-3 ) )
# build acoustical source 
Xs, Ys, Zs = GPSS.BuildRectangularSource(ds, SourceWidth, SourchHeigth  )
# amplitude weighting 
I0 = 1 / Ys.size 
# Calculating the resulting two-dimensional complex field
Is = np.ones ( Xs.shape ) * I0
# Plot source configurations 
GPSSPlot.PlottingSourceConfiguration ( Xs, Ys, Zs, Is, fileName = prefix + '/source_config.png' ) 
# Phase shifting of each simulation source is zero
Phs = np.zeros ( Ys.shape )
# run the calculation
p = GPSS.run_calc_3d(Xs, Ys, Zs, Phs, fm, Xmesh, Ymesh, Zmesh, Is )

paramDict = {
                            "ds":  ds,
                            "Frequency": fm, 
                            "SourceWidth": SourceWidth,
                            "SourchHeigth": SourchHeigth,
                            "MaxAmplitude": np.amax(p),
                            "MinAmplitude": np.amin(p) 
                        }

# save the data
SaveData(fileName, Xmesh, Ymesh, Zmesh, p, paramDict = paramDict )

