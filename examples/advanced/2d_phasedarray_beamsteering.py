import numpy as np
#import matplotlib
#matplotlib.use('Agg')

#import matplotlib.pyplot as plt 
from datetime import datetime  
from GPSS import GPSS
import gpss
from GPSSPlot import GPSSPlot
import os
import h5py as hd

# centre frequency of the pulse source 
fm =250e3
c = 343 
WaveLength = c / fm
# spatial resolution (should be at least a fifth of the wavelength)
ds = WaveLength/5
ElementWidth = 1.0e-3
ElementHeigth = 20e-3
GapWidth = 100e-6
NElements = 8 

def SaveData ( fileName, Xmesh, Ymesh, Zmesh, pdata, paramDict = None ):
    with hd.File( fileName + '.mat', 'w') as fd:

        if paramDict:
            for key in paramDict.keys():
                fd.attrs[key] = paramDict[key]

        fd.create_dataset('Xmesh', data=Xmesh,compression="gzip", compression_opts=9)
        fd.create_dataset('Ymesh', data=Ymesh,compression="gzip", compression_opts=9)
        fd.create_dataset('Zmesh', data=Zmesh,compression="gzip", compression_opts=9)
        fd.create_dataset('p', data=pdata,compression="gzip", compression_opts=9)

def main():

    idx = 0

    now = datetime.now()
    prefix = 'simdata/' + now.strftime("%Y%m%d-%H%M")
    print ( 'Data ist stored under : %s.' % (prefix) )

    fileName = prefix + '/PhasedArray_YZ'

    ## create path for storing the simulation data (if not exists)
    if (os.path.exists ( prefix ) == False ):
        os.mkdir ( prefix )

    print ( 'Calculating soundfields @ %3.1f kHz\n' % ( fm*1e-3 ) )

    dSteeringAngles = np.arange ( -45, 45, 1.0 )

    for dAngle in dSteeringAngles:

        X = np.arange ( -200e-3, 200e-3, ds)
        Y = np.arange ( -200e-3, 200e-3, ds)
        Z = np.arange ( 0, 200e-3, ds )

        # build mesh
        Ymesh, Zmesh = np.meshgrid(Y , Z )
        Xmesh = np.zeros ( Ymesh.shape )


        # create new filename
        fileName = prefix + "/{:04d}_2d_phasedarray_beamsteering".format ( int(idx) )

        # build acoustical source 
        Xs, Ys, Zs, Is, Phs, dPhase = GPSS.BuildArraySource( ds, [20e-3, ElementWidth ], GapWidth, WaveLength, dAngle, Offset=[0,0], Type='Rect', NElement = NElements )
        
        Is =  np.ones ( Xs.shape ) / Xs.size

        # run the calculation
        pp = GPSS.run_calc_2d (Xs, Ys, Zs, Phs, fm, Xmesh, Ymesh, Zmesh, Is )
        # Plot soundfield 
        GPSSPlot.PlotFieldData( fileName, pp, Ymesh, Zmesh)

        paramDict = {
                            "Phi": dAngle,
                            "ds":  ds,
                            "Frequency": fm,
                            "Phase": dPhase, 
                            "ElementWidth": ElementWidth,
                            "ElementHeigth": ElementHeigth,
                            "GapWidth": GapWidth,
                            "MaxAmplitude": np.amax(pp),
                            "MinAmplitude": np.amin(pp) 
                        }

        # save the data
        SaveData(fileName, Xmesh, Ymesh, Zmesh, pp )
        idx+=1

if __name__ == "__main__":
    main()
