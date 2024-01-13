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
fm =340e3
c = 343 
WaveLength = c / fm
# spatial resolution (should be at least a fifth of the wavelength)
ds = WaveLength/5
dAngle = 10
ElementWidth = 3*1.4e-3 + 2*0.1e-3
ElementHeigth = 20e-3
GapWidth = 100e-6
NElements = 8

def BuildRectangularSource ( ds, a, b, Start = [0, 0], PlotSource = False ):
    x = np.arange ( -a/2, a/2, ds ) + Start[1]
    y = np.arange ( -b/2, b/2, ds ) + Start[0]
    Xs, Ys = np.meshgrid ( x, y )
    Xs = Xs.ravel()
    Ys = Ys.ravel()
    Zs = np.zeros ( Xs.shape )

    if PlotSource:
        PlotSources (Xs, Ys, Zs ) 

    return Xs, Ys, Zs

def SaveData ( fileName, Xmesh, Ymesh, Zmesh, pdata, paramDict = None ):
    with hd.File( fileName + '.mat', 'w') as fd:

        if paramDict:
            for key in paramDict.keys():
                fd.attrs[key] = paramDict[key]

        fd.create_dataset('Xmesh', data=Xmesh,compression="gzip", compression_opts=9)
        fd.create_dataset('Ymesh', data=Ymesh,compression="gzip", compression_opts=9)
        fd.create_dataset('Zmesh', data=Zmesh,compression="gzip", compression_opts=9)
        fd.create_dataset('p', data=pdata,compression="gzip", compression_opts=9)

def BuildArraySource ( ds, ElementSize, GapSize, WaveLength, dAngleIncident = 0,  Offset = [0,0], Type = 'Rect', NElement = 8, PlotSource = False ):
    Xs = Ys = Zs = Ps = np.array([])
    # Shift by half elements multiplied with element width (and a half element as gap size)
    # This is done for centering the array 
    dOffSetY = -.5 * ((NElement -1 ) * ElementSize[1] + (NElement - 1) * GapSize ) 
    print ( dOffSetY )
    dOffSetX = 0# -.5 * ElementSize[0] 
    dElementPitch = ElementSize[1] + GapSize 
    dPhase = np.zeros( ( NElement ))

    print ( 'Element size h=%f mm, w=%f mm and g=%f mm\n' % (ElementSize[0]*1e3, ElementSize[1]*1e3, GapSize*1e3))

    for iElement in range(0, NElement ):
        dY = Offset[1] + dOffSetY + ( ElementSize[1] + GapSize ) * iElement 
        dX = Offset[0] + dOffSetX# + ElementSize[1] * iElement 
        dXs, dYs, dZs = BuildRectangularSource( ds, ElementSize[0], ElementSize[1], Start=[dY, dX] )

        ## Element phase shift
        dPhase[ iElement ] = ( -2.0 * np.pi * np.sin ( np.pi * dAngleIncident / 180 ) ) * iElement * dElementPitch  / WaveLength  
        print ( '%d / %d phasehifted by %1.3f / %04.0f ns' % ( iElement, NElement, dPhase[iElement], 1e9 * dPhase[iElement] / (2 * np.pi * 250e3) ) )
        Ps = np.append ( Ps, np.ones( (dXs.size) ) * dPhase[iElement] )

        ## Append new source to the list of source points 
        Xs = np.append ( Xs, dXs )
        Ys = np.append ( Ys, dYs )
        Zs = np.append ( Zs, dZs )

    if PlotSource == True:
        PlotSources(Xs, Ys, Zs, Ps )

    # Ps = np.zeros ( Xs.size ) 

    return Xs, Ys, Zs, Ps, dPhase 

def main():

    X = np.arange ( -150e-3, 150e-3, ds)
    Y = np.arange ( -150e-3, 150e-3, ds)
    Z = np.arange ( 0, 300e-3, ds )

    # build mesh
    Ymesh, Zmesh = np.meshgrid(Y , Z )
    Xmesh = np.zeros ( Ymesh.shape )

    p = np.zeros ( shape=Ymesh.shape, dtype=np.cdouble )

    now = datetime.now()
    prefix = 'simdata/' + now.strftime("%Y%m%d-%H%M")
    print ( 'Data ist stored under : %s.' % (prefix) )

    fileName = prefix + '/SimpleRect_XZ'

    ## create path for storing the simulation data (if not exists)
    if (os.path.exists ( prefix ) == False ):
        os.mkdir ( prefix )

    print ( 'Calculating soundfield @ %3.1f kHz\n' % ( fm*1e-3 ) )
    # build acoustical source 
    #Xs, Ys, Zs, Is, Phs = GPSS.BuildArraySource( ds, ElementSize = [ElementHeigth, ElementWidth ], GapSize=GapWidth,  WaveLength=c/fm, dAngleIncident = dAngle, Offset=[0,0], Type='Rect', NElement = NElements  )
    Xs, Ys, Zs, Phs, dPhase = BuildArraySource( ds, [20e-3, ElementWidth ], GapWidth, WaveLength, dAngle, Offset=[0,0], Type='Rect', NElement = NElements )

    Is =  np.ones ( Xs.shape ) / Xs.size

    # Plot the source configuration
    GPSSPlot.PlottingSourceConfiguration ( Xs, Ys, Zs, Ps = Phs, fileName = prefix + '/source_config.png' ) 
    # run the calculation
    #        run_calc_2d (Xs, Ys, Zs, Ps, freq, Xmesh, Ymesh, Zmesh, I0=1):
    #        gpss_calculation2D (  Xs,  Ys,  Zs,  Phase, Q, f, Xmesh, Ymesh,  Zmesh, p ):
    #pp = GPSS.run_calc_2d (Xs, Ys, Zs, Phs, fm, Xmesh, Ymesh, Zmesh, Is )
    gpss.gpss_calculation2D ( Xs, Ys, Zs, Phs , Is, fm, Xmesh, Ymesh, Zmesh, p )
    pp = np.sqrt ( p.imag*p.imag + p.real*p.real) 
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
                        "MaxAmplitude": np.amax(p),
                        "MinAmplitude": np.amin(p) 
                    }

    # save the data
    SaveData(fileName, Xmesh, Ymesh, Zmesh, pp )

if __name__ == "__main__":
    main()
