import numpy as np
#import matplotlib
#matplotlib.use('Agg')

#import matplotlib.pyplot as plt 
from datetime import datetime  
from pss import GPSS
#from GPSSPlot import GPSSPlot
import os
import h5py as hd

df = 25e3 
fmax = 1.2e6
c = 343 
ds = (c/fmax)/5
pulsewidth = 1e-6
Curv = [23e-3, 69e-3, 114e-3 ]
iCurvIndex = 1
X = np.arange ( -20e-3, 20e-3, ds)
Y = np.arange ( -20e-3, 20e-3, ds)
Z = np.arange ( 0, 200e-3, ds )

Xmesh, Zmesh = np.meshgrid(X , Z )
Ymesh = np.zeros ( Xmesh.shape )

p = np.zeros ( shape=Ymesh.shape, dtype=np.double )

now = datetime.now()
prefix = 'data/' + now.strftime("%Y%m%d-%H%S")
print (prefix) 
fileName = prefix + '/SphericallyRect_Rc_' + str(int(Curv[iCurvIndex]*1e3)) + '_mm_XZ'


def thermoacoustic_source ( f, t_pulse, rc ):
    ## constants
    f_prf = 50                                              # pulse repeatition frequency
    Pe =  f_prf * t_pulse * 150**2 / 9.6                     # Watt
    #Pe =  400**2 / 9.6                     # Watt
    T0 =  293.15
    k = 1

    ### Substrate 
    lamb_sub0 = 1.3                                         # heat conduction substrate [W * (m*K)^-1]
    rho_sub = 2201 
    cp_sub = 78.75
    ### Gas (Air)
    lamb_gas0 = 0.0242                                      # heat conduction gas [W * (m*K)^-1]
    cv_gas = 718
    cp_gas = 1005
    sos_gas = 343.2*np.sqrt(T0/293.15)                      # speed of sound 
    eta = 17.8e-6                                           # viscosity [Pa*s]
    rho_gas = 1.204                                         # density
    h_gas = 20                                              # convective heat conduction 
    ### Device 
    sigma = 5.67037321e-8                                   # Stefan-Boltzmann constant
    epsi_dev = .95                                          # emissivity of the device material
    fthi_dev = 200e-9                                       # film thickness of the coating 
    Rc = 69e-3
    R = 25e-3 * .5 
    a = 17.25e-3
    b = 17.54e-3
    S_dev = 4 * rc**2 * np.arcsin( np.tan ( a / (2*rc) )*np.tan ( b / (2*rc) ) )
    rho_dev = 7200
    cp_dev = 230 

    coeff = [ epsi_dev*sigma*k*S_dev, 0, 0, h_gas*k*S_dev, -1*(epsi_dev*sigma*k*S_dev*(T0**4))-((h_gas*k*S_dev)*T0)-Pe ]
    r = np.roots (coeff)
    ## Transducers temperature in Kelvin
    Tb = r[np.where( (r.imag == 0) & ( r > 0) )].real[0]
    ## Get temperature in Celsius
    Tc = Tb - 273.15
    print ( 'Input power %1.3f  W' % ( Pe ) )
    print ( 'Temperature %3.1f  K' % ( Tb ) )
    print ( 'Temperature %3.1f  C' % ( Tc ) )

    ## Recalculate the thermal parameters using the actual temperature of the device
    ### Substrate
    lamb_sub = lamb_sub0 + 7.6e-3*Tc
    alpha_sub = np.sqrt ( lamb_sub * rho_sub * cp_sub )     # thermal penetration coefficient
    a_sub = lamb_sub / (cp_sub * rho_sub)                   # thermal diffusion

    ### Gas 
    lamb_gas = lamb_gas0 + 7e-5*Tc                  
    alpha_gas = np.sqrt ( lamb_gas * rho_gas * cp_gas )     # thermal penetration coefficient 
    a_gas = lamb_gas / (cp_gas * rho_gas )                  # thermal diffusion

    dQ = Pe / f                                             # amount of input energ
    dSourceR = sos_gas / (2*f)                              # sphere radius for thermodynamic calculation

    ### calculate the heat capacities of the system
    #### Gas 
    pd_gas = np.sqrt ( a_gas / ( f * 2*np.pi ) )            # thermal penetration depth
    m_gas = S_dev * pd_gas * rho_gas                        # heated mass 
    hcap_gas = m_gas * cp_gas                               # heat capcacity of the gas 
    #### Substrate 
    pd_sub = np.sqrt( a_sub / ( f * 2 * np.pi ) )            # thermal penetration depth 
    m_sub = S_dev * pd_sub * rho_sub                        # heated mass 
    hcap_sub = m_gas * cp_sub                               # heat capcacity of the substrate
    #### Device (film)
    m_dev = S_dev * fthi_dev * rho_dev                      # mass of the heated film 
    hcap_dev = cp_dev * m_dev                               # heat capcacity of the coating 

    N = 1                                                   # number of sources 
    V_gas = (1/12)*np.pi*dSourceR**3 
    Q_gas = ( dQ / N ) * ( alpha_gas / ( alpha_gas + alpha_sub + fthi_dev * rho_dev * cp_dev * np.sqrt( 2*np.pi*f ) ) )
    P_gas = (Q_gas / V_gas )
    return P_gas

def rect_puls ( f, tpw ):
    dPulse = np.abs ( np.sin ( f * np.pi * pulsewidth ) / ( np.pi * f ) )
    return dPulse / dPulse[0]

def SaveData ( fileName, Xmesh, Ymesh, Zmesh, pdata ):
    with hd.File( fileName + '.mat', 'w') as fd:
        fd.create_dataset('Xmesh', data=Xmesh,compression="gzip", compression_opts=9)
        fd.create_dataset('Ymesh', data=Ymesh,compression="gzip", compression_opts=9)
        fd.create_dataset('Zmesh', data=Zmesh,compression="gzip", compression_opts=9)
        fd.create_dataset('p', data=pdata,compression="gzip", compression_opts=9)


f = np.arange ( df, fmax + df, df) 

A_Rect = rect_puls(f, pulsewidth)
A_TA   = thermoacoustic_source(f, pulsewidth, Curv[iCurvIndex] )
idx = 0

if (os.path.exists ( prefix ) == False ):
    os.mkdir ( prefix )

for iFreq in np.arange ( df, fmax + df, df):
    X = np.arange ( -20e-3, 20e-3, ds)
    Y = np.arange ( -20e-3, 20e-3, ds)
    Z = np.arange ( 0, 100e-3, ds )

    Xmesh, Zmesh = np.meshgrid(X , Z )
    Ymesh = np.zeros ( Xmesh.shape )

    print ( 'Calculating soundfield @ %3.1f kHz\n' % ( iFreq * 1e-3 ) )
    # build acoustical source 
    Xs, Ys, Zs = GPSS.BuildSphericallyRectSource(ds, 17.54e-3, 17.25e-3, 12.5e-3, Curv[iCurvIndex] )
#    GPSSPlot.PlottingSourceConfiguration(Xs, Ys, Zs)
    I0 = A_Rect[idx] * A_TA[idx] / Xs.size 
    # Calculating the resulting two-dimensional complex field
    Is = np.ones ( Xs.shape ) * I0
    Phs = np.zeros ( Xs.shape )
    dP = GPSS.RunCalculation2D(Xs, Ys, Zs, Phs, iFreq, Xmesh, Ymesh, Zmesh, Is )
    # weighting using rectangular window 
    #dP *= A_Rect[idx] * A_TA[idx]
    # apply to the cummulative pressure field 
    sActualFreq = "{:0>4d}".format(int(iFreq*1e-3)) 
    SliceName = prefix + '/SphericallyRect_Rc_' + str(int(Curv[iCurvIndex]*1e3)) + '_mm_XZ_' + sActualFreq + '_kHz'
    SaveData(SliceName, Xmesh, Ymesh, Zmesh, dP)
    p += dP
    idx = idx + 1
pp = np.sqrt(p.real*p.real + p.imag*p.imag)
GPSSPlot.PlotFieldData( prefix  + fileName, p, Xmesh, Zmesh)
SaveData(fileName, Xmesh, Ymesh, Zmesh, p )
