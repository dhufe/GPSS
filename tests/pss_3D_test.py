import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.colors as colors
import seaborn as sns

sns.set()
sns.set_style("white")
plt.style.use('seaborn-deep')

plt.rcParams['figure.figsize'] = (12, 8)

fig_prefix = ""

matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)
colormap = 'inferno'
interpolation='bilinear'
NLevels = 50
margin = 12
fontsize = 16
linewidth = 2.0

from Helpers import *
import gpss
import h5py as hd

def BuildCylindricalProfile( I, Xs, Ys, Zs ):
    r = 1.5e-3                                              # Kanalradius
    l = 1e-3                                                # Kanallänge
    nU = Xs.size                                            # Anzahl d. Elemente auf dem Umfang
    nL = Ys.size                                            # Anzahl d. Elemente auf der Länge
    V = 2*np.pi*r**2*l

    x = np.linspace(    -r,    r, nU, endpoint=True)
    y = np.linspace( -.5*l, .5*l, nL, endpoint=True)

    Xc, Yc=np.meshgrid(x, y)
    Lc = I * np.exp( -(Yc**2/ ( r**2)) ) / ( np.pi * r**2 ) / V
    Zc = np.sqrt(r**2 - Xc**2)

    return Lc[:,0]

def BuildRectangularSource ( ds, a, b, PlotSource = False ):
    x = np.arange ( -a/2, a/2, ds )
    y = np.arange ( -b/2, b/2, ds )
    Xs, Ys = np.meshgrid ( x, y )
    Xs = Xs.ravel()
    Ys = Ys.ravel()
    Zs = np.zeros ( Xs.shape )

    if PlotSource:
        ####################
        # Plotting source configuration
        ####################

        fig, ax = plt.subplots(2, 2)
        ax[0][0].scatter ( Xs*1e3, Ys*1e3, c='r', marker='o')
        ax[0][0].set_xlabel ( r'$x$ / $mm$', fontsize=fontsize )
        ax[0][0].set_ylabel ( r'$y$ / $mm$', fontsize=fontsize )
        ax[0][0].grid(True)

        ax[0][1].scatter ( Xs*1e3, Zs*1e3, c='r', marker='o')
        ax[0][1].set_xlabel ( r'$x$ / $mm$', fontsize=fontsize )
        ax[0][1].set_ylabel ( r'$z$ / $mm$', fontsize=fontsize )
        ax[0][1].grid(True)


        ax[1][0].scatter ( Ys*1e3, Zs*1e3, c='r', marker='o')
        ax[1][0].set_xlabel ( r'$y$ / $mm$', fontsize=fontsize )
        ax[1][0].set_ylabel ( r'$z$ / $mm$', fontsize=fontsize )
        ax[1][0].grid(True)

        fig.tight_layout()
        plt.savefig( fig_prefix + 'PSS_Source_configuration.png', dpi=300 )

    return Xs, Ys, Zs

def BuildCircularSource ( ds, r, PlotSource = False ):

    x = np.arange ( -r, r, ds )
    y = np.arange ( -r, r, ds )
    Xa,Ya = np.meshgrid(x,y);
    Za = np.zeros( Xa.shape );
    idx = np.where( np.sqrt( Xa**2.0 + Ya**2.0) > r)
    Xa[idx] = np.nan;
    idx = np.where ( ~np.isnan(Xa) )
    Xs = np.array ( Xa [ idx ] )
    Ys = np.array ( Ya [ idx ] )
    Zs = np.array ( Za [ idx ] )

    if PlotSource:
        ####################
        # Plotting source configuration
        ####################

        fig, ax = plt.subplots(2, 2)
        ax[0][0].scatter ( Xs*1e3, Ys*1e3, c='r', marker='o')
        ax[0][0].set_xlabel ( r'$x$ / $mm$', fontsize=16 )
        ax[0][0].set_ylabel ( r'$y$ / $mm$', fontsize=16 )
        ax[0][0].grid(True)

        ax[0][1].scatter ( Xs*1e3, Zs*1e3, c='r', marker='o')
        ax[0][1].set_xlabel ( r'$x$ / $mm$', fontsize=16 )
        ax[0][1].set_ylabel ( r'$z$ / $mm$', fontsize=16 )
        ax[0][1].grid(True)


        ax[1][0].scatter ( Ys*1e3, Zs*1e3, c='r', marker='o')
        ax[1][0].set_xlabel ( r'$y$ / $mm$', fontsize=16 )
        ax[1][0].set_ylabel ( r'$z$ / $mm$', fontsize=16 )
        ax[1][0].grid(True)

        fig.tight_layout()
        plt.savefig( fig_prefix + 'PSS_Source_configuration.png', dpi=300 )
 #       plt.show()

    return Xs, Ys, Zs

def BuildCylindricalSource ( PlotSource=False ):
    """Build up cylindrical source configuration and plotting it to file"""
    r = 1.5e-3                                              # Kanalradius
    l = 1e-3                                                # Kanallänge
    nU = 50                                                 # Anzahl d. Elemente auf dem Umfang
    nL = 50                                                 # Anzahl d. Elemente auf der Länge
    dS = 1e-3                                               # Ortsauflösung
    u =  (2 * np.pi * r)                                    # Zylinderumfang
    dU = u / nU                                             # Bogenlänge eines Elements
    dL = l / nL
    alpha = 2 * np.arcsin ( np.sin ( dU / (2*r) ) )         # halber Mittelpunktswinkel

    Xs = np.zeros( nL * (nU) )
    Ys = np.zeros( nL * (nU) )
    Zs = np.zeros( nL * (nU) )

    for iL in range (0, nL):
        y = -0.5 * l + iL * dL
        for iU in range (0, nU):
            Xs[ iL * nU + iU ] = -r * ( np.cos ( (iU * alpha ) ) )
            Ys[ iL * nU + iU ] = y
            Zs[ iL * nU + iU ] = r * ( np.sin ( (iU * alpha ) ) )


    if PlotSource:
        ####################
        # Plotting source configuration
        ####################

        fig, ax = plt.subplots(2, 2)
        ax[0][0].scatter ( Xs*1e3, Ys*1e3, c='r', marker='o')
        ax[0][0].set_xlabel ( r'$x$ / $mm$', fontsize=fontsize )
        ax[0][0].set_ylabel ( r'$y$ / $mm$', fontsize=fontsize )
        ax[0][0].grid(True)
        ax[0][0].set_xlim ( -1.1*r*1e3, 1.1*r*1e3 )
        ax[0][0].set_ylim ( -1.1*l*1e3, 1.1*l*1e3 )

        ax[0][1].scatter ( Xs*1e3, Zs*1e3, c='r', marker='o')
        ax[0][1].set_xlabel ( r'$x$ / $mm$', fontsize=fontsize )
        ax[0][1].set_ylabel ( r'$z$ / $mm$', fontsize=fontsize )
        ax[0][1].grid(True)
        ax[0][1].set_xlim ( -1.1*r*1e3, 1.1*r*1e3 )
        ax[0][1].set_ylim ( -1.1*r*1e3, 1.1*r*1e3 )


        ax[1][0].scatter ( Ys*1e3, Zs*1e3, c='r', marker='o')
        ax[1][0].set_xlabel ( r'$y$ / $mm$', fontsize=fontsize )
        ax[1][0].set_ylabel ( r'$z$ / $mm$', fontsize=fontsize )
        ax[1][0].grid(True)
        ax[1][0].set_xlim ( -1.1*l*1e3, 1.1*l*1e3 )
        ax[1][0].set_ylim ( -1.1*r*1e3, 1.1*r*1e3 )

        fig.tight_layout()
        plt.savefig( fig_prefix + 'PSS_Source_configuration.png', dpi=300 )

    return Xs, Ys, Zs

def SaveData ( fileName, Xmesh, Ymesh, Zmesh, pdata ):
    with hd.File(fileName, 'w') as fd:
        fd.create_dataset('Xmesh', data=Xmesh,compression="gzip", compression_opts=9)
        fd.create_dataset('Ymesh', data=Ymesh,compression="gzip", compression_opts=9)
        fd.create_dataset('Zmesh', data=Zmesh,compression="gzip", compression_opts=9)
        fd.create_dataset('p', data=pdata,compression="gzip", compression_opts=9)



def PlotData ( data, Xmesh, Ymesh ):
    h, w = data.shape
    data_ratio = h / float(w)
    ### axes lengths in inche
    width_ax0 = 8.
    width_ax1 = 2.
    height_ax2 = 2.

    Xmesh *= 1e3
    Ymesh *= 1e3

    height_ax0 = width_ax0 * data_ratio

    ### define margins size in inche

    left_margin  = 1.00
    right_margin = 0.2
    bottom_margin = 0.5
    top_margin = 0.25
    inter_margin = 0.5

    ### calculate total figure size in inche

    fwidth = left_margin + right_margin + inter_margin + width_ax0 + width_ax1
    fheight = bottom_margin + top_margin + inter_margin + height_ax0 + height_ax2

    fig = plt.figure(figsize=(fwidth, fheight))
    fig.patch.set_facecolor('white')

    ### create axes

    ax1 = fig.add_axes([(left_margin ) / fwidth,
                        (bottom_margin + inter_margin + height_ax2) / fheight,
                         width_ax1 / fwidth, height_ax0 / fheight])

    ax0 = fig.add_axes([(left_margin + width_ax1 + inter_margin) / fwidth,
                        (bottom_margin + inter_margin + height_ax2) / fheight,
                        width_ax0 / fwidth, height_ax0 / fheight])


    ax2 = fig.add_axes([(left_margin + width_ax1 + inter_margin) / fwidth, bottom_margin / fheight,
                        width_ax0 / fwidth, height_ax2 / fheight])

    ### plotting the data

    bounds = [ Xmesh.min(),Xmesh.max(),Ymesh.min(),Ymesh.max()]
    ax0.imshow( data, cmap=colormap, extent = bounds, origin='lower')
    line = data[:,0]
    ax1.plot( line , Ymesh[:,0], 'k-', linewidth=1.0 )
    ax1.set_ylim ( bounds[2], bounds[3] )
    ax1.set_xlim ( .9 * np.amin (line) , 1.1 * np.amax (line) )
    ax1.invert_xaxis()

    line = data[int(h/2),:]
    ax2.plot( Xmesh[int(h/2),:], line , 'k-', linewidth=1.0 )
    ax2.set_xlim ( bounds[0], bounds[1] )
    ax2.set_ylim ( .9 * np.amin (line) , 1.1 * np.amax (line) )
    plt.savefig( fig_prefix + 'Arc_discharge_PSS_Sound_Pressurelevel.png', dpi=300 )
    plt.savefig( fig_prefix + 'Arc_discharge_PSS_Sound_Pressurelevel.pdf', dpi=300 )


def main():
    c = 343
    frequs = np.array ( [ 56e3 ] )#, 28.8e3, 57.6e3, 86.4e3 ] )

    ds = (c/np.amax(frequs))/10
    for iFrequ in frequs:

        print ( "Calculating soundfield @ %2.0f kHz." % ( iFrequ / 1e3) )

        Xs, Ys, Zs = BuildCircularSource ( ds, 17e-3, PlotSource=True)
        #Xs, Ys, Zs = BuildRectangularSource(ds, 80e-3, 210e-3, PlotSource=False )
        #Xs, Ys, Zs = BuildCylindricalSource()

        X = np.arange ( -20e-3, 20e-3, ds)
        Y = np.arange ( -20e-3, 20e-3, ds)
        Z = np.arange( 0, 100e-3, ds )
        I0 = 20

        #[Xmesh, Ymesh] = np.meshgrid(X, Y)
        #Zmesh = np.zeros ( Ymesh.shape )

        Xmesh, Ymesh, Zmesh = np.meshgrid(X, Y, Z, indexing='ij')
        #Xmesh, Zmesh = np.meshgrid(X, Z )
        #Ymesh = np.zeros ( Xmesh.shape )

        p = np.zeros ( shape=Xmesh.shape, dtype=np.cdouble )
        gpss.gpss_calculation3D ( Xs, Ys, Zs, 1.0, iFrequ, Xmesh, Ymesh, Zmesh, p )
        # Lp = 20*np.log10( p / 20e-6 )
        fileName = str(int(iFrequ)) + 'XYZ.mat'
        SaveData ( fileName, Xmesh, Ymesh, Zmesh, np.abs(p) )
        # PlotData( 20*np.log10( p/20e-6) ,Xmesh, Ymesh )


if __name__ == "__main__":
    main()
