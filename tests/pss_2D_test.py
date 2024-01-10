import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.colors as colors
import seaborn as sns

import pybamcmap

sns.set()
sns.set_style("white")
plt.style.use('seaborn-deep')

plt.rcParams['figure.figsize'] = (4, 1.68*4)

fig_prefix = ""

matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)
colormap = 'bam_jet'
interpolation='bilinear'
NLevels = 50
margin = 12
fontsize = 16
linewidth = 2.0

from Helpers import *
import gpss
import h5py as hd

# flood fill algorithm https://gist.github.com/JDWarner/1158a9515c7f1b1c21f1
def floodfill ( data, seed_coords, fill_value ):
    xsize, ysize = data.shape
    orig_value = data[ seed_coords[0], seed_coords[1] ]

    stack = set(((seed_coords[0], seed_coords[1]),))
    if fill_value == orig_value:
        raise ValueError("Filling region with same value "
                         "already present is unsupported. "
                         "Did you already fill this region?")

    while stack:
        x, y = stack.pop()

        if data [x, y] == orig_value:
            data[x, y] = fill_value
            if x > 0:
                stack.add((x - 1, y))
            if x < (xsize - 1):
                stack.add((x + 1, y))
            if y > 0:
                stack.add((x, y - 1))
            if y < (ysize - 1):
                stack.add((x, y + 1))


def circle(indices, xm, ym, r):

    x = r - 1
    y = 0
    dx = 1
    dy = 1
    err = dx - (r * 2)


    while ( x >= y):
        indices [ xm + x , ym + y] = 1.0
        indices [ xm + y , ym + x] = 1.0
        indices [ xm - y , ym + x] = 1.0
        indices [ xm - x , ym + y] = 1.0
        indices [ xm - x , ym - y] = 1.0
        indices [ xm - y , ym - x] = 1.0
        indices [ xm + y , ym - x] = 1.0
        indices [ xm + x , ym - y] = 1.0

        if err <= 0:
            y+=1
            err += dy
            dy += 2

        if err > 0:
            x-=1
            dy += 2
            err += dx - ( 2*r )



def BuildCircularSource ( ds, r, PlotSource = False ):
    """ Build a circular acoustic source 
    
    ds: stepsize
    r: radius

    PlotSource: Plots source arrangement for testing purposes

    """

    NX = NY = int((2*r)/ds)

    XXs = np.zeros((NX, NY))
    # draw circle using bresenham
    circle(XXs, NX//2, NY//2, int(r/ds) )
    # fill em up
    floodfill( XXs, [ NX//2, NY//2] , 1.0 )

    Xs,  Ys = XXs.nonzero()
    Zs = np.zeros(Xs.shape)

    Xs = (Xs - (np.amax(Xs)//2))*ds
    Ys = (Ys - (np.amax(Ys)//2))*ds

    if PlotSource:
        PlottingSourceConfiguration(Xs, Ys, Zs)

    return Xs, Ys, Zs

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
        PlottingSourceConfiguration(Xs, Ys, Zs)

    return Xs, Ys, Zs

def PlottingSourceConfiguration ( Xs, Ys, Zs, Ps = None, fileName = 'PSS_Source_configuration.png' ):
    """

    """
    if Ps == None:
        MarkerSize = 1
    else:
        MarkerSize = Ps 


    fig, ax = plt.subplots(2, 2)
    ax[0][0].scatter ( Xs*1e3, Ys*1e3, s=MarkerSize, c='r', marker='o')
    ax[0][0].set_xlabel ( r'$x$ / $mm$', fontsize=fontsize )
    ax[0][0].set_ylabel ( r'$y$ / $mm$', fontsize=fontsize )
    ax[0][0].grid(True)

    ax[0][1].scatter ( Xs*1e3, Zs*1e3, s=MarkerSize, c='r', marker='o')
    ax[0][1].set_xlabel ( r'$x$ / $mm$', fontsize=fontsize )
    ax[0][1].set_ylabel ( r'$z$ / $mm$', fontsize=fontsize )
    ax[0][1].grid(True)

    ax[1][0].scatter ( Ys*1e3, Zs*1e3, s=MarkerSize, c='r', marker='o')
    ax[1][0].set_xlabel ( r'$y$ / $mm$', fontsize=fontsize )
    ax[1][0].set_ylabel ( r'$z$ / $mm$', fontsize=fontsize )
    ax[1][0].grid(True)

    fig.tight_layout()
    plt.savefig( fileName, dpi=300 )


def SphereSegment( r, l ):
    """
    Calculating a sphere segment

    r: segment radius
    l: half segment length (over the surface of the sphere
    """
    alpha = l * 180.0 / (np.pi * r )
    rl = r * np.sind (alpha)
    hl = r - np.sqrt ( r**2 - rl**2 )
    sl = 2 * np.pi * rl * hl
    ll = 2 * np.pi * rl 
    return rl, hl, sl, ll

def BuildSphericallyCircularSource ( ds, Rs, R, PlotSource=True ):
    """
    Implements a spherical curved acoustic source pattern

    ds: numeric stepsize
    r:  radius of the source
    rc: radius of the curvature
    Plot: Plotting source distribution

    """

    alpha = np.arcsin(Rs/ R) * 180 / np.pi
    L = np.pi * R / 180.0 * alpha

    H = R - np.sqrt ( R**2 - Rs**2)
    S = 2 * np.pi * R * H
    l0 = ds*.5 
    r0, h0, s0, ll0 = SphereSegment( R, l0 )

    Xs = Ys = Zs = np.array([])

    nRings = int( np.ceil((L - l0)/ds ) )

    dRings = L / nRings 
    l = l0 
    sprev = s0 
    nSource = 0 

    for iRing in range (1, nRings ):
        l += dRings 
        r, h, s, ll = SphereSegment(R, l)
        sRing = s - sprev
        sprev = s

        nPieces = int ( np.ceil ( ll / ds ) )
        dBeta = 360 / nPieces 
        nSource += nPieces

        rh, hh, sh, llh = SphereSegment(R , l + dRings*.5 )

        for iPiece in range ( 0, nPieces ):
            beta = iPiece * dBeta
            Xs = np.append ( Xs, rh * np.sind(beta) )
            Ys = np.append ( Ys, rh * np.cosd(beta) )
            Zs = np.append ( Zs, hh)

    if PlotSource:
       PlottingSourceConfiguration(Xs, Ys, Zs)

    return Xs, Ys, Zs

def BuildSphericallyRectSource ( ds, a, b, Rs, R, PlotSource=True ):
    """
    Implements a spherical curved rectangular acoustic source pattern

    ds: numeric stepsize
    a:  width of the rect
    b:  height of the rect
    Rs: radius of the source
    r:  radius of the curvature
    Plot: Plotting source distribution

    """

    alpha = np.arcsin(Rs/ R) * 180 / np.pi
    L = np.pi * R / 180.0 * alpha

    H = R - np.sqrt ( R**2 - Rs**2)
    S = 2 * np.pi * R * H
    l0 = ds*.5 
    r0, h0, s0, ll0 = SphereSegment( R, l0 )

    Xs = Ys = Zs = np.array([])

    nRings = int( np.ceil((L - l0)/ds ) )

    dRings = L / nRings 
    l = l0 
    sprev = s0 
    nSource = 0 

    for iRing in range (1, nRings ):
        l += dRings 
        r, h, s, ll = SphereSegment(R, l)
        sRing = s - sprev
        sprev = s

        nPieces = int ( np.ceil ( ll / ds ) )
        dBeta = 360 / nPieces 
        nSource += nPieces

        rh, hh, sh, llh = SphereSegment(R , l + dRings*.5 )

        for iPiece in range ( 0, nPieces ):
            beta = iPiece * dBeta
            if ( ( np.abs(rh * np.cosd(beta)) < .5*a ) & ( np.abs( rh * np.sind(beta) ) < .5*b) ):  
                Xs = np.append ( Xs, rh * np.sind(beta) )
                Ys = np.append ( Ys, rh * np.cosd(beta) )
                Zs = np.append ( Zs, hh)

    if PlotSource:
       PlottingSourceConfiguration(Xs, Ys, Zs)

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
        PlottingSourceConfiguration(Xs, Ys, Zs)

    return Xs, Ys, Zs

def SaveData ( fileName, Xmesh, Ymesh, Zmesh, pdata ):
    with hd.File(fileName + '.mat', 'w') as fd:
        fd.create_dataset('Xmesh', data=Xmesh,compression="gzip", compression_opts=9)
        fd.create_dataset('Ymesh', data=Ymesh,compression="gzip", compression_opts=9)
        fd.create_dataset('Zmesh', data=Zmesh,compression="gzip", compression_opts=9)
        fd.create_dataset('p', data=pdata,compression="gzip", compression_opts=9)



def PlotData ( fileName, data, Xmesh, Ymesh ):
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
    # ax0.imshow ( data, cmap=colormap, extent = bounds, origin='lower')
    
    im1 = ax0.pcolor ( Xmesh, Ymesh, data, cmap=colormap ) 
    #fig.colorbar( im1, ax=ax.)

    line = data[:, w//2]

    ax1.plot( line , Ymesh[:,w//2], 'k-', linewidth=1.0 )
    ax1.set_ylim ( bounds[2], bounds[3] )
    ax1.set_xlim ( .9 * np.amin (line) , 1.1 * np.amax (line) )
    ax1.invert_xaxis()

    idx_max_x = int(np.argmax(line)) 
    line = data[ idx_max_x , : ]
    ax2.plot( Xmesh[ idx_max_x, : ], line , 'k-', linewidth=1.0 )
    ax2.set_xlim ( bounds[0], bounds[1] )
    ax2.set_ylim ( .9 * np.amin (line) , 1.1 * np.amax (line) )
    plt.savefig( fileName + '.png', dpi=300 )
    plt.savefig( fileName + '.pdf', dpi=300 )



def main():
    c = 343
    frequs = np.array ( [ 100e3 ] )

    ds = (c/np.amax(frequs))/20
    for iFrequ in frequs:

        print ( "Calculating soundfield @ %2.0f kHz." % ( iFrequ / 1e3) )

        Xs, Ys, Zs = BuildCircularSource(ds, 9.5e-3, PlotSource=False)

        # ds, a , b 
        # a in x direction
        # b in y direction 
        # Xs, Ys, Zs = BuildRectangularSource(ds, 20e-3, 8e-3, PlotSource=False )

        Xs, Ys, Zs = BuildSphericallyRectSource(ds, 20e-3, 8e-3, 12.5e-3, 23e-3 )

        # Generate spherically focussed circular point source distribution 
        # BuildSphericallyCircularSource( spatioal resolution, transducer radius, transducer curvature )
        #Xs, Ys, Zs = BuildSphericallyCircularSource(ds, 12.5e-3, 23e-3)
        
        # Generate cylindrical point source distribution 
        #Xs, Ys, Zs = BuildCylindricalSource()

        X = np.arange ( -20e-3, 20e-3, ds)
        Y = np.arange ( -20e-3, 20e-3, ds)
        Z = np.arange( 0, 100e-3, ds )
        I0 = 1 / Xs.size 

        #[Xmesh, Ymesh] = np.meshgrid(X, Y)
        #Zmesh = np.zeros ( Ymesh.shape )

        #Xmesh, Ymesh, Zmesh = np.meshgrid(X, Y, Z, indexing='ij')
        Xmesh, Zmesh = np.meshgrid(X , Z )
        Ymesh = np.zeros ( Xmesh.shape )

        p = np.zeros ( shape=Xmesh.shape, dtype=np.cdouble )
        # gpss.gpss_calculation3D ( Xs, Ys, Zs, I0, iFrequ, Xmesh, Ymesh, Zmesh, p )
        gpss.gpss_calculation2D ( Xs, Ys, Zs, I0, iFrequ, Xmesh, Ymesh, Zmesh, p )
        
        pp = np.sqrt ( p.imag*p.imag + p.real*p.real) 
        Lp = 20*np.log10( pp  / 20e-6 )
        fileName = str(int(iFrequ)) + '_SphereRect_XZ'
        SaveData ( fileName, Xmesh, Ymesh, Zmesh, pp )
        PlotData( fileName, pp , Xmesh, Zmesh )


if __name__ == "__main__":
    main()
