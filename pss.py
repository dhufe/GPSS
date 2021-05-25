import numpy as np 
import gpss 

np.cosd = lambda x : np.cos( np.deg2rad(x) )
np.acosd = lambda x : np.arccos( np.deg2rad(x) )
np.sind = lambda x : np.sin( np.deg2rad(x) )
np.asind = lambda x : np.arcsin( np.deg2rad(x) )

class GPSS:

    @staticmethod
    def BuildRectangularSource ( ds, a, b ):
        x = np.arange ( -a/2, a/2, ds )
        y = np.arange ( -b/2, b/2, ds )
        Xs, Ys = np.meshgrid ( x, y )
        Xs = Xs.ravel()
        Ys = Ys.ravel()
        Zs = np.zeros ( Xs.shape )

        return Xs, Ys, Zs

    @staticmethod 
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

    @staticmethod
    def BuildSphericallyCircularSource ( ds, Rs, R ):
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
        r0, h0, s0, ll0 = GPSS.SphereSegment( R, l0 )

        Xs = Ys = Zs = np.array([])

        nRings = int( np.ceil((L - l0)/ds ) )
        dRings = L / nRings 
        l = l0 
        sprev = s0 
        nSource = 0 

        for iRing in range (1, nRings ):
            l += dRings 
            r, h, s, ll = GPSS.SphereSegment(R, l)
            sRing = s - sprev
            sprev = s

            nPieces = int ( np.ceil ( ll / ds ) )
            dBeta = 360 / nPieces 
            nSource += nPieces

            rh, hh, sh, llh = GPSS.SphereSegment(R , l + dRings*.5 )

            for iPiece in range ( 0, nPieces ):
                beta = iPiece * dBeta
                Xs = np.append ( Xs, rh * np.sind(beta) )
                Ys = np.append ( Ys, rh * np.cosd(beta) )
                Zs = np.append ( Zs, hh)

        return Xs, Ys, Zs

    @staticmethod 
    def BuildSphericallyRectSource ( ds, a, b, Rs, R ):
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
        r0, h0, s0, ll0 = GPSS.SphereSegment( R, l0 )

        Xs = Ys = Zs = np.array([])

        nRings = int( np.ceil((L - l0)/ds ) )

        dRings = L / nRings 
        l = l0 
        sprev = s0 
        nSource = 0 

        for iRing in range (1, nRings ):
            l += dRings 
            r, h, s, ll = GPSS.SphereSegment(R, l)
            sRing = s - sprev
            sprev = s

            nPieces = int ( np.ceil ( ll / ds ) )
            dBeta = 360 / nPieces 
            nSource += nPieces

            rh, hh, sh, llh = GPSS.SphereSegment(R , l + dRings*.5 )

            for iPiece in range ( 0, nPieces ):
                beta = iPiece * dBeta
                if ( ( np.abs(rh * np.cosd(beta)) < .5*a ) & ( np.abs( rh * np.sind(beta) ) < .5*b) ):  
                    Xs = np.append ( Xs, rh * np.sind(beta) )
                    Ys = np.append ( Ys, rh * np.cosd(beta) )
                    Zs = np.append ( Zs, hh)


        return Xs, Ys, Zs


    @staticmethod 
    def RunCalculation2D ( Xs, Ys, Zs, freq, Xmesh, Ymesh, Zmesh, I0 = 1 ):
        p = np.zeros ( shape=Xmesh.shape, dtype=np.cdouble )
        gpss.gpss_calculation2D ( Xs, Ys, Zs, I0, freq, Xmesh, Ymesh, Zmesh, p )    
        pp = np.sqrt ( p.imag*p.imag + p.real*p.real)
        return pp 

    @staticmethod 
    def RunCalculation2DComplex ( Xs, Ys, Zs, freq, Xmesh, Ymesh, Zmesh, I0 = 1 ):
        p = np.zeros ( shape=Xmesh.shape, dtype=np.cdouble )
        gpss.gpss_calculation2D ( Xs, Ys, Zs, I0, freq, Xmesh, Ymesh, Zmesh, p )     
        return p 
 
