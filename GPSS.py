import numpy as np
import gpss

np.cosd = lambda x: np.cos(np.deg2rad(x))
np.acosd = lambda x: np.arccos(np.deg2rad(x))
np.sind = lambda x: np.sin(np.deg2rad(x))
np.asind = lambda x: np.arcsin(np.deg2rad(x))


class GPSS:

    @staticmethod
    def BuildRectangularSource(ds, a, b, Start=[0, 0]):
        x = np.arange(-a / 2, a / 2, ds) + Start[1]
        y = np.arange(-b / 2, b / 2, ds) + Start[0]
        xs, ys = np.meshgrid(x, y)
        xs = xs.ravel()
        ys = ys.ravel()
        zs = np.zeros(xs.shape)

        return xs, ys, zs

    @staticmethod
    def BuildCircularSource ( ds, r, Start=[0, 0] ):
        N = int(r/ds)
        ri = np.linspace ( 0, r, num = N)
        ni = ri * np.pi / ds
        ni = ni.astype(int)

        xs = np.array( [] )
        ys = np.array( [] )

        for ra, n in zip(ri, ni):
            t = np.linspace(0, 2*np.pi, n, endpoint=False)
            x = ra * np.cos(t) + Start[1]
            y = ra * np.sin(t) + Start[0]
            xs = np.append( xs, x )
            ys = np.append( ys, y )


        zs = np.zeros ( xs.size )
        return xs, ys, zs

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
    def BuildSphericallyRectSource(ds, a, b, Rs, R):
        """
        Implements a spherical curved rectangular acoustic source pattern

        ds: numeric stepsize
        a:  width of the rect
        b:  height of the rect
        Rs: radius of the source
        r:  radius of the curvature
        Plot: Plotting source distribution

        """

        alpha = np.arcsin(Rs / R) * 180 / np.pi
        L = np.pi * R / 180.0 * alpha

        H = R - np.sqrt(R ** 2 - Rs ** 2)
        S = 2 * np.pi * R * H
        l0 = ds * .5
        r0, h0, s0, ll0 = GPSS.sphere_segment(R, l0)

        Xs = Ys = Zs = np.array([])

        nRings = int(np.ceil((L - l0) / ds))

        dRings = L / nRings
        l = l0
        sprev = s0
        nSource = 0

        for iRing in range(1, nRings):
            l += dRings
            r, h, s, ll = GPSS.sphere_segment(R, l)
            sRing = s - sprev
            sprev = s

            nPieces = int(np.ceil(ll / ds))
            dBeta = 360 / nPieces
            nSource += nPieces

            rh, hh, sh, llh = GPSS.sphere_segment(R, l + dRings * .5)

            for iPiece in range(0, nPieces):
                beta = iPiece * dBeta
                if ((np.abs(rh * np.cosd(beta)) < .5 * a) & (np.abs(rh * np.sind(beta)) < .5 * b)):
                    Xs = np.append(Xs, rh * np.sind(beta))
                    Ys = np.append(Ys, rh * np.cosd(beta))
                    Zs = np.append(Zs, hh)

        return Xs, Ys, Zs

    @staticmethod
    def BuildArraySource(ds, ElementSize, GapSize, WaveLength, dAngleIncident=0, Offset=[0, 0], Geometry='Rect',
                         Type='Thermoacoustic', NElement=8, ApElement=None):
        """
        Implements an acoustic array source.

        ds: numeric stepsize
        ElementSize:
        GapSize:
        WaveLength:
        dAngleIncident: 
        Offset: Array [x,y ]
        Geometry: geometry of sources Rect
        Type: type of source: Thermoacoustic or moving mass transducer
        NElement: Number of elements
        ApElement: Element amplitude weighting
        """

        Xs = Ys = Zs = Ps = Is = np.array([])
        # Shift by half elements multiplied with element width (and a half element as gap size)
        # This is done for centering the array 
        dOffSetY = -.5 * ((NElement - 1 ) * ElementSize[1] + (NElement - 1) * GapSize ) 
        dOffSetX = 0  # -.5 * ElementSize[0]
        dElementPitch = ElementSize[1] + GapSize
        dPhase = np.zeros((NElement))

        print('Element size %.1f x %.1f mm and g=%.1f mm\n' % (ElementSize[0] * 1e3, ElementSize[1] * 1e3, GapSize * 1e3))

        for iElement in range(0, NElement):
            dY = Offset[1] + dOffSetY + (ElementSize[1] + GapSize) * iElement
            dX = Offset[0] + dOffSetX  # + ElementSize[1] * iElement
            dXs, dYs, dZs = GPSS.BuildRectangularSource(ds, ElementSize[0], ElementSize[1], Start=[dY, dX])

            ## Element phase shift
            dPhase[ iElement ] = ( -2.0 * np.pi * np.sin ( np.pi * dAngleIncident / 180 ) ) * iElement * dElementPitch  / WaveLength  
            print ( '%d / %d phasehifted by %1.3f / %04.0f ns' % ( iElement, NElement, dPhase[iElement], 1e9 * dPhase[iElement] / (2 * np.pi * 250e3) ) )
        
            # NHat = .5 * (NElement - 1)
            # dPhase [ iElement] = - 2.0 * np.pi * 250e3 * ( FocalPoint / c ) * ( np.sqrt( 1 + np.power ( ( NHat * dElementPitch ) / FocalPoint , 2.0 ) + 2 * np.sin( np.pi * dAngleIncident / 180 ) * NHat * dElementPitch / FocalPoint  ) - np.sqrt ( 1 + np.power( (iElement - NHat ) * dElementPitch / FocalPoint , 2.0 ) - 2 * np.sin( np.pi * dAngleIncident / 180 ) * (iElement - NHat ) * dElementPitch / FocalPoint ) )
            # print ( '%d / %d phasehifted by %2.1f DEG / %f ns' % ( iElement, NElement, np.pi * dPhase[iElement] / 180, 1e9 * dPhase[iElement] / ( 2 * np.pi * 250e3) ) )
            Ps = np.append ( Ps, np.ones( (dXs.size) ) * dPhase[iElement] )

            ## Append new source to the list of source points 
            Xs = np.append(Xs, dXs)
            Ys = np.append(Ys, dYs)
            Zs = np.append(Zs, dZs)
            if (ApElement is not None):
                Is = np.append(Is, np.ones((dXs.size)) * ApElement[iElement] / dXs.size)
            else:
                Is = np.append(Is, np.ones((dXs.size)) * 1.0 / ((iElement + 1) * dXs.size))

        return Xs, Ys, Zs, Is, Ps, dPhase

    @staticmethod
    def run_calc_2d(Xs, Ys, Zs, Ps, freq, Xmesh, Ymesh, Zmesh, I0):
        p = np.zeros(shape=Xmesh.shape, dtype=np.cdouble)
        # gpss_calculation2D (  Xs,  Ys,  Zs,  Phase, Q, f, Xmesh, Ymesh,  Zmesh, p ):
        gpss.gpss_calculation2D(Xs, Ys, Zs, Ps, I0, freq, Xmesh, Ymesh, Zmesh, p)
        pp = np.sqrt(p.imag * p.imag + p.real * p.real)
        return pp

    @staticmethod
    def run_calc_2d_complex(Xs, Ys, Zs, Ps, freq, Xmesh, Ymesh, Zmesh, I0=1):
        p = np.zeros(shape=Xmesh.shape, dtype=np.cdouble)
        gpss.gpss_calculation2D(Xs, Ys, Zs, Ps, I0, freq, Xmesh, Ymesh, Zmesh, p)
        return p

    @staticmethod
    def run_calc_3d(Xs, Ys, Zs, Ps, freq, Xmesh, Ymesh, Zmesh, I0):
        p = np.zeros(shape=Xmesh.shape, dtype=np.cdouble)
        # gpss_calculation2D (  Xs,  Ys,  Zs,  Phase, Q, f, Xmesh, Ymesh,  Zmesh, p ):
        gpss.gpss_calculation3D(Xs, Ys, Zs, Ps, I0, freq, Xmesh, Ymesh, Zmesh, p)
        pp = np.sqrt(p.imag * p.imag + p.real * p.real)
        return pp

    @staticmethod
    def run_calc_3d_complex(Xs, Ys, Zs, Ps, freq, Xmesh, Ymesh, Zmesh, I0=1):
        p = np.zeros(shape=Xmesh.shape, dtype=np.cdouble)
        gpss.gpss_calculation3D(Xs, Ys, Zs, Ps, I0, freq, Xmesh, Ymesh, Zmesh, p)
        return p
