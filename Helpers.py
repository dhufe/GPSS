import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from scipy.interpolate import griddata
#from matplotlib.mlab import griddata
import scipy.special as spc

np.cosd = lambda x : np.cos( np.deg2rad(x) )
np.acosd = lambda x : np.arccos( np.deg2rad(x) )
np.sind = lambda x : np.sin( np.deg2rad(x) )
np.asind = lambda x : np.arcsin( np.deg2rad(x) )

def NextPowerOfTwo( NSamples ):
    return int(2**np.ceil ( np.log2( NSamples ) ) )

def safe_log10(x, minval=0.0000000001):
    return np.log10(x.clip(min=minval))

def safe_log(x, minval=0.0000000001):
    return np.log(x.clip(min=minval))

def CalculateSPL ( pressure ):
    return ( 20 * np.log10 ( pressure / 20e-6 ))

def CalculateSpectre ( signal, wnd=True, cAbs=True ):
    """
    Calulating the spetre of a passed series using numpy.
    signal: series to transform
    wnd: using windowing
    cAbs: magnitude
    """

    N = signal.size
    NBins = NextPowerOfTwo( N ) 
    wndBlack= np.blackman( N )
    if wnd:
        Yw    = 2 * np.fft.fft( signal * wndBlack, NBins ) / NBins
    else:
        Yw    = 2 * np.abs(np.fft.fft( signal, NBins ) ) / NBins
        
    if cAbs:
        Yw = np.abs ( Yw )

    return Yw

def CalculateBandwidth ( signal, level=6.0, nOffset=0 ):
    inv = signal[::-1]
    max_idx = np.argmax ( signal )
    imax_idx = np.argmax ( inv )
    dB_left = np.argmin ( np.abs ( signal[0:max_idx] - ( np.amax ( signal ) - level ) )  )
    dB_righ = np.argmin ( np.abs ( inv[0:imax_idx] - ( np.amax ( signal ) - level ) ) + 1 )
    return (dB_left + nOffset, dB_righ + nOffset)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sp.butter(order, normal_cutoff, btype='low')
    return b, a

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sp.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_bandpass( lowcut, highcut, fs, order=5 ):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sp.butter(order, [low, high], btype='band')
    return b, a

def phi(x,mu, sigma = 1):
    return 0.5*( 1.0 + spc.erf((x-mu)/np.sqrt(2 * sigma* sigma ) ) )

def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)

def setAxisParams ( axis ):
    for tick in axis.get_xaxis().get_major_ticks():
        tick.set_pad( 12 )
        tick.label1 = tick._get_text1()
        
    for tick in axis.get_yaxis().get_major_ticks():
        tick.set_pad( 12 )
        tick.label1 = tick._get_text1()

    for ax in ['top','bottom','left','right']:
        axis.spines[ax].set_linewidth(2.0)

def hampel(x,k, t0=3):
    '''adapted from hampel function in R package pracma
    x= 1-d numpy array of numbers to be filtered
    k= number of items in window/2 (# forward and backward wanted to capture in median filter)
    t0= number of standard deviations to use; 3 is default
    '''
    n = len(x)
    y = x #y is the corrected series
    L = 1.4826
    for i in range((k + 1),(n - k)):
        if np.isnan(x[(i - k):(i + k+1)]).all():
            continue
        x0 = np.nanmedian(x[(i - k):(i + k+1)])
        S0 = L * np.nanmedian(np.abs(x[(i - k):(i + k+1)] - x0))
        if (np.abs(x[i] - x0) > t0 * S0):
            y[i] = x0
    return(y)


def set_rcparams(width=6.69291, fontsize=16, for_article=True, for_beamer=False):
    """
    Setting rcparams of matplotlib ready for publishing
    """

    height = width / 1.618

    if for_article or for_beamer:
        params = {
            #'backend': 'pdf',
            'axes.labelsize': fontsize,
            'font.size': fontsize,
            'figure.figsize': (width, height),
            'legend.fontsize': fontsize,
            'axes.titlesize': fontsize,
            'xtick.labelsize': fontsize,
            'ytick.labelsize': fontsize,
            'xtick.major.pad': fontsize,
            'xtick.major.pad': fontsize,
            'text.usetex': True,
            #'font.family': 'Times New Roman',
            'image.cmap' : 'RdBu',
            'image.interpolation'  : 'bilinear',
            'image.resample'  : False }
            #'font.serif': 'Times New Roman',
            #'font.sans-serif': 'Times New Roman'}
#            'ps.usedistiller': 'xpdf'}

    if for_beamer:
#       params['font.family'] = 'sans-serif'
        preamble = r'''\usepackage[cm]{sfmath}'''
        plt.rc('text.latex', preamble=preamble)

    if for_article or for_beamer:
        plt.rcParams.update(params)

def interpolate_griddata ( x, y, z, delta=10 ):
    rangeX = np.linspace ( np.amin ( x ), np.amax( x ), delta )
    rangeY = np.linspace ( np.amin ( y ), np.amax( y ), delta )

    Xdata, Ydata = np.meshgrid ( rangeX, rangeY )
    Zdata = griddata( (x, y), z, (rangeX[None,:], rangeY[:,None]), method='cubic')
    # Zdata = griddata( x, y, z, rangeX, rangeY, interp='linear' )

    return rangeX,rangeY,Zdata


class ASCReader:
    
    def __init__ (self, fileName):
        self.FileName = fileName
        self.Options  = self.parse_config ( )
        self.Time, self.Amplitude = self.parse_data();

    def getParameterByName (self, Name ):
        return self.Options[Name]
    
    def getTime ( self, us=False ):
        if  us:
            return self.Time*1e6
        else:
            return self.Time
    
    def getData ( self ):
        return self.Amplitude
    
    def getGain ( self, Linear=False ):
        if Linear:
            return 10 ** ( float( self.Options['Gain'] ) / -20)
        else:
            return float( self.Options['Gain'] )
        
    def parse_data ( self ):
        data = np.loadtxt( self.FileName, skiprows=72, encoding='ISO-8859-1' )
        dt = float ( self.Options ['TimeBase'] ) * 1e-6
        dtO = float ( self.Options ['TriggerDelay'] ) *1e-6
        amp = 100 * data / 2048 
        t = np.arange ( 0, (data.shape[0]*dt), dt) + dtO
        return t, amp
        
    def parse_config(self):
        COMMENT_CHAR = '#'
        OPTION_CHAR =  ':'

        options = {}
        f = open(self.FileName, encoding='latin9')
        for line in f:
            # First, remove comments:
            if COMMENT_CHAR in line:
                # split on comment char, keep only the part before
                line, comment = line.split(COMMENT_CHAR, 1)
            # Second, find lines with an option=value:
            if OPTION_CHAR in line:           
                # split on option char:
                option, value = line.split(OPTION_CHAR, 1)
                # strip spaces:
                strOption = "".join(option)
                if '[' in strOption:
                    option, ba = strOption.split ( '[', 1 )

                option = option.strip()
                # print ( option )
                if option == 'Bemerkungen':
                    param_list = value.split(',')
                    for opt in param_list:
                        #print ( opt.strip() )
                        if '=' in opt:
                            param, val = "".join(opt).split('=', 1)
                            options[param.strip() ] = val.strip()
                            #print ('%s = %s' % (param, val ) )

                value = value.strip()
                # store in dictionary:
                options[option] = value
        f.close()
        return options