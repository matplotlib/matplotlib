"""

Numerical python functions written for compatability with matlab(TM)
commands with the same names.  

  Matlab(TM) compatible functions:

    * cohere - Coherence (normalized cross spectral density)

    * conv     - convolution
    
    * corrcoef - The matrix of correlation coefficients

    * csd - Cross spectral density uing Welch's average periodogram

    * detrend -- Remove the mean or best fit line from an array

    * find - Return the indices where some condition is true
    
    * linspace -- Linear spaced array from min to max

    * hist -- Histogram
    
    * polyfit - least squares best polynomial fit of x to y

    * polyval - evaluate a vector for a vector of polynomial coeffs

    * prctile - find the percentiles of a sequence
    
    * prepca - Principal Component's Analysis
    
    * psd - Power spectral density uing Welch's average periodogram

    * rk4 - A 4th order runge kutta integrator for 1D or ND systems
 
    * vander - the Vandermonde matrix

    * trapz - trapeziodal integration
    
  Functions that don't exist in matlab(TM), but are useful anyway:

    * cohere_pairs - Coherence over all pairs.  This is not a matlab
      function, but we compute coherence a lot in my lab, and we
      compute it for alot of pairs.  This function is optimized to do
      this efficiently by caching the direct FFTs.

Credits:

  Unless otherwise noted, these functions were written by
  Author: John D. Hunter <jdhunter@ace.bsd.uchicago.edu>

  Some others are from the Numeric documentation, or imported from
  MLab or other Numeric packages

"""

from __future__ import division
import sys, random
from matplotlib import verbose
import numerix
import numerix.mlab 
from numerix import linear_algebra
import numerix as nx
import nxutils

from numerix import array, asarray, arange, divide, exp, arctan2, \
     multiply, transpose, ravel, repeat, resize, reshape, floor, ceil,\
     absolute, matrixmultiply, power, take, where, Float, Int, asum,\
     dot, convolve, pi, Complex, ones, zeros, diagonal, Matrix, nonzero, \
     log, searchsorted, concatenate, sort, ArrayType, clip, size, indices,\
     conjugate, typecode, iscontiguous


from numerix.mlab import hanning, cov, diff, svd, rand, std
from numerix.fft import fft, inverse_fft

from cbook import iterable, is_string_like


def mean(x, dim=None):
   if len(x)==0: return None
   elif dim is None:
      return numerix.mlab.mean(x)
   else: return numerix.mlab.mean(x, dim)
   

def linspace(xmin, xmax, N):
   if N==1: return array([xmax])
   dx = (xmax-xmin)/(N-1)
   return xmin + dx*arange(N)

def logspace(xmin,xmax,N):
    return exp(linspace(log(xmin), log(xmax),Nh))

def _norm(x):
    "return sqrt(x dot x)"
    return numerix.mlab.sqrt(dot(x,x))

def window_hanning(x):
    "return x times the hanning window of len(x)"
    return hanning(len(x))*x

def window_none(x):
    "No window function; simply return x"
    return x

def conv(x, y, mode=2):
    'convolve x with y'
    return convolve(x,y,mode)

def detrend(x, key=None):
    if key is None or key=='constant':
        return detrend_mean(x)
    elif key=='linear':
        return detrend_linear(x)

def detrend_mean(x):
    "Return x minus the mean(x)"
    return x - mean(x)

def detrend_none(x):
    "Return x: no detrending"
    return x

def detrend_linear(x):
    "Return x minus best fit line; 'linear' detrending "

    # I'm going to regress x on xx=range(len(x)) and return x -
    # (b*xx+a).  Now that I have polyfit working, I could convert the
    # code here, but if it ain't broke, don't fix it!
    xx = arange(float(len(x)))
    X = transpose(array([xx]+[x]))
    C = cov(X)
    b = C[0,1]/C[0,0]
    a = mean(x) - b*mean(xx)
    return x-(b*xx+a)

def psd(x, NFFT=256, Fs=2, detrend=detrend_none,
        window=window_hanning, noverlap=0):
    """
    The power spectral density by Welches average periodogram method.
    The vector x is divided into NFFT length segments.  Each segment
    is detrended by function detrend and windowed by function window.
    noperlap gives the length of the overlap between segments.  The
    absolute(fft(segment))**2 of each segment are averaged to compute Pxx,
    with a scaling to correct for power loss due to windowing.  Fs is
    the sampling frequency.

    -- NFFT must be a power of 2
    -- detrend is a functions, unlike in matlab where it is a vector.
    -- window can be a function or a vector of length NFFT. To create window
       vectors see numpy.blackman, numpy.hamming, numpy.bartlett,
       scipy.signal, scipy.signal.get_window etc.
    -- if length x < NFFT, it will be zero padded to NFFT
    

    Returns the tuple Pxx, freqs

    Refs:
      Bendat & Piersol -- Random Data: Analysis and Measurement
        Procedures, John Wiley & Sons (1986)

    """

    if NFFT % 2:
        raise ValueError, 'NFFT must be a power of 2'

    # zero pad x up to NFFT if it is shorter than NFFT
    if len(x)<NFFT:
        n = len(x)
        x = resize(x, (NFFT,))
        x[n:] = 0
    

    # for real x, ignore the negative frequencies
    if typecode(x)==Complex: numFreqs = NFFT
    else: numFreqs = NFFT//2+1
        
    if iterable(window):
       assert(len(window) == NFFT)
       windowVals = window
    else:
       windowVals = window(ones((NFFT,),typecode(x)))
    step = NFFT-noverlap
    ind = range(0,len(x)-NFFT+1,step)
    n = len(ind)
    Pxx = zeros((numFreqs,n), Float)
    # do the ffts of the slices
    for i in range(n):
        thisX = x[ind[i]:ind[i]+NFFT]
        thisX = windowVals*detrend(thisX)
        fx = absolute(fft(thisX))**2
        Pxx[:,i] = divide(fx[:numFreqs], norm(windowVals)**2)

    # Scale the spectrum by the norm of the window to compensate for
    # windowing loss; see Bendat & Piersol Sec 11.5.2
    if n>1:
       Pxx = mean(Pxx,1)

    freqs = Fs/NFFT*arange(numFreqs)
    Pxx.shape = len(freqs),

    return Pxx, freqs

def csd(x, y, NFFT=256, Fs=2, detrend=detrend_none,
        window=window_hanning, noverlap=0):
    """
    The cross spectral density Pxy by Welches average periodogram
    method.  The vectors x and y are divided into NFFT length
    segments.  Each segment is detrended by function detrend and
    windowed by function window.  noverlap gives the length of the
    overlap between segments.  The product of the direct FFTs of x and
    y are averaged over each segment to compute Pxy, with a scaling to
    correct for power loss due to windowing.  Fs is the sampling
    frequency.

    NFFT must be a power of 2

    window can be a function or a vector of length NFFT. To create 
    window vectors see numpy.blackman, numpy.hamming, numpy.bartlett,
    scipy.signal, scipy.signal.get_window etc.

    Returns the tuple Pxy, freqs

    

    Refs:
      Bendat & Piersol -- Random Data: Analysis and Measurement
        Procedures, John Wiley & Sons (1986)

    """

    if NFFT % 2:
        raise ValueError, 'NFFT must be a power of 2'

    # zero pad x and y up to NFFT if they are shorter than NFFT
    if len(x)<NFFT:
        n = len(x)
        x = resize(x, (NFFT,))
        x[n:] = 0
    if len(y)<NFFT:
        n = len(y)
        y = resize(y, (NFFT,))
        y[n:] = 0

    # for real x, ignore the negative frequencies
    if typecode(x)==Complex: numFreqs = NFFT
    else: numFreqs = NFFT//2+1
        
    if iterable(window):
       assert(len(window) == NFFT)
       windowVals = window
    else:
       windowVals = window(ones((NFFT,),typecode(x)))
    step = NFFT-noverlap
    ind = range(0,len(x)-NFFT+1,step)
    n = len(ind)
    Pxy = zeros((numFreqs,n), Complex)

    # do the ffts of the slices
    for i in range(n):
        thisX = x[ind[i]:ind[i]+NFFT]
        thisX = windowVals*detrend(thisX)
        thisY = y[ind[i]:ind[i]+NFFT]
        thisY = windowVals*detrend(thisY)
        fx = fft(thisX)
        fy = fft(thisY)
        Pxy[:,i] = conjugate(fx[:numFreqs])*fy[:numFreqs]



    # Scale the spectrum by the norm of the window to compensate for
    # windowing loss; see Bendat & Piersol Sec 11.5.2
    if n>1: Pxy = mean(Pxy,1)
    Pxy = divide(Pxy, norm(windowVals)**2)
    freqs = Fs/NFFT*arange(numFreqs)
    Pxy.shape = len(freqs),
    return Pxy, freqs

def cohere(x, y, NFFT=256, Fs=2, detrend=detrend_none,
           window=window_hanning, noverlap=0):
    """
    cohere the coherence between x and y.  Coherence is the normalized
    cross spectral density

    Cxy = |Pxy|^2/(Pxx*Pyy)

    The return value is (Cxy, f), where f are the frequencies of the
    coherence vector.  See the docs for psd and csd for information
    about the function arguments NFFT, detrend, window, noverlap, as
    well as the methods used to compute Pxy, Pxx and Pyy.

    Returns the tuple Cxy, freqs

    """
    
    if len(x)<2*NFFT:
       raise RuntimeError('Coherence is calculated by averaging over NFFT length segments.  Your signal is too short for your choice of NFFT')
    Pxx, f = psd(x, NFFT, Fs, detrend, window, noverlap)
    Pyy, f = psd(y, NFFT, Fs, detrend, window, noverlap)
    Pxy, f = csd(x, y, NFFT, Fs, detrend, window, noverlap)

    Cxy = divide(absolute(Pxy)**2, Pxx*Pyy)
    Cxy.shape = len(f),
    return Cxy, f

def corrcoef(*args):
    """
    corrcoef(X) where X is a matrix returns a matrix of correlation
    coefficients for each numrows observations and numcols variables.
    
    corrcoef(x,y) where x and y are vectors returns the matrix or
    correlation coefficients for x and y.

    Numeric arrays can be real or complex

    The correlation matrix is defined from the covariance matrix C as

    r(i,j) = C[i,j] / sqrt(C[i,i]*C[j,j])
    """


    if len(args)==2:
        X = transpose(array([args[0]]+[args[1]]))
    elif len(args)==1:
        X = args[0]
    else:
        raise RuntimeError, 'Only expecting 1 or 2 arguments'

    
    C = cov(X)

    if len(args)==2:
       d = resize(diagonal(C), (2,1))
       denom = numerix.mlab.sqrt(matrixmultiply(d,transpose(d)))
    else:
       dc = diagonal(C)
       N = len(dc)       
       shape = N,N
       vi = resize(dc, shape)
       denom = numerix.mlab.sqrt(vi*transpose(vi)) # element wise multiplication
       

    r = divide(C,denom)
    try: return r.real
    except AttributeError: return r




def polyfit(x,y,N):
    """

    Do a best fit polynomial of order N of y to x.  Return value is a
    vector of polynomial coefficients [pk ... p1 p0].  Eg, for N=2

      p2*x0^2 +  p1*x0 + p0 = y1
      p2*x1^2 +  p1*x1 + p0 = y1
      p2*x2^2 +  p1*x2 + p0 = y2
      .....
      p2*xk^2 +  p1*xk + p0 = yk
      
      
    Method: if X is a the Vandermonde Matrix computed from x (see
    http://mathworld.wolfram.com/VandermondeMatrix.html), then the
    polynomial least squares solution is given by the 'p' in

      X*p = y

    where X is a len(x) x N+1 matrix, p is a N+1 length vector, and y
    is a len(x) x 1 vector

    This equation can be solved as

      p = (XT*X)^-1 * XT * y

    where XT is the transpose of X and -1 denotes the inverse.

    For more info, see
    http://mathworld.wolfram.com/LeastSquaresFittingPolynomial.html,
    but note that the k's and n's in the superscripts and subscripts
    on that page.  The linear algebra is correct, however.

    See also polyval

    """

    x = asarray(x)+0.
    y = asarray(y)+0.
    y = reshape(y, (len(y),1))
    X = Matrix(vander(x, N+1))
    Xt = Matrix(transpose(X))
    c = array(linear_algebra.inverse(Xt*X)*Xt*y)  # convert back to array
    c.shape = (N+1,)
    return c
    

    

def polyval(p,x):
    """
    y = polyval(p,x)

    p is a vector of polynomial coeffients and y is the polynomial
    evaluated at x.

    Example code to remove a polynomial (quadratic) trend from y:

      p = polyfit(x, y, 2)
      trend = polyval(p, x)
      resid = y - trend

    See also polyfit
    
    """
    x = asarray(x)+0.
    p = reshape(p, (len(p),1))
    X = vander(x,len(p))
    y =  matrixmultiply(X,p)
    return reshape(y, x.shape)


def vander(x,N=None):
    """
    X = vander(x,N=None)

    The Vandermonde matrix of vector x.  The i-th column of X is the
    the i-th power of x.  N is the maximum power to compute; if N is
    None it defaults to len(x).

    """
    if N is None: N=len(x)
    X = ones( (len(x),N), typecode(x))
    for i in range(N-1):
        X[:,i] = x**(N-i-1)
    return X



def donothing_callback(*args):
    pass

def cohere_pairs( X, ij, NFFT=256, Fs=2, detrend=detrend_none,
                  window=window_hanning, noverlap=0,
                  preferSpeedOverMemory=True,
                  progressCallback=donothing_callback,
                  returnPxx=False):

    """
    Cxy, Phase, freqs = cohere_pairs( X, ij, ...)
    
    Compute the coherence for all pairs in ij.  X is a
    numSamples,numCols Numeric array.  ij is a list of tuples (i,j).
    Each tuple is a pair of indexes into the columns of X for which
    you want to compute coherence.  For example, if X has 64 columns,
    and you want to compute all nonredundant pairs, define ij as

      ij = []
      for i in range(64):
          for j in range(i+1,64):
              ij.append( (i,j) )

    The other function arguments, except for 'preferSpeedOverMemory'
    (see below), are explained in the help string of 'psd'.

    Return value is a tuple (Cxy, Phase, freqs).

      Cxy -- a dictionary of (i,j) tuples -> coherence vector for that
        pair.  Ie, Cxy[(i,j) = cohere(X[:,i], X[:,j]).  Number of
        dictionary keys is len(ij)
      
      Phase -- a dictionary of phases of the cross spectral density at
        each frequency for each pair.  keys are (i,j).

      freqs -- a vector of frequencies, equal in length to either the
        coherence or phase vectors for any i,j key.  Eg, to make a coherence
        Bode plot:

          subplot(211)
          plot( freqs, Cxy[(12,19)])
          subplot(212)
          plot( freqs, Phase[(12,19)])
      
    For a large number of pairs, cohere_pairs can be much more
    efficient than just calling cohere for each pair, because it
    caches most of the intensive computations.  If N is the number of
    pairs, this function is O(N) for most of the heavy lifting,
    whereas calling cohere for each pair is O(N^2).  However, because
    of the caching, it is also more memory intensive, making 2
    additional complex arrays with approximately the same number of
    elements as X.

    The parameter 'preferSpeedOverMemory', if false, limits the
    caching by only making one, rather than two, complex cache arrays.
    This is useful if memory becomes critical.  Even when
    preferSpeedOverMemory is false, cohere_pairs will still give
    significant performace gains over calling cohere for each pair,
    and will use subtantially less memory than if
    preferSpeedOverMemory is true.  In my tests with a 43000,64 array
    over all nonredundant pairs, preferSpeedOverMemory=1 delivered a
    33% performace boost on a 1.7GHZ Athlon with 512MB RAM compared
    with preferSpeedOverMemory=0.  But both solutions were more than
    10x faster than naievly crunching all possible pairs through
    cohere.

    See test/cohere_pairs_test.py in the src tree for an example
    script that shows that this cohere_pairs and cohere give the same
    results for a given pair.

    """
    numRows, numCols = X.shape

    # zero pad if X is too short
    if numRows < NFFT:
        tmp = X
        X = zeros( (NFFT, numCols), typecode(X))
        X[:numRows,:] = tmp
        del tmp

    numRows, numCols = X.shape
    # get all the columns of X that we are interested in by checking
    # the ij tuples
    seen = {}
    for i,j in ij:
        seen[i]=1; seen[j] = 1
    allColumns = seen.keys()
    Ncols = len(allColumns)
    del seen
    
    # for real X, ignore the negative frequencies
    if typecode(X)==Complex: numFreqs = NFFT
    else: numFreqs = NFFT//2+1

    # cache the FFT of every windowed, detrended NFFT length segement
    # of every channel.  If preferSpeedOverMemory, cache the conjugate
    # as well
    if iterable(window):
       assert(len(window) == NFFT)
       windowVals = window
    else:
       windowVals = window(ones((NFFT,), typecode(X)))
    ind = range(0, numRows-NFFT+1, NFFT-noverlap)
    numSlices = len(ind)
    FFTSlices = {}
    FFTConjSlices = {}
    Pxx = {}
    slices = range(numSlices)
    normVal = norm(windowVals)**2
    for iCol in allColumns:
        progressCallback(i/Ncols, 'Cacheing FFTs')
        Slices = zeros( (numSlices,numFreqs), Complex)
        for iSlice in slices:                    
            thisSlice = X[ind[iSlice]:ind[iSlice]+NFFT, iCol]
            thisSlice = windowVals*detrend(thisSlice)
            Slices[iSlice,:] = fft(thisSlice)[:numFreqs]
            
        FFTSlices[iCol] = Slices
        if preferSpeedOverMemory:
            FFTConjSlices[iCol] = conjugate(Slices)
        Pxx[iCol] = divide(mean(absolute(Slices)**2), normVal)
    del Slices, ind, windowVals    

    # compute the coherences and phases for all pairs using the
    # cached FFTs
    Cxy = {}
    Phase = {}
    count = 0
    N = len(ij)
    for i,j in ij:
        count +=1
        if count%10==0:
            progressCallback(count/N, 'Computing coherences')

        if preferSpeedOverMemory:
            Pxy = FFTSlices[i] * FFTConjSlices[j]
        else:
            Pxy = FFTSlices[i] * conjugate(FFTSlices[j])
        if numSlices>1: Pxy = mean(Pxy)
        Pxy = divide(Pxy, normVal)
        Cxy[(i,j)] = divide(absolute(Pxy)**2, Pxx[i]*Pxx[j])
        Phase[(i,j)] =  arctan2(Pxy.imag, Pxy.real)

    freqs = Fs/NFFT*arange(numFreqs)
    if returnPxx:
       return Cxy, Phase, freqs, Pxx
    else:
       return Cxy, Phase, freqs



def entropy(y, bins):
   """
   Return the entropy of the data in y

   \sum p_i log2(p_i) where p_i is the probability of observing y in
   the ith bin of bins.  bins can be a number of bins or a range of
   bins; see hist

   Compare S with analytic calculation for a Gaussian
   x = mu + sigma*randn(200000)
   Sanalytic = 0.5  * ( 1.0 + log(2*pi*sigma**2.0) ) 

   """

   
   n,bins = hist(y, bins)
   n = n.astype(Float)

   n = take(n, nonzero(n))         # get the positive

   p = divide(n, len(y))

   delta = bins[1]-bins[0]
   S = -1.0*asum(p*log(p)) + log(delta)
   #S = -1.0*asum(p*log(p))
   return S

def hist(y, bins=10, normed=0):
    """
    Return the histogram of y with bins equally sized bins.  If bins
    is an array, use the bins.  Return value is
    (n,x) where n is the count for each bin in x

    If normed is False, return the counts in the first element of the
    return tuple.  If normed is True, return the probability density
    n/(len(y)*dbin)

    If y has rank>1, it will be raveled
    Credits: the Numeric 22 documentation

    

    """
    y = asarray(y)
    if len(y.shape)>1: y = ravel(y)

    if not iterable(bins):       
        ymin, ymax = min(y), max(y)
        if ymin==ymax:
            ymin -= 0.5
            ymax += 0.5

        if bins==1: bins=ymax
        dy = (ymax-ymin)/bins 
        bins = ymin + dy*arange(bins)


    n = searchsorted(sort(y), bins)
    n = diff(concatenate([n, [len(y)]]))
    if normed:
       db = bins[1]-bins[0]
       return 1/(len(y)*db)*n, bins
    else:
       return n, bins


def normpdf(x, *args):
   "Return the normal pdf evaluated at x; args provides mu, sigma"
   mu, sigma = args
   return 1/(numerix.mlab.sqrt(2*pi)*sigma)*exp(-0.5 * (1/sigma*(x - mu))**2)
                 

def levypdf(x, gamma, alpha):
   "Returm the levy pdf evaluated at x for params gamma, alpha"

   N = len(x)

   if N%2 != 0:
      raise ValueError, 'x must be an event length array; try\n' + \
            'x = linspace(minx, maxx, N), where N is even'
   

   dx = x[1]-x[0]


   f = 1/(N*dx)*arange(-N/2, N/2, Float)

   ind = concatenate([arange(N/2, N, Int),
                      arange(N/2,Int)])
   df = f[1]-f[0]
   cfl = exp(-gamma*absolute(2*pi*f)**alpha)

   px = fft(take(cfl,ind)*df).astype(Float)
   return take(px, ind)




      

def find(condition):
   "Return the indices where condition is true"
   return nonzero(condition)



def trapz(x, y):
   if len(x)!=len(y):
      raise ValueError, 'x and y must have the same length'
   if len(x)<2:
      raise ValueError, 'x and y must have > 1 element'
   return asum(0.5*diff(x)*(y[1:]+y[:-1]))
   
   

def longest_contiguous_ones(x):
    """
    return the indicies of the longest stretch of contiguous ones in x,
    assuming x is a vector of zeros and ones.
    """
    if len(x)==0: return array([])

    ind = find(x==0)
    if len(ind)==0:  return arange(len(x))
    if len(ind)==len(x): return array([])

    y = zeros( (len(x)+2,),  typecode(x))
    y[1:-1] = x
    dif = diff(y)
    up = find(dif ==  1);
    dn = find(dif == -1);
    ind = find( dn-up == max(dn - up))
    ind = arange(take(up, ind), take(dn, ind))

    return ind


def longest_ones(x):
    """
    return the indicies of the longest stretch of contiguous ones in x,
    assuming x is a vector of zeros and ones.

    If there are two equally long stretches, pick the first
    """
    x = asarray(x)
    if len(x)==0: return array([])

    #print 'x', x
    ind = find(x==0)
    if len(ind)==0:  return arange(len(x))
    if len(ind)==len(x): return array([])

    y = zeros( (len(x)+2,), Int)
    y[1:-1] = x
    d = diff(y)
    #print 'd', d
    up = find(d ==  1);
    dn = find(d == -1);

    #print 'dn', dn, 'up', up, 
    ind = find( dn-up == max(dn - up))
    # pick the first
    if iterable(ind): ind = ind[0]
    ind = arange(up[ind], dn[ind])

    return ind

def prepca(P, frac=0):
    """
    Compute the principal components of P.  P is a numVars x
    numObservations numeric array.  frac is the minimum fraction of
    variance that a component must contain to be included

    Return value are
    Pcomponents : a num components x num observations numeric array
    Trans       : the weights matrix, ie, Pcomponents = Trans*P
    fracVar     : the fraction of the variance accounted for by each
                  component returned
    """
    U,s,v = svd(P)
    varEach = s**2/P.shape[1]
    totVar = asum(varEach)
    fracVar = divide(varEach,totVar)
    ind = int(asum(fracVar>=frac))

    # select the components that are greater
    Trans = transpose(U[:,:ind])
    # The transformed data
    Pcomponents = matrixmultiply(Trans,P)
    return Pcomponents, Trans, fracVar[:ind]

def prctile(x, p = (0.0, 25.0, 50.0, 75.0, 100.0)):
    """
    Return the percentiles of x.  p can either be a sequence of
    percentil values or a scalar.  If p is a sequence the i-th element
    of the return sequence is the p(i)-th percentile of x
    """
    x = sort(ravel(x))
    Nx = len(x)

    if not iterable(p):
        return x[int(p*Nx/100.0)]

    p = multiply(array(p), Nx/100.0)
    ind = p.astype(Int)
    ind = where(ind>=Nx, Nx-1, ind)        
    return take(x, ind)


def center_matrix(M, dim=0):
    """
    Return the matrix M with each row having zero mean and unit std

    if dim=1, center columns rather than rows
    """
    # todo: implement this w/o loop.  Allow optional arg to specify
    # dimension to remove the mean from
    if dim==1: M = transpose(M)
    M = array(M, Float)
    if len(M.shape)==1 or M.shape[0]==1 or M.shape[1]==1:
       M = M-mean(M)
       sigma = std(M)
       if sigma>0:
          M = divide(M, sigma)
       if dim==1: M=transpose(M)
       return M
     
    for i in range(M.shape[0]):
        M[i] -= mean(M[i])
        sigma = std(M[i])
        if sigma>0:
           M[i] = divide(M[i], sigma)
    if dim==1: M=transpose(M)
    return M

def meshgrid(x,y):
    """
    For vectors x, y with lengths Nx=len(x) and Ny=len(y), return X, Y
    where X and Y are (Ny, Nx) shaped arrays with the elements of x
    and y repeated to fill the matrix

    EG,

      [X, Y] = meshgrid([1,2,3], [4,5,6,7])

       X =
         1   2   3
         1   2   3
         1   2   3
         1   2   3


       Y =
         4   4   4
         5   5   5
         6   6   6
         7   7   7
  """
  
    x = array(x)
    y = array(y)
    numRows, numCols = len(y), len(x)  # yes, reversed
    x.shape = 1, numCols
    X = repeat(x, numRows)

    y.shape = numRows,1
    Y = repeat(y, numCols, 1)
    return X, Y



def rk4(derivs, y0, t):
    """
    Integrate 1D or ND system of ODEs from initial state y0 at sample
    times t.  derivs returns the derivative of the system and has the
    signature

     dy = derivs(yi, ti)

    Example 1 :

        ## 2D system
        # Numeric solution
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)

    Example 2:

        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)

        y0 = 1
        yout = rk4(derivs, y0, t)


    """
   
    try: Ny = len(y0)
    except TypeError:
        yout = zeros( (len(t),), Float)
    else:
        yout = zeros( (len(t), Ny), Float)
        
        
    yout[0] = y0
    i = 0
    
    for i in arange(len(t)-1):

        thist = t[i]
        dt = t[i+1] - thist
        dt2 = dt/2.0
        y0 = yout[i]

        k1 = asarray(derivs(y0, thist))
        k2 = asarray(derivs(y0 + dt2*k1, thist+dt2))
        k3 = asarray(derivs(y0 + dt2*k2, thist+dt2))
        k4 = asarray(derivs(y0 + dt*k3, thist+dt))
        yout[i+1] = y0 + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)
    return yout




def specgram(x, NFFT=256, Fs=2, detrend=detrend_none,
             window=window_hanning, noverlap=128):
    """
    Compute a spectrogram of data in x.  Data are split into NFFT
    length segements and the PSD of each section is computed.  The
    windowing function window is applied to each segment, and the
    amount of overlap of each segment is specified with noverlap.

    window can be a function or a vector of length NFFT. To create
    window vectors see numpy.blackman, numpy.hamming, numpy.bartlett,
    scipy.signal, scipy.signal.get_window etc.


    See pdf for more info.

    If x is real (i.e. non-Complex) only the positive spectrum is
    given.  If x is Complex then the complete spectrum is given.

    The returned times are the midpoints of the intervals over which
    the ffts are calculated
    """
    x = asarray(x)
    assert(NFFT>noverlap)
    if log(NFFT)/log(2) != int(log(NFFT)/log(2)):
       raise ValueError, 'NFFT must be a power of 2'

    # zero pad x up to NFFT if it is shorter than NFFT
    if len(x)<NFFT:
        n = len(x)
        x = resize(x, (NFFT,))
        x[n:] = 0
    

    # for real x, ignore the negative frequencies
    if typecode(x)==Complex: numFreqs=NFFT
    else: numFreqs = NFFT//2+1

    if iterable(window):
       assert(len(window) == NFFT)
       windowVals = window
    else:
       windowVals = window(ones((NFFT,),typecode(x)))
    step = NFFT-noverlap
    ind = arange(0,len(x)-NFFT+1,step)
    n = len(ind)
    Pxx = zeros((numFreqs,n), Float)
    # do the ffts of the slices

    for i in range(n):
        thisX = x[ind[i]:ind[i]+NFFT]
        thisX = windowVals*detrend(thisX)
        fx = absolute(fft(thisX))**2
        # Scale the spectrum by the norm of the window to compensate for
        # windowing loss; see Bendat & Piersol Sec 11.5.2
        Pxx[:,i] = divide(fx[:numFreqs], norm(windowVals)**2)
    t = 1/Fs*(ind+NFFT/2)
    freqs = Fs/NFFT*arange(numFreqs)

    if typecode(x) == Complex:
       freqs = concatenate((freqs[NFFT/2:]-Fs,freqs[:NFFT/2]))
       Pxx   = concatenate((Pxx[NFFT/2:,:],Pxx[:NFFT/2,:]),0)

    return Pxx, freqs, t

def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                     mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate gaussan distribution for equal shape X, Y

    http://mathworld.wolfram.com/BivariateNormalDistribution.html
    """
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    return 1.0/(2*pi*sigmax*sigmay*numerix.mlab.sqrt(1-rho**2)) * exp( -z/(2*(1-rho**2)))




def get_xyz_where(Z, Cond):
    """
    Z and Cond are MxN matrices.  Z are data and Cond is a boolean
    matrix where some condition is satisfied.  Return value is x,y,z
    where x and y are the indices into Z and z are the values of Z at
    those indices.  x,y,z are 1D arrays
    """
    
    M,N = Z.shape
    z = ravel(Z)
    ind = nonzero( ravel(Cond) )

    x = arange(M); x.shape = M,1
    X = repeat(x, N, 1)
    x = ravel(X)

    y = arange(N); y.shape = 1,N
    Y = repeat(y, M)
    y = ravel(Y)

    x = take(x, ind)
    y = take(y, ind)
    z = take(z, ind)
    return x,y,z

def get_sparse_matrix(M,N,frac=0.1):
    'return a MxN sparse matrix with frac elements randomly filled'
    data = zeros((M,N))*0.
    for i in range(int(M*N*frac)):
        x = random.randint(0,M-1)
        y = random.randint(0,N-1)
        data[x,y] = rand()
    return data

def dist(x,y):
    'return the distance between two points'
    d = x-y
    return numerix.mlab.sqrt(dot(d,d))

def dist_point_to_segment(p, s0, s1):
    """
    get the distance of a point to a segment.

    p, s0, s1 are xy sequences

    This algorithm from
    http://softsurfer.com/Archive/algorithm_0102/algorithm_0102.htm#Distance%20to%20Ray%20or%20Segment
    """
    p = asarray(p, Float)
    s0 = asarray(s0, Float)
    s1 = asarray(s1, Float)    
    v = s1 - s0
    w = p - s0

    c1 = dot(w,v);
    if ( c1 <= 0 ):
        return dist(p, s0);

    c2 = dot(v,v)
    if ( c2 <= c1 ):
        return dist(p, s1);

    b = c1 / c2
    pb = s0 + b * v;
    return dist(p, pb)

def segments_intersect(s1, s2):
    """
    Return True if s1 and s2 intersect.
    s1 and s2 are defines as

    s1: (x1, y1), (x2, y2)
    s2: (x3, y3), (x4, y4)

    """
    (x1, y1), (x2, y2) = s1
    (x3, y3), (x4, y4) = s2

    den = ((y4-y3) * (x2-x1)) - ((x4-x3)*(y2-y1))

    n1 = ((x4-x3) * (y1-y3)) - ((y4-y3)*(x1-x3))
    n2 = ((x2-x1) * (y1-y3)) - ((y2-y1)*(x1-x3))

    if den == 0:
        # lines parallel
        return False

    u1 = n1/den
    u2 = n2/den

    return 0.0 <= u1 <= 1.0 and 0.0 <= u2 <= 1.0


def fftsurr(x, detrend=detrend_none, window=window_none):
    """
    Compute an FFT phase randomized surrogate of x
    """
    if iterable(window):
       x=window*detrend(x)
    else:
       x = window(detrend(x))
    z = fft(x)
    a = 2.*pi*1j
    phase = a*rand(len(x))
    z = z*exp(phase)
    return inverse_fft(z).real


def liaupunov(x, fprime):
   """
   x is a very long trajectory from a map, and fprime returns the
   derivative of x.  Return lambda = 1/n\sum ln|fprime(x_i)|.  See Sec
   10.5 Strogatz (1994)"Nonlinear Dynamics and Chaos".   
   """
   return mean(log(fprime(x)))

class FIFOBuffer:
    """
    A FIFO queue to hold incoming x, y data in a rotating buffer using
    numerix arrrays under the hood.  It is assumed that you will call
    asarrays much less frequently than you add data to the queue --
    otherwise another data structure will be faster

    This can be used to support plots where data is added from a real
    time feed and the plot object wants grab data from the buffer and
    plot it to screen less freqeuently than the incoming

    If you set the dataLim attr to a matplotlib BBox (eg ax.dataLim),
    the dataLim will be updated as new data come in

    TODI: add a grow method that will extend nmax
    """
    def __init__(self, nmax):
        'buffer up to nmax points'
        self._xa = nx.zeros((nmax,), typecode=nx.Float)
        self._ya = nx.zeros((nmax,), typecode=nx.Float)        
        self._xs = nx.zeros((nmax,), typecode=nx.Float)
        self._ys = nx.zeros((nmax,), typecode=nx.Float)        
        self._ind = 0
        self._nmax = nmax
        self.dataLim = None
        self.callbackd = {}
        
    def register(self, func, N):
        'call func everytime N events are passed; func signature is func(fifo)'
        self.callbackd.setdefault(N, []).append(func)
        
    def add(self, x, y):
        'add scalar x and y to the queue'
        if self.dataLim is not None:
           xys = ((x,y),) 
           self.dataLim.update(xys, -1) #-1 means use the default ignore setting
        ind = self._ind % self._nmax
        #print 'adding to fifo:', ind, x, y
        self._xs[ind] = x
        self._ys[ind] = y

        for N,funcs in self.callbackd.items():
           if (self._ind%N)==0:
              for func in funcs:
                 func(self)
                 
        self._ind += 1

    def last(self):
        'get the last x, y or None, None if no data set'
        if self._ind==0: return None, None
        ind = (self._ind-1) % self._nmax
        return self._xs[ind], self._ys[ind]

    def asarrays(self):
        """
        return x and y as arrays; their length will be the len of data
        added or nmax
        """
        if self._ind<self._nmax:
            return self._xs[:self._ind], self._ys[:self._ind]
        ind = self._ind % self._nmax

        self._xa[:self._nmax-ind] = self._xs[ind:]
        self._xa[self._nmax-ind:] = self._xs[:ind]
        self._ya[:self._nmax-ind] = self._ys[ind:]
        self._ya[self._nmax-ind:] = self._ys[:ind]

        return self._xa, self._ya

    def update_datalim_to_current(self):
        'update the datalim in the current data in the fifo'
        if self.dataLim is None:
            raise ValueError('You must first set the dataLim attr')
        x, y = self.asarrays()
        self.dataLim.update_numerix(x, y, True)

def movavg(x,n):
    'compute the len(n) moving average of x'
    n = int(n)
    N = len(x)
    assert(N>n)
    y = zeros(N-(n-1),Float)
    for i in range(n):
       y += x[i:N-(n-1)+i]
    y /= float(n)
    return y

def save(fname, X, fmt='%.18e',delimiter=' '):
    """
    Save the data in X to file fname using fmt string to convert the
    data to strings

    fname can be a filename or a file handle.  If the filename ends in .gz,
    the file is automatically saved in compressed gzip format.  The load()
    command understands gzipped files transparently.

    Example usage:

    save('test.out', X)         # X is an array
    save('test1.out', (x,y,z))  # x,y,z equal sized 1D arrays
    save('test2.out', x)        # x is 1D
    save('test3.out', x, fmt='%1.4e')  # use exponential notation

    delimiter is used to separate the fields, eg delimiter ',' for
    comma-separated values
    """

    if is_string_like(fname):
        if fname.endswith('.gz'):
            import gzip
            fh = gzip.open(fname,'wb')
        else:
            fh = file(fname,'w')
    elif hasattr(fname, 'seek'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')


    X = asarray(X)
    origShape = None
    if len(X.shape)==1:
        origShape = X.shape
        X.shape = len(X), 1
    for row in X:
        fh.write(delimiter.join([fmt%val for val in row]) + '\n')

    if origShape is not None:
        X.shape = origShape




def load(fname,comments='#',delimiter=None, converters=None,skiprows=0,
         usecols=None, unpack=False):
    """
    Load ASCII data from fname into an array and return the array.

    The data must be regular, same number of values in every row

    fname can be a filename or a file handle.  Support for gzipped files is
    automatic, if the filename ends in .gz

    matfile data is not currently supported, but see
    Nigel Wade's matfile ftp://ion.le.ac.uk/matfile/matfile.tar.gz

    Example usage:

      X = load('test.dat')  # data in two columns
      t = X[:,0]
      y = X[:,1]

    Alternatively, you can do the same with "unpack"; see below

      X = load('test.dat')    # a matrix of data
      x = load('test.dat')    # a single column of data

    comments - the character used to indicate the start of a comment
    in the file

    delimiter is a string-like character used to seperate values in the
    file. If delimiter is unspecified or none, any whitespace string is
    a separator.

    converters, if not None, is a dictionary mapping column number to
    a function that will convert that column to a float.  Eg, if
    column 0 is a date string: converters={0:datestr2num}

    skiprows is the number of rows from the top to skip

    usecols, if not None, is a sequence of integer column indexes to
    extract where 0 is the first column, eg usecols=(1,4,5) to extract
    just the 2nd, 5th and 6th columns

    unpack, if True, will transpose the matrix allowing you to unpack
    into named arguments on the left hand side

        t,y = load('test.dat', unpack=True) # for  two column data
        x,y,z = load('somefile.dat', usecols=(3,5,7), unpack=True)

    See examples/load_demo.py which exeercises many of these options.
    """

    if converters is None: converters = {}
    if is_string_like(fname):
        if fname.endswith('.gz'):
            import gzip
            fh = gzip.open(fname)
        else:
            fh = file(fname)
    elif hasattr(fname, 'seek'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')
    X = []

    for i,line in enumerate(fh):
        if i<skiprows: continue
        line = line[:line.find(comments)].strip()
        if not len(line): continue
        if usecols is not None:
            vals = line.split(delimiter)
            row = [converters.get(i,float)(vals[i]) for i in usecols]
        else:
            row = [converters.get(i,float)(val) for i,val in enumerate(line.split(delimiter))]
        thisLen = len(row)
        X.append(row)

    X = array(X)
    r,c = X.shape
    if r==1 or c==1:
        X.shape = max([r,c]),
    if unpack: return transpose(X)
    else:  return X

def slopes(x,y):
    """
    SLOPES calculate the slope y'(x) Given data vectors X and Y SLOPES
    calculates Y'(X), i.e the slope of a curve Y(X). The slope is
    estimated using the slope obtained from that of a parabola through
    any three consecutive points.

    This method should be superior to that described in the appendix
    of A CONSISTENTLY WELL BEHAVED METHOD OF INTERPOLATION by Russel
    W. Stineman (Creative Computing July 1980) in at least one aspect:

    Circles for interpolation demand a known aspect ratio between x-
    and y-values.  For many functions, however, the abscissa are given
    in different dimensions, so an aspect ratio is completely
    arbitrary.
    
    The parabola method gives very similar results to the circle
    method for most regular cases but behaves much better in special
    cases

    Norbert Nemec, Institute of Theoretical Physics, University or
    Regensburg, April 2006 Norbert.Nemec at physik.uni-regensburg.de

    (inspired by a original implementation by Halldor Bjornsson,
    Icelandic Meteorological Office, March 2006 halldor at vedur.is)
    """
    # Cast key variables as float.
    x=nx.asarray(x, nx.Float)
    y=nx.asarray(y, nx.Float)

    yp=nx.zeros(y.shape, nx.Float)

    dx=x[1:] - x[:-1]
    dy=y[1:] - y[:-1]
    dydx = dy/dx
    yp[1:-1] = (dydx[:-1] * dx[1:] + dydx[1:] * dx[:-1])/(dx[1:] + dx[:-1])
    yp[0] = 2.0 * dy[0]/dx[0] - yp[1]
    yp[-1] = 2.0 * dy[-1]/dx[-1] - yp[-2]
    return yp


def stineman_interp(xi,x,y,yp=None):
    """
    STINEMAN_INTERP Well behaved data interpolation.  Given data
    vectors X and Y, the slope vector YP and a new abscissa vector XI
    the function stineman_interp(xi,x,y,yp) uses Stineman
    interpolation to calculate a vector YI corresponding to XI.

    Here's an example that generates a coarse sine curve, then
    interpolates over a finer abscissa:

      x = linspace(0,2*pi,20);  y = sin(x); yp = cos(x)
      xi = linspace(0,2*pi,40);
      yi = stineman_interp(xi,x,y,yp);
      plot(x,y,'o',xi,yi)

    The interpolation method is described in the article A
    CONSISTENTLY WELL BEHAVED METHOD OF INTERPOLATION by Russell
    W. Stineman. The article appeared in the July 1980 issue of
    Creative computing with a note from the editor stating that while
    they were

      not an academic journal but once in a while something serious
      and original comes in adding that this was
      "apparently a real solution" to a well known problem.
    
    For yp=None, the routine automatically determines the slopes using
    the "slopes" routine.

    X is assumed to be sorted in increasing order

    For values xi[j] < x[0] or xi[j] > x[-1], the routine tries a
    extrapolation.  The relevance of the data obtained from this, of
    course, questionable...

    original implementation by Halldor Bjornsson, Icelandic
    Meteorolocial Office, March 2006 halldor at vedur.is

    completely reworked and optimized for Python by Norbert Nemec,
    Institute of Theoretical Physics, University or Regensburg, April
    2006 Norbert.Nemec at physik.uni-regensburg.de

    """

    # Cast key variables as float.
    x=nx.asarray(x, nx.Float)
    y=nx.asarray(y, nx.Float)
    assert x.shape == y.shape
    N=len(y)

    if yp is None:
        yp = slopes(x,y)
    else:
        yp=nx.asarray(yp, nx.Float)

    xi=nx.asarray(xi, nx.Float)
    yi=nx.zeros(xi.shape, nx.Float)

    # calculate linear slopes
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    s = dy/dx  #note length of s is N-1 so last element is #N-2

    # find the segment each xi is in
    # this line actually is the key to the efficiency of this implementation
    idx = nx.searchsorted(x[1:-1], xi)

    # now we have generally: x[idx[j]] <= xi[j] <= x[idx[j]+1]
    # except at the boundaries, where it may be that xi[j] < x[0] or xi[j] > x[-1]

    # the y-values that would come out from a linear interpolation:
    sidx = nx.take(s, idx)
    xidx = nx.take(x, idx)
    yidx = nx.take(y, idx)    
    xidxp1 = nx.take(x, idx+1)
    yo = yidx + sidx * (xi - xidx)

    # the difference that comes when using the slopes given in yp
    dy1 = (nx.take(yp, idx)- sidx) * (xi - xidx)       # using the yp slope of the left point
    dy2 = (nx.take(yp, idx+1)-sidx) * (xi - xidxp1) # using the yp slope of the right point

    dy1dy2 = dy1*dy2
    # The following is optimized for Python. The solution actually
    # does more calculations than necessary but exploiting the power
    # of numpy, this is far more efficient than coding a loop by hand
    # in Python
    yi = yo + dy1dy2 * nx.choose(nx.array(nx.sign(dy1dy2), nx.Int32)+1, 
                                 ((2*xi-xidx-xidxp1)/((dy1-dy2)*(xidxp1-xidx)),
                                  0.0,
                                  1/(dy1+dy2),))
        
    return yi

def _inside_poly_deprecated(points, verts):
    """
    # use nxutils.points_inside_poly instead
    points is a sequence of x,y points
    verts is a sequence of x,y vertices of a poygon

    return value is a sequence on indices into points for the points
    that are inside the polygon
    """
    xys = nx.asarray(points)
    Nxy = xys.shape[0]
    Nv = len(verts)

    def angle(x1, y1, x2, y2):
        twopi = 2*nx.pi
        theta1 = nx.arctan2(y1, x1)
        theta2 = nx.arctan2(y2, x2)
        dtheta = theta2-theta1
        d = dtheta%twopi
        d = nx.where(nx.less(d, 0), twopi + d, d)
        return nx.where(nx.greater(d,nx.pi), d-twopi, d)

    angles = nx.zeros((Nxy,), nx.Float)
    x1 = nx.zeros((Nxy,), nx.Float)
    y1 = nx.zeros((Nxy,), nx.Float)
    x2 = nx.zeros((Nxy,), nx.Float)
    y2 = nx.zeros((Nxy,), nx.Float)    
    x = xys[:,0]
    y = xys[:,1]
    for i in range(Nv):
        thisx, thisy = verts[i]
        x1 = thisx - x
        y1 = thisy - y
        thisx, thisy = verts[(i+1)%Nv]
        x2 = thisx - x
        y2 = thisy - y

        a = angle(x1, y1, x2, y2)
        angles += a
    return nx.nonzero(nx.greater_equal(nx.absolute(angles), nx.pi))

def inside_poly(points, verts):
    """"
    points is a sequence of x,y points
    verts is a sequence of x,y vertices of a poygon

    return value is a sequence on indices into points for the points
    that are inside the polygon
    """
    return nx.nonzero(nxutils.points_inside_poly(points, verts))

### the following code was written and submitted by Fernando Perez
### from the ipython numutils package under a BSD license
# begin fperez functions
"""
A set of convenient utilities for numerical work.

Most of this module requires Numerical Python or is meant to be used with it.
See http://www.pfdubois.com/numpy for details.

Copyright (c) 2001-2004, Fernando Perez. <Fernando.Perez@colorado.edu>
All rights reserved.

This license was generated from the BSD license template as found in:
http://www.opensource.org/licenses/bsd-license.php

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of the IPython project nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import operator
import math


#*****************************************************************************
# Globals

#****************************************************************************
# function definitions        
exp_safe_MIN = math.log(2.2250738585072014e-308)
exp_safe_MAX = 1.7976931348623157e+308

def exp_safe(x):
    """Compute exponentials which safely underflow to zero.

    Slow but convenient to use. Note that NumArray will introduce proper
    floating point exception handling with access to the underlying
    hardware."""

    if type(x) is ArrayType:
        return exp(clip(x,exp_safe_MIN,exp_safe_MAX))
    else:
        return math.exp(x)

def amap(fn,*args):
    """amap(function, sequence[, sequence, ...]) -> array.

    Works like map(), but it returns an array.  This is just a convenient
    shorthand for Numeric.array(map(...))"""
    return array(map(fn,*args))


def zeros_like(a):
    """Return an array of zeros of the shape and typecode of a."""

    return zeros(a.shape,typecode(a))

def sum_flat(a):
    """Return the sum of all the elements of a, flattened out.

    It uses a.flat, and if a is not contiguous, a call to ravel(a) is made."""

    if iscontiguous(a):
        return asum(a.flat)
    else:
        return asum(ravel(a))

def mean_flat(a):
    """Return the mean of all the elements of a, flattened out."""

    return sum_flat(a)/float(size(a))

def rms_flat(a):
    """Return the root mean square of all the elements of a, flattened out."""

    return numerix.mlab.sqrt(sum_flat(absolute(a)**2)/float(size(a)))

def l1norm(a):
    """Return the l1 norm of a, flattened out.

    Implemented as a separate function (not a call to norm() for speed)."""

    return sum_flat(absolute(a))

def l2norm(a):
    """Return the l2 norm of a, flattened out.

    Implemented as a separate function (not a call to norm() for speed)."""

    return numerix.mlab.sqrt(sum_flat(absolute(a)**2))

def norm(a,p=2):
    """norm(a,p=2) -> l-p norm of a.flat

    Return the l-p norm of a, considered as a flat array.  This is NOT a true
    matrix norm, since arrays of arbitrary rank are always flattened.

    p can be a number or the string 'Infinity' to get the L-infinity norm."""
    
    if p=='Infinity':
        return max(absolute(a).flat)
    else:
        return (sum_flat(absolute(a)**p))**(1.0/p)    
    
def frange(xini,xfin=None,delta=None,**kw):
    """frange([start,] stop[, step, keywords]) -> array of floats

    Return a Numeric array() containing a progression of floats. Similar to
    arange(), but defaults to a closed interval.

    frange(x0, x1) returns [x0, x0+1, x0+2, ..., x1]; start defaults to 0, and
    the endpoint *is included*. This behavior is different from that of
    range() and arange(). This is deliberate, since frange will probably be
    more useful for generating lists of points for function evaluation, and
    endpoints are often desired in this use. The usual behavior of range() can
    be obtained by setting the keyword 'closed=0', in this case frange()
    basically becomes arange().

    When step is given, it specifies the increment (or decrement). All
    arguments can be floating point numbers.

    frange(x0,x1,d) returns [x0,x0+d,x0+2d,...,xfin] where xfin<=x1.

    frange can also be called with the keyword 'npts'. This sets the number of
    points the list should contain (and overrides the value 'step' might have
    been given). arange() doesn't offer this option.

    Examples:
    >>> frange(3)
    array([ 0.,  1.,  2.,  3.])
    >>> frange(3,closed=0)
    array([ 0.,  1.,  2.])
    >>> frange(1,6,2)
    array([1, 3, 5])
    >>> frange(1,6.5,npts=5)
    array([ 1.   ,  2.375,  3.75 ,  5.125,  6.5  ])
    """

    #defaults
    kw.setdefault('closed',1)
    endpoint = kw['closed'] != 0
        
    # funny logic to allow the *first* argument to be optional (like range())
    # This was modified with a simpler version from a similar frange() found
    # at http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/66472
    if xfin == None:
        xfin = xini + 0.0
        xini = 0.0
        
    if delta == None:
        delta = 1.0

    # compute # of points, spacing and return final list
    try:
        npts=kw['npts']
        delta=(xfin-xini)/float(npts-endpoint)
    except KeyError:
        # round() gets npts right even with the vagaries of floating point.
        npts=int(round((xfin-xini)/delta+endpoint))

    return arange(npts)*delta+xini
# end frange()

def diagonal_matrix(diag):
    """Return square diagonal matrix whose non-zero elements are given by the
    input array."""

    return diag*identity(len(diag))

def identity(n,rank=2,typecode='l'):
    """identity(n,r) returns the identity matrix of shape (n,n,...,n) (rank r).

    For ranks higher than 2, this object is simply a multi-index Kronecker
    delta:
                        /  1  if i0=i1=...=iR,
    id[i0,i1,...,iR] = -|
                        \  0  otherwise.

    Optionally a typecode may be given (it defaults to 'l').

    Since rank defaults to 2, this function behaves in the default case (when
    only n is given) like the Numeric identity function."""
    
    iden = zeros((n,)*rank,typecode=typecode)
    for i in range(n):
        idx = (i,)*rank
        iden[idx] = 1
    return iden

def base_repr (number, base = 2, padding = 0):
    """Return the representation of a number in any given base."""
    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if number < base: \
       return (padding - 1) * chars [0] + chars [int (number)]
    max_exponent = int (math.log (number)/math.log (base))
    max_power = long (base) ** max_exponent
    lead_digit = int (number/max_power)
    return chars [lead_digit] + \
           base_repr (number - max_power * lead_digit, base, \
                      max (padding - 1, max_exponent))

def binary_repr(number, max_length = 1025):
    """Return the binary representation of the input number as a string.

    This is more efficient than using base_repr with base 2.

    Increase the value of max_length for very large numbers. Note that on
    32-bit machines, 2**1023 is the largest integer power of 2 which can be
    converted to a Python float."""
    

    #assert number < 2L << max_length
    shifts = map (operator.rshift, max_length * [number], \
                  range (max_length - 1, -1, -1))
    digits = map (operator.mod, shifts, max_length * [2])
    if not digits.count (1): return 0
    digits = digits [digits.index (1):]
    return ''.join (map (repr, digits)).replace('L','')

def log2(x,ln2 = math.log(2.0)):
    """Return the log(x) in base 2.
    
    This is a _slow_ function but which is guaranteed to return the correct
    integer value if the input is an ineger exact power of 2."""

    try:
        bin_n = binary_repr(x)[1:]
    except (AssertionError,TypeError):
        return math.log(x)/ln2
    else:
        if '1' in bin_n:
            return math.log(x)/ln2
        else:
            return len(bin_n)

def ispower2(n):
    """Returns the log base 2 of n if n is a power of 2, zero otherwise.

    Note the potential ambiguity if n==1: 2**0==1, interpret accordingly."""

    bin_n = binary_repr(n)[1:]
    if '1' in bin_n:
        return 0
    else:
        return len(bin_n)

def fromfunction_kw(function, dimensions, **kwargs):
    """Drop-in replacement for fromfunction() from Numerical Python.
 
    Allows passing keyword arguments to the desired function.

    Call it as (keywords are optional):
    fromfunction_kw(MyFunction, dimensions, keywords)

    The function MyFunction() is responsible for handling the dictionary of
    keywords it will recieve."""

    return function(tuple(indices(dimensions)),**kwargs)

### end fperez numutils code

### begin mlab2 functions
# From MLab2: http://pdilib.sourceforge.net/MLab2.py
readme = \
       """
MLab2.py, release 1

Created on February 2003 by Thomas Wendler as part of the Emotionis Project.
This script is supposed to implement Matlab functions that were left out in
numerix.mlab.py (part of Numeric Python).
For further information on the Emotionis Project or on this script, please
contact their authors:
Rodrigo Benenson, rodrigob at elo dot utfsm dot cl
Thomas Wendler,   thomasw at elo dot utfsm dot cl
Look at: http://pdilib.sf.net for new releases.
"""

_eps_approx = 1e-13

def fix(x):

    """
    Rounds towards zero.
    x_rounded = fix(x) rounds the elements of x to the nearest integers
    towards zero.
    For negative numbers is equivalent to ceil and for positive to floor.
    """
    
    dim = numerix.shape(x)
    if numerix.mlab.rank(x)==2:
        y = reshape(x,(1,dim[0]*dim[1]))[0]
        y = y.tolist()
    elif numerix.mlab.rank(x)==1:
        y = x
    else:
        y = [x]
    for i in range(len(y)):
	if y[i]>0:
		y[i] = floor(y[i])
	else:
		y[i] = ceil(y[i])
    if numerix.mlab.rank(x)==2:
        x = reshape(y,dim)
    elif numerix.mlab.rank(x)==0:
        x = y[0]
    return x

def rem(x,y):
    """
    Remainder after division.
    rem(x,y) is equivalent to x - y.*fix(x./y) in case y is not zero.
    By convention, rem(x,0) returns None.
    We keep the convention by Matlab:
    "The input x and y must be real arrays of the same size, or real scalars."
    """
    
    x,y = asarray(x),asarray(y)
    if numerix.shape(x) == numerix.shape(y) or numerix.shape(y) == ():
        try:
            return x - y * fix(x/y)
        except OverflowError:
            return None
    raise RuntimeError('Dimension error')


def norm(x,y=2):
    """
    Norm of a matrix or a vector according to Matlab.
    The description is taken from Matlab:
    
        For matrices...
          NORM(X) is the largest singular value of X, max(svd(X)).
          NORM(X,2) is the same as NORM(X).
          NORM(X,1) is the 1-norm of X, the largest column sum,
                          = max(sum(abs((X)))).
          NORM(X,inf) is the infinity norm of X, the largest row sum,
                          = max(sum(abs((X')))).
          NORM(X,'fro') is the Frobenius norm, sqrt(sum(diag(X'*X))).
          NORM(X,P) is available for matrix X only if P is 1, 2, inf or 'fro'.
     
        For vectors...
          NORM(V,P) = sum(abs(V).^P)^(1/P).
          NORM(V) = norm(V,2).
          NORM(V,inf) = max(abs(V)).
          NORM(V,-inf) = min(abs(V)).
    """

    x = asarray(x)
    if numerix.mlab.rank(x)==2:
        if y==2:
            return numerix.mlab.max(numerix.mlab.svd(x)[1])
        elif y==1:
            return numerix.mlab.max(asum(absolute((x))))
        elif y=='inf':
            return numerix.mlab.max(asum(absolute((transpose(x)))))
        elif y=='fro':
            return numerix.mlab.sqrt(asum(numerix.mlab.diag(matrixmultiply(transpose(x),x))))
        else:
            raise RuntimeError('Second argument not permitted for matrices')
        
    else:
        if y == 'inf':
            return numerix.mlab.max(absolute(x))
        elif y == '-inf':
            return numerix.mlab.min(absolute(x))
        else:
            return power(asum(power(absolute(x),y)),1/float(y))


def orth(A):
    """
    Orthogonalization procedure by Matlab.
    The description is taken from its help:
    
        Q = ORTH(A) is an orthonormal basis for the range of A.
        That is, Q'*Q = I, the columns of Q span the same space as 
        the columns of A, and the number of columns of Q is the 
        rank of A.
    """

    A     = array(A)
    U,S,V = numerix.mlab.svd(A)

    m,n = numerix.shape(A)
    if m > 1:
        s = S
    elif m == 1:
        s = S[0]
    else:
        s = 0

    tol = numerix.mlab.max((m,n)) * numerix.mlab.max(s) * _eps_approx
    r = asum(s > tol)
    Q = take(U,range(r),1)

    return Q

def rank(x):
        """
        Returns the rank of a matrix.
        The rank is understood here as the an estimation of the number of
        linearly independent rows or columns (depending on the size of the
        matrix).
        Note that numerix.mlab.rank() is not equivalent to Matlab's rank.
        This function is!
        """
        
	x      = asarray(x)
	u,s,v  = numerix.mlab.svd(x)
	# maxabs = numerix.mlab.max(numerix.absolute(s)) is also possible.
	maxabs = norm(x)	
	maxdim = numerix.mlab.max(numerix.shape(x))
	tol    = maxabs*maxdim*_eps_approx
	r      = s>tol
	return asum(r)

def sqrtm(x):
    	"""
	Returns the square root of a square matrix.
	This means that s=sqrtm(x) implies s*s = x.
	Note that s and x are matrices.
	"""
	return mfuncC(numerix.mlab.sqrt, x)

def mfuncC(f, x):
	"""
	mfuncC(f, x) : matrix function with possibly complex eigenvalues.
	Note: Numeric defines (v,u) = eig(x) => x*u.T = u.T * Diag(v)
	This function is needed by sqrtm and allows further functions.
	"""
	
	x      = array(x) 
	(v, u) = numerix.mlab.eig(x)
	uT     = transpose(u)
	V      = numerix.mlab.diag(f(v+0j))
	y      = matrixmultiply(
           uT, matrixmultiply(
           V, linear_algebra.inverse(uT)))
	return approx_real(y)

def approx_real(x):

	"""
	approx_real(x) : returns x.real if |x.imag| < |x.real| * _eps_approx.
	This function is needed by sqrtm and allows further functions.
	"""

	if numerix.mlab.max(numerix.mlab.max(absolute(x.imag))) <= numerix.mlab.max(numerix.mlab.max(absolute(x.real))) * _eps_approx:
		return x.real
	else:
		return x

### end mlab2 functions
