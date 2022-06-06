#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:24:16 2020

@author: lukejohnson1
"""

# =============================================================================
# Module Imports
# =============================================================================

import numpy as np
import pandas as pd
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

    plt.close("all")
except ImportError:
    print('could not load in matplotlib packages, visualisation methods unavailable')
import scipy.constants as con
import scipy.signal as sig
import scipy.interpolate as itp
import scipy.stats as stats
from scipy.stats import binned_statistic as bs
import os
import time
from astropy.convolution import convolve, Box1DKernel
import warnings

# =============================================================================
# Functions Below
# =============================================================================

def sin(x):
    return np.sin(np.deg2rad(x))

def arcsin(x):
    return np.rad2deg(np.arcsin(x))

def cos(x):
    return np.cos(np.deg2rad(x))

def arccos(x):
    return np.rad2deg(np.arccos(x))

def timefn(fn, N=1):
    start = time.time()
    
    for _ in range(N):
        fn()
    end = time.time()
    return end-start

def sigma_interval(mean, sig, pc=0.68):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yl, yh = stats.norm.interval(pc, loc=mean, scale=sig)
        return yl, yh
    
def sigma_interval2(a, confidence=0.68):
    a = 1.0 * np.array(a)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def find_nearest(array, value, retidx=False, force=None):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    N = len(array)
    
    if force!=None:
        force=force.lower()
        if force not in ['down', 'up']:
            raise Exception("force must be None, 'down', or 'up'")
    
        if 'down' in force and array[idx] > value:
            idx -= 1
            if idx < 0:
                idx = 0
        elif 'up' in force and array[idx] < value:
            idx += 1
            if idx > N-1:
                idx = N-1
    
    if retidx == True:
        return idx
    else:
        return array[idx]
    
def linear(x, a, b):
    return a*x + b

def linearinv(y, a, b):
    return (y - b)/a

def typename(x):
    return type(x).__name__


def transittimes(lc):
    N = len(lc)
    for i in range(N):
        if lc[i] < 1.0:
            break
    
    Ni = N - 2*(i-1)
    
    xx = np.linspace(-0.5, 0.5, Ni)
    delx = xx[1] - xx[0]
    
    xlo = xx[0] - (i-1)*delx
    xhi = xx[-1] + (i-1)*delx
    
    return np.linspace(xlo, xhi, N)
    
    
def transitpos(xsize=800, pad=100, angle=0.0, b=0.0, N=101, otherb=False,
               plot=False, figname=None):
    
    size = xsize + 2*pad
    
    vec = np.linspace(-size, size, 11)
    
    imp = b*(xsize/2)
    
    if np.abs(angle) > 45.0 and otherb:
        m = 'x'
    else:
        m = 'y'
        
    if angle < 0.0 and otherb:
        imp = -imp
        
    
    X=vec*cos(angle)

    Y=vec*sin(angle)
    
    if m=='x':
        X = X - imp
    if m=='y':
        Y = Y + imp
        
    
    if plot==True:
        plt.figure(figname)
        ax = plt.subplot(111)
        plt.xlim(-size/2 - 200, size/2 + 200)
        plt.ylim(-size/2 - 200, size/2 + 200)
        rect = plt.Rectangle((-size/2 - 50, -size/2 - 50), size + 100, size + 100, fill=False)
        circ = plt.Circle((0,0), size/2, fill=False)
        ax.add_artist(rect)
        ax.add_artist(circ)
        plt.scatter(0, 0, color='black')
        plt.axvline(0, ls='--', color='black', zorder=1)
        plt.axhline(0, ls='--', color='black', zorder=1)
        
        
        plt.plot(X, Y, label=r'$\theta = {}^\circ,\ b = {}$'.format(angle, b), zorder=10)
        
    else:
        X = X + size/2
        Y = Y + size/2
        
        m, c, r, p, err = stats.linregress(X, Y)
        
        if angle==0.0:
            xx = np.linspace(0, size, N)
            yy = linear(xx, m, c)
        elif np.abs(angle)==90.0:
            yy = np.linspace(0, size, N)
            xx = linearinv(yy, m, c)
        else:
            x0 = linearinv(0, m, c)
            y0 = linear(0, m, c)
            x1 = linearinv(size, m, c)
            y1 = linear(size, m, c)
                        
            if x0 >= 0 and x0 <= size:
                lo_x = x0
            elif angle >= 0:
                lo_x = 0
            else:
                lo_x = size
            
            if x1>=0 and x1<=size:
                hi_x = x1
            elif angle >= 0:
                hi_x = size
            else:
                hi_x = 0
                

            xx = np.linspace(lo_x, hi_x, N)
            
            if lo_x > hi_x:
                xx = xx[::-1]
            
            yy = linear(xx, m, c)
        
        return xx, yy
    
def lognormal(x, mx=9.21, mu=0.57, sig=3.95, utoavg=5):
    x1 = x/utoavg
    a = -(np.log(x1) - np.log(mu))**2
    b = 2*np.log(sig)
    
    return mx * np.exp(a/b)

def lognormalSU(x, mx=9.21, mu=0.57, sig=3.95, lowlim=7.5):

    y = lognormal(x, mx, mu, sig, 5)
    y[x<7.5] = 0
    
    return y

def lognormalPDF(factor=1e2, grid = np.linspace(7, 1000, 994), prnt=True):
    x = grid
    y = lognormalSU(x)
    N = len(x)
    
    ymin = y[y!=0].min() 
#    xprint(ymin)
    
    y_mul = y/ymin * factor
    y_ints = y_mul.astype(int)
    
    
    data = []
    for i in range(N):
        val = x[i]
        count = y_ints[i]
        if prnt==True:
            print(val, count)
        for _ in range(count):
            data.append(val)
            
    hist = np.histogram(data, bins=N)
    
    hist_dist = stats.rv_histogram(hist)
    
    return hist_dist
    
def betafunc(x, a, b, c, d):
    
    return d*stats.beta.pdf(x, a, b) + c

def lognormfunc(x, a, b, c, d):
    
    return c*stats.lognorm.pdf(x+b, a)+d
    

def planck_wl(wl, T):
    h = con.h
    c = con.c
    k = con.k
    A = 2*h*c**2 / wl**5
    B = np.exp(h*c/ (wl*k*T)) - 1
    return A/B


def CalcVar(LC, method='RVar', bound=0.8, sep='peaks', tbin=0.01, retSD=False):
    
    method = method.upper()

    mlist = ['RVAR', 'AMP', 'MDV', 'RMS', 'SDR']
    
    if sep not in ['peaks', 'dips']:
        raise Exception("sep must be one of either 'peaks' or 'dips'")
    #print(LC)
    x = np.linspace(0, 1, len(LC)+1)[:-1]
    
    
    if method=='RVAR':
        P5 = np.percentile(LC, 5)
        P95 = np.percentile(LC, 95)
        return np.abs(P5 - P95)
    
    elif method=='AMP':
        return LC.max() - LC.min()
    
    elif method=='MDV': #not sure
        
        xnew = np.linspace(0, 1, 1/tbin)[:-1]
        
        LCin = np.interp(xnew, x, LC)
        
        N = len(LCin)
        diffs = np.zeros(N)
        
        for i in range(N):
            diffs[i] = np.abs(LCin[i] - LCin[i-1])
            
        return np.median(diffs)
    
    elif method=='RMS':
        
        med = np.median(LC)
        return np.sqrt(((LC - med) ** 2).mean())
    
    elif method=='SDR':
        
        dt = x - np.roll(x, 1)
        dt = dt[1:]
        ddt = np.median(dt)
        prot=1 #always 1, units of phase
        width = int(np.round(prot/(ddt*8)))
        
        LC3 = np.concatenate([LC, LC, LC])
        x3 = np.concatenate([x-1, x, x+1])
        
        #plt.plot(x3, LC3)
        LC3 = convolve(LC3, Box1DKernel(width), 'wrap') #smoothing
        #lt.plot(x3, LC3)
        
        if sep=='peaks':
            peaks3 = sig.find_peaks(LC3)[0]
        elif sep=='dips':
            peaks3 = sig.find_peaks(-1*LC3)[0]
        #dips3 = sig.find_peaks(-1*LC3)[0]

        xpeaks = x3[peaks3]
        #print('xpeaks ', xpeaks)

        psep = xpeaks - np.roll(xpeaks, 1)
        psep = psep[1:]

        try:
            qwe = np.array([xpeaks[0] - xpeaks[-1] + 3.])
            psep = np.concatenate([psep, qwe])
        except IndexError:
            pass
        
        psep = np.round(psep, 13)
        psep, counts = np.unique(psep, return_counts=True)
        try:
            if psep[-1]==1. and len(psep)>1:
                psep = psep[:-1]
        except IndexError:
            pass
            counts = counts[:-1]
        #print('psep ', psep, counts)
        
        #tdb = np.sum(psep<=bound)
        #tsn = np.sum(psep>bound)
#        
#        print(tdb, tsn)
#        
#        rat = tsn/tdb #single/double ratio
#        
#        
#        if rat > 100:
#            rat = 100
#        if rat < 0.01:
#            rat = 0.01
        if retSD==False:    
            return psep, xpeaks
        elif retSD==True:
            if len(psep) < 1:
                return 's'
            elif psep[-1] > 0.8:
                return 's'
            else:
                return 'd'
        else:
            raise Exception("retSD must be True or False")
    else:
        raise Exception("method must be one of the following: {}".format(mlist))
        

def MLNA(lc, starmag=12, cadence='SC', seed=None, interplen=None, fn=np.random.normal, lcwrap=True):
    """
    MAKE LIGHTCURVES NOISY AGAIN: Gaussian white noise generator
    """
    
    starmag = float(starmag)
    
    if fn not in [np.random.normal, np.random.uniform]:
        raise Exception("fn must be np.random.normal or np.random.uniform")
    
    if cadence not in ['SC', 'LC', '5h']:
        raise Exception("cadence must be 'SC', 'LC', or '5h'")
        
    pdat = pd.read_csv('~/Dropbox/Luke/Imperial/Year2/v6/data/KeplerPrecision/KpSCLC5h.csv')
    
    Kp = pdat['Kp'].values
    prec = pdat[cadence].values
    
    ppmnoise = np.interp(starmag, Kp, prec)
    
    if lcwrap==True:
        lc = np.append(lc, lc[0])
        x = np.linspace(0, 1, len(lc))
        if interplen!=None:
            interplen += 1
    
    if interplen!=None:
        xnew = np.linspace(0, 1, interplen)
        lc = np.interp(xnew, x, lc)
        x = xnew
    
    if lcwrap==True:
        lc = lc[:-1]
        x = x[:-1]
    
    N = len(lc)
    mu = lc.mean()
    noise = mu*(ppmnoise*1e-6)
    
    mol = 0
    if fn is np.random.uniform:
        mol = -noise
    
    np.random.seed(seed)
    s = fn(mol, noise, N)
    
    return lc + s, x

def KLN(lc, sigma=1000,  seed=None, interplen=None, fn=np.random.normal, lcwrap=True):
    """
    KEEP LIGHTCURVES NOISY: Gaussian white noise generator 2020
    """
        
    if lcwrap==True:
        lc = np.append(lc, lc[0])
        x = np.linspace(0, 1, len(lc))
        if interplen!=None:
            interplen += 1
    
    if interplen!=None:
        xnew = np.linspace(0, 1, interplen)
        lc = np.interp(xnew, x, lc)
        x = xnew
    
    if lcwrap==True:
        lc = lc[:-1]
        x = x[:-1]
    
    N = len(lc)
    mu = lc.mean()
    noise = mu*(sigma*1e-6)
    
    mol = 0
    if fn is np.random.uniform:
        mol = -noise
    
    np.random.seed(seed)
    s = fn(mol, noise, N)
    
    return lc + s, x
    
def GetX(N=90, units='phase'):
    """
    Get x-axis (phase) for lightcurves (outside of class)
    """
    if units=='phase':
        mx = 1
    elif units=='degrees':
        mx = 360
    else:
        raise Exception("'units' must be 'phase' or 'degrees'")
        
    return np.linspace(0, mx, N+1)[:-1]    


def nonlin3_nn(mu, I0, a, b, c, no_negatives=True):
    """
    3-parameter non-linear limb darkening fit
    """
    y = I0 * (1 - a*(1-mu) - b*(1-mu**(3/2)) - c*(1-mu**2))
    
    if no_negatives == True:
        y[y < 0] = 0
    
    return y
    
def nonlin3(mu, I0, a, b, c):
    """
    3-parameter non-linear limb darkening fit
    """
    y = I0 * (1 - a*(1-mu) - b*(1-mu**(3/2)) - c*(1-mu**2))
    
    return y



def nonlin3_norm(mu, a, b, c):
    """
    3-parameter non-linear limb darkening fit
    """
    #return 1 * (1 - a*(1-mu) - b*(1-mu**(3/2)) - c*(1-mu**2))
    return nonlin3(mu, 1, a, b, c)



def MSHtoR(x):
    path = '/Users/lukejohnson1/Dropbox/Luke/Imperial/Year2/v5/'
    data = pd.read_csv(path+ 'Av5_area_radii.csv')
    ff = data.ff.values
    R = data.r.values
    MSH = ff*1e6 / 2
    return np.interp(x, MSH, R)

def RtoMSH(x):
    path = '/Users/lukejohnson1/Dropbox/Luke/Imperial/Year2/v5/'
    data = pd.read_csv(path+ 'Av5_area_radii.csv')
    ff = data.ff.values
    R = data.r.values
    MSH = ff*1e6 / 2
    return np.interp(x, R, MSH)

def HEMtoR(x, path='/actress/rf_relation_060320.npy'):
    try:
        cwd = os.path.dirname(os.path.abspath(__file__))
        #print(cwd)
    except NameError:
        #print(Warning("Running in console"))
        cwd = os.getcwd()
        
    if cwd[-7:] == 'actress':
        cwd = cwd[:-8]
    
    data = loadnpy(cwd+path)
    
    fs = data['f']
    rs = data['r']
    
    A = x *1e-6 /2
    
    return np.interp(A, fs, rs)

def radfillrelation(rads='default', res=12, prnt=True):
    
    if rads=='default':
        rads = np.linspace(0, 60, 601)[:-1]
        rads = np.concatenate((rads, np.linspace(60, 120, 21)[:-1]))
        rads = np.concatenate((rads, np.linspace(120, 180, 601)))
    
    sim = Simulator([], [], resolution=res)
    
    ffs = []
    
    for i in rads:
        sim.SetFeatureList([{'r':i, 'lon':0, 'lat':0}])
        f = sim.GetFF()
        ffs.append(f)
        if prnt==True:
            print('r = {} deg, f = {}%'.format(np.round(i, 3), np.round(f, 3)))
            
    ffs = np.array(ffs)
    
    return {'r':rads, 'f':ffs}
            
    
    
    

def distlims(mu=None, sig=None, lims=None, func=np.random.random, rstate=None):
    
    if isinstance(mu, stats._continuous_distns.rv_histogram):
        
        draw = mu.rvs(random_state=rstate)
        if lims!=None:
            
            while draw < lims[0] or draw > lims[1]:
#                print(draw)
                draw = mu.rvs(random_state=rstate)
        
        return draw #random sampling from emre's PDFs
    
    if func == np.random.random:
        if isinstance(mu, int) or isinstance(mu, float):
            pass
        else:
            mu = 1.0
        
        x = mu*func()
        if lims!=None:
            if len(lims)!=2:
                raise Exception("lims must be length 2 (min and max value) if not 'None'")
            if lims[1] <= lims[0]:
                raise Exception("max (index 1) must be higher than min (index 0)")
            while x <= lims[0] or x >= lims[1]:
                x = mu*func()
        return x
        
        
    else:
        if mu==None or sig==None:
            raise Exception("mu and sig must have a value if func!=np.random.random")
    x = func(mu, sig)
    if lims!=None:
        if len(lims)!=2:
            raise Exception("lims must be length 2 (min and max value) if not 'None'")
        if lims[1] <= lims[0]:
            raise Exception("max (index 1) must be higher than min (index 0)")
        while x <= lims[0] or x >= lims[1]:
            x = func(mu, sig)
    return x


def TfromMURaM(SpectralClass, mag, retflux=False, path='/actress/muramdata/'):
    try:
        cwd = os.path.dirname(os.path.abspath(__file__))
        #print(cwd)
    except NameError:
        #print(Warning("Running in console"))
        cwd = os.getcwd()

        
    if cwd[-7:] == 'actress':
        cwd = cwd[:-8]
    
    skippy = None
    sep = " "

    filepath = cwd + path + 'intbL2012_atlkurab_vt1lit_' + \
    SpectralClass + '_' + mag + '_meanspectra_LD'
    
    #if SpectralClass == 'G2' and mag=='hydro':
    #    filepath = '~/Dropbox/MuramShare/MURaM Mean Spectra/int_vt1lit_G2_hydro_meannewspectra_LD'
    #elif SpectralClass == 'G2' and (mag!='hydro'):
    if SpectralClass =='G2':
        filepath = cwd + path + 'intbL2012_atlkurab_vt1lit_G2_' + mag + '_meanoldspectra_LD'

    
    if mag != 'hydro':
        sep = "\t"
        skippy=2
    
    if SpectralClass == 'F3':
        filepath = filepath.replace('vt1', 'vt2')
          
    cols = ['wl', 'mu10', 'mu09', 'mu08', 'mu07', 'mu06', 'mu05', 'mu04', 'mu03', 'mu02']

    DF = pd.read_csv(filepath, sep=sep, header=None, names=cols, skiprows=skippy)
    l1 = len(DF)
    DF = DF.dropna()   #remove nan rows
    l2 = len(DF)
    
    Ndroppedrows = l1 - l2
    mu = np.linspace(1, .2, 9)

#EVERYTHING IN CGS UNITS
    #ClassTemps = {'G2':5792, 'K0':4894, 'M0':3849, 'M2':3602}



    c = 3e10
    #sig = 5.6704e-5
    sig = con.sigma *1e3

    #sigT4 = Tc**4 * sig


    F = []
    for i in range(len(DF)):
        i = i+Ndroppedrows
    
        row = DF.loc[i]
        Iwl = row[1:].values * c / (row[0]*1e-7)**2
        #flux = 4 * np.pi * np.trapz(Iwl*mu, x=mu)
        flux = 2 * np.pi * np.trapz(Iwl*mu, x=mu)
        F.append(-flux)



    F = np.array(F)
    
    F = np.delete(F, 3)
    wl = np.delete(DF['wl'].values, 3) * 1e-7
    st4 = np.trapz(F, x=wl)


    Teff = (st4/sig)**0.25

    if retflux==True:
        return st4
    return Teff

def TfromLJ(SpectralClass, feature='photosphere'):
    
    dp = {'G2':5792, 'K0':4894, 'M0':3849, 'M2':3602}
    ds = {'G2':4785, 'K0':4086, 'M0':3273, 'M2':3080}
    
    if feature=='photosphere':
        d = dp
    elif feature=='spot':
        d = ds
    else:
        raise Exception("feature must be 'photosphere' or 'spot'")
    
    return d[SpectralClass]

def SpotTRackham(Teff):
    return 0.418 * Teff + 1620.

def SpotTLJ(Teff):
    
    A = 2.21407344e-01
    B = -2.75657965e+02
    delT = A*Teff + B
    return Teff - delT

def FFS14(s, feature='facula'):
    """
    sunspot disk-area coverage AS :
        As = (0.105 ± 0.011) − (1.315 ± 0.130)S + (4.102 ± 0.370)S**2
    and for facular disk-area coverage AF:
        Af = −(0.233 ± 0.002) + (1.400 ± 0.010)S
    
    
    """
    if feature=='spot':
        A, B, C = 0.105, -1.315, 4.102
    elif feature=='facula':
        A, B, C = -0.233, 1.40, 0
    else:
        raise Exception("feature must be 'spot' or 'facula'")
    return A + B*s + C*s**2

def SFF14(ff, feature='facula'):
    
    sarr = np.linspace(0.16, 0.20, 2000)
    farr = FFS14(sarr, feature=feature)
    
    return np.interp(ff, farr, sarr)


def Qfromff(ffspot, x = np.linspace(0.17, 0.55, 1000)):
    
    yf = FFS14(x)
    ys = FFS14(x, 'spot')
    rat = yf/ys
    return np.interp(ffspot, ys, rat)

def QfromS(s):
    yf = FFS14(s)
    ys = FFS14(s, 'spot')
    return yf/ys
        
    


def rackrf(r, Q):
    return  np.sqrt((Q/2 + 1) * r**2)

def lattoY(lat, pad=0):
    if hasattr(lat, "__len__") == False:
        lat = [lat]
    
    Y = []
    for i in lat:
        if i > 90 or i < -90:
            raise Exception("latitude must be between -90, 90 degrees")
        coeff = 1
        add = 0
        if i < 0:
            coeff = -1
            add = 800
        

        y =  coeff*(np.cos(i*np.pi/180)*400) + add + pad
        Y.append(y)
    
    Y = np.array(Y)
    if len(Y)==1:
        Y = Y[0]
    return Y
    
    
def reshape2D(arr):
    
    arrnew = []
    N = len(arr)
    Nx = len(arr[0])

    for i in range(Nx):  
        A1 = []
        for j in range(N):    
            A1.append(arr[j][i])
        arrnew.append(A1)
    
    return np.array(arrnew)

def loadnpy(path, ftype='.npy'):
    """
    load .npy files with numpy.load
    """
    if path[-4:] != ftype:
        path += ftype
    try:
        data = np.load(path).tolist()
    except ValueError:
#        print('ValueError, pickling allowed')
        data = np.load(path, allow_pickle=True).tolist()
    
    return data



def actresscombine(savename, folder, 
                   path = '/Users/lukejohnson1/Dropbox/Luke/Imperial/Year2/v5/data/'):
    
    datum = []
    
    files = os.listdir(path+folder)
    
    for j in files:
        if j[0]==savename[0]:
            data = loadnpy(path+folder+j)
            datum.append(data)
    
    np.save(savename, datum)
    

def TransitDuration(a, rp, rs, i, P):
    """
    Like PyAstronomy's function, but actually works (actually does it work??)
    
    Parameters
    ----------
    a : float
        The semi-major axis in AU.
    rp : float
        The planetary radius in Jovian radii.
    rs : float
        The stellar radius in solar radii.
    i : float
        The orbital inclination in degrees.
    P : float
        The orbital period.
    
    Returns
    -------
    Transit duration : float
        The duration of the transit (same units as
        `period`).
    """
    
    rs = rs * 6.957e8
    rp = rp * 69911e3
    a = a*con.au
    
    b = a/rs * np.cos(i*np.pi/180)
    
    num = np.sqrt((rs+rp)**2 - b**2)
    
    return P/np.pi * np.arcsin(num/a)

def LDA(mu):
    
    angle = np.arccos(mu)
    
    annular_rad = np.sin(angle)
    
    return annular_rad

    

    
def LimbDistancesonDisc(sim, mus, pad=20, fig=None, ax=None, lw=2, whitebg=False,
                        col='black', stw2=False, xsize=800, vmin=None):
    if fig==None or ax==None:
        fig, ax = plt.subplots()
    
    sim.setxsize=xsize
    MAT = sim.stellarmodel(pad=pad)
    if whitebg==True:
        MAT[MAT == 0] = np.nan 
        vmin=0.3
    ax.imshow(MAT, cmap='plasma', vmin=vmin)
    plt.xticks([])
    plt.yticks([])
    r = (xsize/2)+pad
    for i in mus:
        
        if i==0.2 and stw2==True:
            col = 'white'
        
        rad = LDA(i)
        if rad != 0:
            circle = plt.Circle((r, r), (xsize/2)*rad, color=col, fill=False, ls='--', lw=lw, alpha=0.75)
        else:
            circle = plt.Circle((r, r), 3, color=col)
        
        ax.add_artist(circle)
        
        
def ColsFromMap(N, cmap='plasma', lims=[0.0, 0.85]):
    mappy = mpl.cm.get_cmap(cmap)
    x = np.linspace(*lims, N)
    return mappy(x)

def TickSetup(ax, xmaj, xmin, ymaj, ymin, lmaj=7, lmin=4, leaveit=False, yonly=False):
    if leaveit==False:
        if yonly==False:
            ax.xaxis.set_major_locator(MultipleLocator(xmaj))
            ax.xaxis.set_minor_locator(MultipleLocator(xmin))
        else:
            pass
        
        ax.yaxis.set_major_locator(MultipleLocator(ymaj))
        ax.yaxis.set_minor_locator(MultipleLocator(ymin))
        
    ax.tick_params(which='both', direction='inout', top=True, right=True)
    ax.tick_params(which='major', direction='inout', length=lmaj)
    ax.tick_params(which='minor', direction='inout', length=lmin)
        

def AvgTSpot(Tu, Tp, ratio=[1, 4]):
    ratio = np.array(ratio)
    
    
    res = (ratio[0]*Tu**4 + ratio[1]*Tp**4)/ratio.sum()
    
    return res**0.25

def LDtoLon(mu):
    
    return np.arccos(mu)*180/np.pi

def Qradrat(r, Q):
    return r * np.sqrt(Q+1)
    
def makeffgrid(*startstopstep):
    ffgrid = np.array([])
    for i in startstopstep:
        if len(i)!=3:
            raise Exception("each entry must be length 3. Current length: {}".format(len(i)))
        new = np.arange(*i)
        ffgrid = np.append(ffgrid, new)
        
    return ffgrid


def extrap1d(interpolator):
    """
    1D linear extrapolator from SO:
    https://stackoverflow.com/questions/2745329/
    how-to-make-scipy-interpolate-give-an-extrapolated-result-beyond-the-input-range
    """
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        if hasattr(xs, "__len__") == False:
            xs = [xs]
        
        res = np.array(list(map(pointwise, np.array(xs))))
        if len(res)==1:
            res = res[0]
        return res
    
    return ufunclike

def S2Lat(s):
    x = np.array([0.18268001925568367, 0.1971117451749696])
    y = np.array([16.0, 30.0])
    ifunc = itp.interp1d(x, y)
    xfunc = extrap1d(ifunc)
    return xfunc(s)


def lonband(mu, sig, N=1, func=np.random.normal, maxlon=360.):
    
    draw = func(mu, sig, N)
    
    if len(draw)==1:
        draw = draw[0]
    
    return draw % maxlon

def rotatearray(arr, n):
    
    l = list(arr)
    n = int(n)
    rl = l[n:] + l[:n]
    
    return np.array(rl)

def quadratic(x, a, b, c):
    
    return a*x**2 + b*x + c

def fourparam(x, a, b, c, d):
    
    return a*x**3 + b*x**2 + c*x + d

def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def rebin_edge(grid='normal'):
    
    grid = grid.lower()
    if grid=='fine':
        interp = makeffgrid([0, 0.1, 0.01], [0.1, 1.0, 0.05], [1.0, 2.7, 0.1])/100
    elif grid=='normal':
        interp = makeffgrid([0, 1.0, 0.1], [1.0, 3.0, 0.2], [3.0, 5.0, 0.5], [5.0, 22.0, 1.0])/100
    
    edges = (interp[1:] + interp[:-1]) / 2
    edges = list(edges)
    edges.insert(0, 0)
    return edges
    

def rebin_pd(x, y, grid='normal'):
    
    grid = grid.lower()
    if grid=='fine':
        interp = makeffgrid([0, 0.1, 0.01], [0.1, 1.0, 0.05], [1.0, 2.6, 0.1])/100
    elif grid=='normal':
        interp = makeffgrid([0, 1.0, 0.1], [1.0, 3.0, 0.2], [3.0, 5.0, 0.5], [5.0, 21.0, 1.0])/100
    else:
        interp = makeffgrid([0, 21, 0.5])/100



    df = pd.DataFrame({'X': x, 'Y': y})
    

    df['Xbins'] = np.digitize(df.X, edges)
    df['Ymean'] = df.groupby('Xbins').Y.transform('mean')
    df.plot(kind='scatter', x='X', y='Ymean')

def rebin_bs(x, y, grid='normal'):
    ax = plt.subplot(111)
    TickSetup(ax, 0.5, 0.1, 2, 1)
    
    e = rebin_edge(grid)
    e = np.array(e)*100
    
    plt.scatter(x, y, alpha=0.2, marker='o', color='blue', s=5)
    
    s, edges, _ = bs(x,y, statistic='mean', bins=e)
    print(s)
    ys = np.repeat(s,2)
    xs = np.repeat(edges,2)[1:-1]
    plt.hlines(s,edges[:-1],edges[1:], color="crimson", lw=2)
    for e in edges:
        plt.axvline(e, color="grey", linestyle="--", alpha=0.5, lw=0.5)
    
    plt.xlabel('Aspot [%]', fontsize=22)
    plt.ylabel('Rvar [ppt]', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # plt.scatter(edges[:-1]+np.diff(edges)/2, s, c="limegreen", zorder=3, alpha=0.5)


def textposlims(low, high, frac):
    
    diff = high - low
    return low + diff*frac
    
    
    