#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:32:28 2020

@author: lukejohnson1
"""

# =============================================================================
# Module Imports
# =============================================================================

import numpy as np
import healpy as hp
import pandas as pd
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.animation as ani
    from matplotlib import gridspec
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
    plt.close("all")
except ImportError:
    print('could not load in matplotlib packages, visualisation methods unavailable')

import scipy.interpolate as itp
import os
import functools
import time
from photutils import CircularAperture as CAp
from photutils import aperture_photometry as APh
import joblib as jl
import copy


try:
    from .handyfuncs import *
except:
    try:
        from actress.handyfuncs import *
    except:
        from handyfuncs import *
        
try:
    from .feature import *
except:
    try:
        from actress.feature import *
    except:
        from feature import *
        
        
        
try:
    from .hmap import *
except:
    try:
        from actress.hmap import *
    except:
        from hmap import *



# =============================================================================
# Simulator Class
# =============================================================================

class Simulator():
    
    def __init__(self, faculae=[], spots=[], resolution=10, xsize=800,
                 fac_strips=[], spot_strips = [],
                 ld = {'phot':[2.33700574020726e21	, 1.512080132947490,-1.2083767839286600, 0.3987324436660390],
                       'fac':[2.36627457833986e21,	1.6191577579961700,	-1.40167196283501,	0.4789048361142440],
                       'spot':[1.39995037645599e21, 1.1846177104142300, -0.7259453209422660, 0.25922887554647400],
                       'func':nonlin3}):
        """
        actress: the active rotating star simulator
        
        
              __  ___  __   ___  __   __  
         /\  /  `  |  |__) |__  /__` /__` 
        /~~\ \__,  |  |  \ |___ .__/ .__/ 
                                          

        Parameters
        ----------
        faculae : LIST, optional
            Facular regions on the photosphere. The default is [].
        spots : LIST, optional
            Spots on the photosphere. The default is [].
        resolution : INT, optional
            Resolution of the HealPix map (1 - 30). The default is 10.
        xsize : INT, optional
            Resolution of the stellar disc projection. The default is 800.
        spotmode : STR, optional
            Spots intensities can either be provided outright ('absolute') or as a contrast ('contrast'). The default is 'contrast'.
        ld : DICT, optional
           Limb-dependent intensity coefficients. The default is for a G2 star with 100G facular regions and spots with T=5150K.
           
        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.__faculae = faculae
        self.__spots = spots
        
        self.__fstrips = fac_strips
        self.__sstrips = spot_strips
        
        self.__xs = xsize
        

        
        if (resolution < 1) or (resolution > 30):
            raise Exception("resolution must be an integer between 1 and 30 ({} provided)".format(resolution))
        self.__res = 2**int(resolution)
        
        
        self.__ld = ld
        
        self.__photmask = self.intensitymasks('phot')
        self.__spotmask = self.intensitymasks('spot')
        self.__facmask = self.intensitymasks('fac')
        
        self.__dphot = 1 #dummy variables
        self.__dspot = 2
        self.__dfac  = 3
    
    
    def setxsize(self, newxs):
        
        self.__xs = newxs
        
        self.__photmask = self.intensitymasks('phot')
        self.__spotmask = self.intensitymasks('spot')
        self.__facmask = self.intensitymasks('fac')
    
    def getxsize(self):
        
        return self.__xs
    
    def setresolution(self, newres):
        
        if (newres < 1) or (newres > 30):
            raise Exception("resolution must be an integer between 1 and 30 ({} provided)".format(newres))
        self.__res = 2**int(newres)
        
    def getresolution(self):
        
        return int(np.log2(self.__res))
        
        
        
    def addfeature(self, r, lon=None, lat=None, feature='fac'):
        """
        Add facular region or spot to the photosphere.

        """
        
        if typename(r) == 'Facula':
            self.__faculae.append(r)
            
        elif typename(r) == 'Spot':
            self.spots.append(r)
        
        else:
            feature = feature.lower()
            if isinstance(r, dict):
                keys = ['r', 'lon', 'lat']
                for i in r.keys():
                    if i not in keys:
                        raise Exception("If a dict object is entered, it must have the following keys: {}".format(keys))
                lon, lat, r = r['lon'], r['lat'], r['r']
            else:
                if lon is None or lat is None:
                    raise Exception("If a Spot, Facula or dict object is not entered, lon and lat must be defined")
            
            if feature=='fac':
                feat = Facula(r, lon, lat)
                self.__faculae.append(feat)
            elif feature=='spot':
                feat = Spot(r, lon, lat)
                self.__spots.append(feat)
            else:
                raise Exception("feature must be 'fac' or 'spot'")
                
            
    def addstrip(self, lower, upper, feature='fac'):
        
        feature = feature.lower()
        if feature=='fac':
            strip = Fac_Strip(lower, upper)
            self.__fstrips.append(strip)
        elif feature=='spot':
            strip = Spot_Strip(lower, upper)
            self.__sstrips.append(strip)
    
    def getstrips(self, feature='fac'):
        
        feature = feature.lower()
        
        if feature=='fac':
            return self.__fstrips
        elif feature=='spot':
            return self.__sstrips
        else:
            raise Exception("feature must be 'fac' or 'spot'")
            
    def setfeaturelist(self, newfeaturelist, feature='fac'):
        
        feature = feature.lower()
        
        if isinstance(newfeaturelist, list)==False:
            raise Exception("newfeaturelist must be a list")
        
        
        if feature=='fac':
            for i in newfeaturelist:
                if 'Facula' not in typename(i):
                    raise Exception("For feature='fac', all elements of newfeaturelist must be instances of Facula (element is {})".format(type(i)))
            self.__faculae = newfeaturelist
        elif feature=='spot':
            for i in newfeaturelist:
                if 'Spot' not in typename(i):
                    raise Exception("For feature='spot', all elements of newfeaturelist must be instances of Spot (element is {})".format(type(i)))
            self.__spots = newfeaturelist
        else:
            raise Exception("feature must be 'fac' or 'spot'")
        
            
    def getfeaturelist(self, feature='fac'):
        
        feature = feature.lower()
        
        if feature=='fac':
            return self.__faculae
        elif feature=='spot':
            return self.__spots
        else:
            raise Exception("feature must be 'fac' or 'spot'")
            
            
    def removefeatureidx(self, idx, feature='fac'):
        
        feature = feature.lower()
        
        if feature=='fac':
            del self.__faculae[idx]
        elif feature=='spot':
            del self.__spots[idx]
            
    def setld(self, ld):
        
        if isinstance(ld, dict):
            self.__ld = ld
        else:
            raise Exception("'ld' must be a dictionary")

        self.__photmask = self.intensitymasks('phot')
        self.__spotmask = self.intensitymasks('spot')
        self.__facmask = self.intensitymasks('fac')


            
    def setld_from_spectra(self, spectype, mag, teff, tspot, filter_conv='kepler',
                           mode='wavelength', rebin_wl=None, path=None, wl_min=200.0, filtpath=None):

        if filter_conv is None or isinstance(filter_conv, list)  or \
            isinstance(filter_conv, np.ndarray) or filter_conv=='all':
            raise Exception("cannot set multiple")
        else:
            ld = actress_ld(spectype, mag, teff, tspot, filter_conv, mode, rebin_wl, path, wl_min, filtpath)
            self.setld(ld)
            
    def getld(self, feature='all'):
        
        if feature=='all':
            return self.__ld
        else:
            if feature not in self.__ld.keys():
                raise Exception("feature must be 'all' or one of the following: {}".format(self.__ld.keys()))
            return self.__ld[feature]
            
        
    def getfill(self, feature='fac', mode='both'):
        """
        Returns total surface coverage as a fraction for chosen feature
        """
        
        if feature not in ['photosphere', 'spot', 'fac', 'spot+fac']:
            raise Exception("feature must be 'photosphere', 'spot', 'fac' or 'spot+fac'")
        
        
        DICT = {'photosphere':1, 'spot':2, 'fac':3}
        if feature=='spot+fac':
            if len(self.getfeaturelist())==0 and len(self.getstrips())==0 and \
               len(self.getfeaturelist('spot'))==0 and len(self.getstrips('spot'))==0:
                return 0.0
            hmap = self.makemap(mode=mode)
            ff = len(hmap[hmap==2])/len(hmap)
            ff += len(hmap[hmap==3])/len(hmap)
            return ff
        elif feature=='photosphere':
            pass
        else:
            if len(self.getfeaturelist(feature))==0 and len(self.getstrips(feature))==0:
                return 0.0
            
        key = DICT[feature]
        
        hmap = self.makemap(mode=mode)

        ff = len(hmap[hmap==key])/len(hmap)
        
        return ff
    
    def getdiscfill(self, feature='fac', mode='both', rot=0.0, inc=90):
        """
        Returns disc filling factor for chosen feature
        """
        
        if feature not in ['photosphere', 'spot', 'fac', 'spot+fac']:
            raise Exception("feature must be 'photosphere', 'spot', 'fac' or 'spot+fac'")
        
        MAT = self.stellarmodel(rot=rot, inc=inc, mode=mode, ldkey=True)
        
        DICT = {'photosphere':1, 'spot':2, 'fac':3}
        
        if feature=='spot+fac':
            s = self.getdiscfill('spot', mode=mode, rot=rot, inc=inc)
            f = self.getdiscfill('fac', mode=mode, rot=rot, inc=inc)
            return s+f
        
        k = DICT[feature]
        
        num = len(MAT[MAT==k])
        den = len(MAT[MAT!=0])
        return num/den


    def intensitymasks(self, feature='phot'):
        
        feature = str(feature.lower())
        fts = ['phot', 'spot', 'fac']
        if feature not in fts:
            raise Exception("feature must be one of the following: {}".format(fts))
        
        x = np.linspace(-1, 1, self.__xs)
        y = np.linspace(-1, 1, self.__xs)
        xx, yy = np.meshgrid(x, y)
        arg = np.sqrt(xx**2 + yy**2)
        
            
        u2 = 1 - arg**2
        u2 = (np.abs(u2)+u2)/2 #changes -ve values to zero, no RuntimeWarning
        u = np.sqrt(u2)
        
        C = self.__ld[feature]

        LDMask = self.__ld['func'](u, *C)
        

        return LDMask
            
            
    def makemap(self, mode='both'):
        """
        Calculate Numerical Stellar Model
        
        rot  -- Rotation values theta /deg and phi /deg (inclination angle and 
                radial coordinate)                              (tuple of scalars)
        
        """
        if isinstance(mode, str) == False:
            raise Exception("mode must be a string")
        
        mode = mode.lower()
        
        if mode not in ['both', 'spotonly', 'faconly', 'quiet']:
            raise Exception("mode must be one of the following: 'both', 'spotonly', 'faconly', 'quiet'")
                
        RES = hp.nside2npix(self.__res)
        
        m = np.linspace(self.__dphot, self.__dphot, RES)
        
        if mode=='both' or mode=='faconly':
            for i in self.__faculae:
                if 'Facula' not in typename(i):
                    raise Exception("All elements of 'faculae' list must be an instance of 'Facula' (element type: {})".format(type(i)))
                vec = hp.ang2vec(i.lon(), i.lat(), lonlat=True) #hp.pixelfunc.ang2vec changed!
                facu = hp.query_disc(self.__res, vec, (np.pi/180)*i.r())
                m[facu] = self.__dfac
            for j in self.__fstrips:
                if 'Fac_Strip' not in typename(j):
                    raise Exception("All elements of 'fac_strips' must be an instance of 'Fac_Strip' (element type: {})".format(type(j)))
                facust = hp.query_strip(self.__res, (j.lower()+90)*(np.pi/180), (j.upper()+90)*(np.pi/180))
                m[facust] = self.__dfac
            
        if mode=='both' or mode=='spotonly':
            for i in self.__spots:
                if 'Spot' not in typename(i):
                    raise Exception("All elements of 'spots' list must be an instance of 'Spot' (element type: {})".format(type(i)))
                vec = hp.ang2vec(i.lon(), i.lat(), lonlat=True)
                spot = hp.query_disc(self.__res, vec, (np.pi/180)*i.r())
                m[spot] = self.__dspot
            for j in self.__sstrips:
                if 'Spot_Strip' not in typename(j):
                    raise Exception("All elements of 'spot_strips' must be an instance of 'Spot_Strip' (element type: {})".format(type(j)))
                spotst = hp.query_strip(self.__res, (j.lower()+90)*(np.pi/180), (j.upper()+90)*(np.pi/180))
                m[spotst] = self.__dspot
                
        
        return m
    
    def stellarmodel(self, rot=0, inc=90, mode='both', norm=False, pad=0,
                     plot=False, cmap='plasma', cbar=True, ldkey=False):
        
        rot = -rot
        inc = inc-90
        m = self.makemap(mode=mode)
        
        v2p = functools.partial(hp.vec2pix, hp.npix2nside(len(m)))

        star = hp.projector.OrthographicProj(rot=[rot, inc], half_sky=True, xsize=self.__xs).projmap(m, v2p)
                
        star[star == -np.inf] = 0
        
        if ldkey==True:
            return star
                    
            
        idx_star = star == self.__dphot
        idx_spot = star == self.__dspot
        idx_facu = star== self.__dfac
        

            

        LST = self.__photmask[idx_star]
        LSP = self.__spotmask[idx_spot]
        LFA = self.__facmask[idx_facu]



        star[idx_star] = star[idx_star]*LST/self.__dphot
        star[idx_spot] = star[idx_spot]*LSP/self.__dspot
        star[idx_facu] = star[idx_facu]*LFA/self.__dfac
            
        if norm==True:
            star /= star.max()
            
        star = np.pad(star, pad_width=pad, mode='constant', constant_values=0) #padding
            
        if plot!=False:
            plt.figure(plot)
            plt.imshow(star/star.max(), cmap=cmap)
            if cbar==True:
                plt.colorbar().ax.tick_params(labelsize=20)
            plt.xticks([])
            plt.yticks([])     
        
        return star
    
    def rotate_lc(self, inc=90, N=90, xmax=360, ret1inlist=False, mode='both',
                   synmatch=False, returndisc=False, njobs=8):
        """
        Calculates rotational lightcurve of model star, uses multithreading
        Supports several inclinations as input
        """
        if isinstance(mode, str) == False:
            raise Exception("mode must be a string")
            
        mode = mode.lower()
        
        if hasattr(inc, "__len__") == False:
            inc = [inc]
        
        m = self.makemap(mode=mode)      
        v2p = functools.partial(hp.vec2pix, hp.npix2nside(len(m)))
        
        
        x = np.linspace(0, xmax, N+1)[:-1]
        Fluxes = []
        for j in inc:
            j -= 90
            #flux = np.zeros(N)
            def multithread(xpos): #local multithreading joblib function
                star = hp.projector.OrthographicProj(rot=[xpos, j], half_sky=True, xsize=self.__xs).projmap(m, v2p)
                
                star[star == -np.inf] = 0                    
                    
                idx_star = star == self.__dphot
                idx_spot = star == self.__dspot
                idx_facu = star== self.__dfac
        
                LST = self.__photmask[idx_star]
                LSP = self.__spotmask[idx_spot]
                LFA = self.__facmask[idx_facu]
        
                star[idx_star] = star[idx_star]*LST/self.__dphot
                star[idx_spot] = star[idx_spot]*LSP/self.__dspot
                star[idx_facu] = star[idx_facu]*LFA/self.__dfac
                    
                
                if returndisc==True:
                    entry = star
                elif returndisc==False:
                    entry = np.nanmean(star)
                else:
                    raise Exception("returndisc must be True or False")
                return entry
            
            
            
            RES = jl.Parallel(n_jobs=njobs)(jl.delayed(multithread)(i) for i in x)
            
            flux = []
            for i in RES:
                flux.append(i)
                
            if returndisc==False:
                flux = np.array(flux)
            Fluxes.append(flux)
        
        
        if len(Fluxes)==1 and ret1inlist==False:
            Fluxes=Fluxes[0]

        
        if synmatch==True:
            halfN = int(N/2)
            y1 = Fluxes[:halfN]
            y2 = Fluxes[halfN:]
            
            Fluxes = np.concatenate([y2, y1])[::-1]
        
        return Fluxes
    

    def transit_lc(self, radratio=0.1, disc='static', N=101, rot=0, inc=90, b=0.0,
                         mode='both', pad='default', angle=0.0, retP=False, njobs=8, plotdisc=False):
        """
        Modelling the planetary transit
        """
        if pad=='default':
            pad = int(self.__xs/8) #default pad is 1/8 of the disc diameter

        Rp = radratio * (self.__xs/2)
        # ln = self.__xs + 2*pad
        
        xp, yp = transitpos(self.__xs, pad, -angle, -b, N)
        #-ve angle and b because stellar disc has inverted y-axis
        N = len(xp)

        MAT = self.stellarmodel(rot=rot, inc=inc, mode=mode)
        MAT = np.pad(MAT, pad_width=pad, mode='constant', constant_values=0)

        P = []
        for i in range(N):
            P.append((xp[i], yp[i]))
        
        def multithread(pos):

            mask = CAp(pos, Rp)
            planet = APh(MAT, mask)[0][3]
            flux = np.nansum(MAT) - planet
            
            return flux
            
        RES = jl.Parallel(n_jobs=njobs)(jl.delayed(multithread)(i) for i in P)
        lc = np.array(RES)
        lc /= lc.max()
            
        if retP:
            return lc, P
        elif plotdisc != False:
            if isinstance(plotdisc, str) or isinstance(plotdisc, int): 
                fig = plt.figure(plotdisc)
                ax = plt.subplot(111)
            elif 'Axes' in typename(plotdisc):
                ax = plotdisc
            else:
                raise Exception("plotdisc must be string, integer or axes object")
            plt.imshow(MAT, cmap='plasma')
            plt.xticks([])
            plt.yticks([])
            for i in P:
                circle = plt.Circle(i, Rp, color='black', alpha=0.1)
                ax.add_artist(circle)
        else:
            return lc
    
    def rotate_anim(self, inc=90, N=50, xmax=360, interval=100,
                    cmap='plasma', ylim=None, save=None, outputLC=False,
                    norm=False, backgroundLC=None, fluxunits='photon', njobs=8):
        """
        Use Numerical Stellar Model to create animation, but faster!
        """
        gs = gridspec.GridSpec(1, 2, width_ratios=[5, 6])

        fig, ax1, ax2 = plt.figure(figsize=(10, 5)), plt.subplot(gs[0]), plt.subplot(gs[1])
        
        #x = self.GetX(N=N, 'degrees')
        x1 = np.linspace(0, 1, N+1)[:-1]
        
        ax2.set_xlim(0, x1[-1])
        if ylim!=None: 
            ax2.set_ylim(ylim[0], ylim[1])

        ax1.set_xticks([])
        ax1.set_yticks([])
        ylab = r'Flux [erg/s/m$^2$]'
        if fluxunits=='photon':
            ylab = ylab.replace('erg', 'photon')
        if norm == True:
            ylab = 'Flux Variability'
        ax2.set_ylabel(ylab, fontsize=16)
        ax2.set_xlabel(r'Phase, $\phi$', fontsize=16)
        
        
        if backgroundLC is not None:
            if  hasattr(backgroundLC[0], "__len__") == False:
                backgroundLC = [backgroundLC]
            lses = ['--', ':', '-.']
            lsidx = 0
            for i in backgroundLC:
                ax2.plot(x1, i[::-1], alpha=0.5, color='black', ls=lses[lsidx], lw=2)
                lsidx += 1
            
        dat = []
        for i in [False, True]:
            d = self.rotate_lc(inc=inc, N=N, xmax=xmax, returndisc=i, njobs=njobs)
            if norm==True:
                if i==False:
                    dm = d.mean()
                d /= dm
            dat.append(d[::-1])
                
            
        ims = []
        for i in range(N):
            im = ax1.imshow(dat[1][i], animated=True, cmap=cmap)
            f = np.nanmean(dat[1][i])
            im2, = ax2.plot(x1, dat[0], color='b', lw=3, zorder=1)
            im2 = ax2.scatter(x1[i], f, color='r', s=100, alpha=1, zorder=2)
            ims.append([im, im2])

        fig.tight_layout()
        anim = ani.ArtistAnimation(fig, ims, interval=interval)
        
        if save!=None:
            anim.save(save)
            
        if outputLC==True:
            return dat[0]
        
        else:
            return anim
        
        
    def transit_anim(self, radratio=0.1, disc='static', N=101, rot=0, inc=90, b=0.0,
                         mode='both', pad=100, angle=0.0, cmap='plasma', interval=100,
                         save=None):
        """
        Animate a planetary transit and the corresponding lightcurve
        """
        
        lc, P = self.transit_lc(radratio, disc, N, rot, inc, b, mode,
                                pad, angle, retP=True)
        
        xt = []
        for i in P:
            xt.append(i[0])
        
        xt = np.array(xt)
        
        MAT = self.stellarmodel(rot=rot, inc=inc, mode=mode)
        MAT = np.pad(MAT, pad_width=pad, mode='constant', constant_values=0)
        Rp = radratio * (self.__xs/2)

        gs = gridspec.GridSpec(1, 2, width_ratios=[8, 8])

        fig, ax1, ax2 = plt.figure(), plt.subplot(gs[0]), plt.subplot(gs[1])
                
        ax2.set_xlim(xt[0], xt[-1])

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xlabel(r'Time', fontsize=16)
        ax2.set_xticks([])

        ax2.set_ylabel(r'$Flux\ Variability$', fontsize=16)
            
        ims = []
        for i in range(N):
            im = ax1.imshow(MAT, cmap=cmap)
            circle = plt.Circle(P[i], Rp, color='black', alpha=0.5)
            im = ax1.add_artist(circle)
            im2 = ax2.plot(xt, lc, color='b', lw=2, zorder=1)
            im2 = ax2.scatter(xt[i], lc[i], color='r', s=100, alpha=1, zorder=2)
            ims.append([im, im2])
        
        fig.tight_layout()
        anim = ani.ArtistAnimation(fig, ims, interval=interval)

        if save!=None:
            anim.save(save)
            
        return anim
    
    
    def build_distribution(self, preset, fspot, Q, seed=None, lonbands=None,
                           sizes='lognormal', skipN=0):
        pass
    
    
    def lightcurve_evolution(self, actressfeatures, Q, fsgrid, incs=[90, 60, 30],
                             modes='both', NLC=90, njobs=1, prnt=False,
                             fastFFs=True, discfrac=False, rethmaps=False):
        ffs = fsgrid
        
        self.setfeaturelist([])
        self.setfeaturelist([], 'spot')
        
        
        if hasattr(incs, "__len__") == False:
            incs = [incs]
        
        if modes=='all':
            modes = ['both', 'spotonly', 'faconly']
        elif isinstance(modes, list) == False:
            modes = [modes]
        
        
        LCs = []

        
        if fastFFs and discfrac==False:
            H = Hmap(self.getresolution(), [], [])
            getfill = H.GetFF
        else:
            getfill = self.getfill
        
        
        spots = actressfeatures['spot']
        facs = actressfeatures['facula']
        if 'Nc' in actressfeatures.keys():
            Nc = actressfeatures['Nc']
            Nu = Nc.sum()
        else:
            Nu = len(spots)
        
        
        facsass = facs[:Nu]
        facsun = facs[Nu:]
        
        Nf = len(facsun)
        
        assoc = False
        if len(facs)!=0:

            if 'Nc' in actressfeatures.keys(): #put in condition for clumping
                assoc='clump'
                facs = facsun
            elif facsass[-1]['lat'] == spots[-1]['lat']:
                assoc = True
                facs = facsun
        
            if prnt==True:
                print(assoc)
        
        Ns = len(spots)
        Nf = len(facs)
        N = len(incs)
        LC = {}
        lc = self.rotate_lc(N=NLC, njobs=njobs) #featureless star lightcurve
        
        for i in modes:
            lcincs = {}
            for j in range(N):
                lcincs[incs[j]] = lc
                    
                ff = 0.0
                
                if rethmaps==True:
                    lcbsf = {'LC':lcincs, 'FFspot':ff, 'FFfac':ff, 'FFboth':ff, 'hmap':H.GetMap()}
                else:
                    lcbsf = {'LC':lcincs, 'FFspot':ff, 'FFfac':ff, 'FFboth':ff}
                LC[i] = lcbsf
        
    
        
        LCs.append(LC)
        
        'spots first'
        ffsidx = 0
        f = 0
        fu = 0
        for s in range(Ns):
            if prnt==True:
                print('spots: {}/{}'.format(s+1, Ns))
            
            self.addfeature(spots[s], feature='spot')
            if fastFFs and discfrac==False:
                H.Add2Map('spot', *spots[s].values())
            if assoc == True: #single associated facula
                self.addfeature(facsass[s])
                if fastFFs and discfrac==False:
                    H.Add2Map('facula', *facsass[s].values())
                
            elif assoc == 'clump': #clump of associated faculae
                
                clumpsize = Nc[s]
                if prnt==True:
                    print(clumpsize)
                    
                for i in range(clumpsize):
                    self.addfeature(facsass[fu])
                    if fastFFs and discfrac==False:
                        H.Add2Map('facula', *facsass[fu].values())
                    fu += 1
                    if prnt==True:
                        print('clump faculae: {}/{}'.format(fu, len(facsass)))
                                            
            ffspot = getfill('spot')
            
            
            if ffspot > ffs[ffsidx]: 
    
                
                for f in range(f, Nf):
                    if prnt==True:
                        print('faculae: {}/{}'.format(f+1, Nf))
                    self.addfeature(facs[f], 'fac')
                    if fastFFs and discfrac==False:
                        H.Add2Map('facula', *facs[f].values())
                    fffac = getfill('facula')
                    if fffac > ffspot*Q:
                        f += 1
                        break
                
                LC = {}
                for i in modes:
                    lc = self.rotate_lc(inc = incs, ret1inlist=True, mode=i, 
                                    njobs=njobs, N=NLC)
                    
                    lcincs = {}
                    for j in range(N):
                        lcincs[incs[j]] = lc[j]
                    
                    spotff = getfill('spot', mode=i)
                    facff = getfill('facula', mode=i)
                    bothff = getfill('spot+fac', mode=i)
                    
                    if rethmaps==True:
                        lcbsf = {'LC':lcincs, 'FFspot':spotff, 'FFfac':facff, 'FFboth':bothff, 'hmap':H.GetMap(i)}
                    else:
                        lcbsf = {'LC':lcincs, 'FFspot':spotff, 'FFfac':facff, 'FFboth':bothff}
                    
                    LC[i] = lcbsf
                    
                
                LCs.append(LC)
                ffsidx += 1
                if ffsidx >= len(ffs): #stop sim going out of bounds
                    #print('ha')
                    break

        return LCs
    
    

