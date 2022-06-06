#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:30:33 2020

@author: lukejohnson1
"""

# =============================================================================
# Module Imports
# =============================================================================

import numpy as np
import healpy as hp

# =============================================================================
# Map Class
# =============================================================================


class Hmap():
    """
    Class for a HealPix map
    """
    def __init__(self, res, FacList=[], SpotList=[]):
        if (res < 1) or (res > 30):
            raise Exception("resolution must be an integer between 1 and 30")
        
        self.res = res
        self.nside = 2**res
        self.npix = hp.nside2npix(self.nside)
        self.map = np.linspace(1, 1, self.npix)
        self.slist = SpotList
        self.flist = FacList
        self.smap = np.linspace(1, 1, self.npix)
        self.fmap = np.linspace(1, 1, self.npix)

        
        for i in self.slist:
            self.Add2Map('spot', *i.values(), add2list=False)
        for i in self.flist:
            self.Add2Map('facula', *i.values(), add2list=False)
        
        
    def Add2Map(self, feature, r, lon, lat, add2list=True):
        """
        Add spots and facular regions to HealPix map. 
        Spots overwrite facular regions and both overwrite the quiet photosphere.
        """
        
        fts = ['spot', 'facula']
        if feature not in fts:
            raise Exception("'feature' must be one of the following: {}".format(fts))
            
        entry = {'r':r, 'lon':lon, 'lat':lat}
        
        vec = hp.ang2vec(lon, lat, lonlat=True)
        dq = hp.query_disc(self.nside, vec, (np.pi/180)*r)
        
#        >>> np.where(a < 5, a, 10*a)
#        array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])
        
        if feature=='spot':
            self.map[dq] = 2
            self.smap[dq] = 2
            if add2list==True:
                self.slist.append(entry)
        elif feature=='facula':
            self.map[dq] = np.where(self.map[dq]==2, self.map[dq], 3)
            self.fmap[dq] = 3
            if add2list==True:
                self.flist.append(entry)
    
    def GetMap(self, mode='both'):
        
        modes = {'both':self.map, 'spotonly':self.smap, 'faconly':self.fmap}
        if mode not in modes.keys():
            raise Exception("'mode' must be one of the following: {}".format(modes.keys()))
        
        M = modes[mode]
        return M
        
    
    def GetFF(self, feature, mode='both'):
        
        d = {'photosphere':1, 'spot':2, 'facula':3, 'spot+fac':None}
        if feature not in d.keys():
            raise Exception("'feature' must be one of the following: {}".format(d.keys()))
        
        modes = {'both':self.map, 'spotonly':self.smap, 'faconly':self.fmap}
        
        if mode not in modes.keys():
            raise Exception("'mode' must be one of the following: {}".format(modes.keys()))
        
        M = modes[mode]
        
        den = len(M)

        if feature=='spot+fac':
            ff = len(M[M==2])/den
            ff += len(M[M==3])/den
        else:
            key = d[feature]
            ff = len(M[M==key])/den
        
        return ff
    
    def SetRes(self, res):
        self.res = res
        self.nside = 2**res
        self.npix = hp.nside2npix(self.nside)
        self.map = np.linspace(1, 1, self.npix)
        self.smap = np.linspace(1, 1, self.npix)
        self.fmap = np.linspace(1, 1, self.npix)
        
        for i in self.slist:
            self.Add2Map('spot', *i.values(), add2list=False)
        for i in self.flist:
            self.Add2Map('facula', *i.values(), add2list=False)
            
    def GetRes(self):
        return self.res