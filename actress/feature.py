#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 09:45:51 2020

@author: lukejohnson1
"""

# =============================================================================
# Feature Class
# =============================================================================

class Feature():
    """
    Stellar surface feature
    """
    def __init__(self, radius, longitude, latitude):
        
        self.__r = float(radius)
        self.__ln = float(longitude)
        self.__lt = float(latitude)
    
    def r(self):
        """
        Returns feature radius
        """
        return self.__r
    
    def setr(self, new_radius):
        """
        Updates feature radius
        """
        self.__r = new_radius
        
    def lon(self):
        """
        Returns feature longitude
        """
        return self.__ln
    
    def setlon(self, new_longitude):
        """
        Updates feature longitude
        """
        self.__ln = new_longitude

    def lat(self):
        """
        Returns feature latitude
        """
        return self.__lt
    
    def setlat(self, new_latitude):
        """
        Updates feature latitude
        """
        self.__lt = new_latitude

class Strip():
    """
    Stellar surface strip
    """
    def __init__(self, lower, upper, lon_or_lat='lat'):
        self.__lower = lower
        self.__upper = upper
        self.__lonlat = lon_or_lat.lower()
        
    def lower(self):
        return self.__lower
    
    def setlower(self, newlower):
        self.__lower = newlower
    
    def upper(self):
        return self.__upper

    def setupper(self, newupper):
        self.__upper = newupper

    def lonlat(self):
        return self.__lonlat
    
    def setlonlat(self, newlonlat):
        self.__lonlat = newlonlat.lower()
    
    
        
        

# =============================================================================
# Spot and Facula classes
# =============================================================================

class Spot(Feature):
    def __repr__(self):
        return "Starspot with radius {} deg, longitude {} deg and latitude {} deg".format(self.r(), self.lon(), self.lat())
        
class Facula(Feature):
    def __repr__(self):
        return "Facular region with radius {} deg, longitude {} deg and latitude {} deg".format(self.r(), self.lon(), self.lat())
        
class Spot_Strip(Strip):
    def __repr__(self):
        if self.lonlat()=='lat':
            bandtype = 'Latitudinal'
        elif self.lonlat()=='lon':
            bandtype = 'Longitudinal'
        return "{} spot band between {} and {} deg".format(bandtype, self.lower(), self.upper())
    
class Fac_Strip(Strip):
    def __repr__(self):
        if self.lonlat()=='lat':
            bandtype = 'Latitudinal'
        elif self.lonlat()=='lon':
            bandtype = 'Longitudinal'
        return "{} facular band between {} and {} deg".format(bandtype, self.lower(), self.upper())

