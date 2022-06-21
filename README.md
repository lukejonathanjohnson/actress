# actress
Stellar Variability Simulator: Open source implementation - written in Python.

Simulate stellar variability due to faculae and spots as a star rotates or simulate planetary transits across the inhomogeneous stellar surface.

Any questions? Email lukejohnsonluke@gmail.com


# ===============================
# INSTALLATION (command-line)
# ===============================

Clone repository:
git clone https://github.com/lukejonathanjohnson/actress.git

Navigate to the actress directory and run setup.py:
sudo python setup.py install

# ==============================
# USAGE (Python)
# ==============================

```
import actress as ac #import actress module

sim = ac.Simulator() #create simulation instance

#define limb-darkening parameters (lists in a dictionary):
ld_dict = {'phot':[c_1, c_2, ... c_(N-1), c(N)], #photospheric coeffs
           'spot':[c_1, c_2, ... c_(N-1), c(N)], #spot coeffs
           'fac':[c_1, c_2, ... c_(N-1), c(N)],  #facular coeffs
           'func':ld_fn}       #limb-darkening fn (that takes N arguments)

"""
default limb-darkening function - ac.nonlin3:
def nonlin3(mu, I0, a, b, c):
    """
    3-parameter non-linear limb darkening fit
    """
    y = I0 * (1 - a*(1-mu) - b*(1-mu**(3/2)) - c*(1-mu**2))
    return y
"""

sim.setld(ld_dict) #update simulation ld coeffs

sim.addfeature(r, lon, lat, 'fac') #add a circular facular region with radius r [deg], longitude lon [deg] and latitude lat [deg]
sim.addfeature(r, lon, lat, 'spot') #add a circular spot with radius r [deg], longitude lon [deg] and latitude lat [deg]

"""
for all following, 
i: stellar inclination [deg] (i=90 deg = equator-on)
N: number of datapoints
mode: available modes - 'both' (spot+fac), 'faconly' (faculae only), 'spotonly' (spots only), 'quiet' (no features)
"""

lcr = sim.rotate_lc(i, N, mode) #calculate single-period rotational lightcurve
sim.rotate_anim(i, N, mode) #create animation of rotating star and resulting lightcurve (same as above)

lct = sim.transit_lc(rr, i=i, N=N, mode=mode) #calculate transit lightcurve, with planet/star radius ratio rr
sim.transit_anim(rr, i=i, N=N, mode=mode) #create animation of planetary transit and resulting lightcurve (same as above)
```




