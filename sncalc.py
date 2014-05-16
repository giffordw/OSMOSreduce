from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import *
from scipy.interpolate import interp1d

def sncalc(redshift,wave,gal_flux):
    line_halfwidth = 10 #angstroms

    Hline = 3968.5*(1+redshift) #angstroms
    Kline = 3933.7*(1+redshift) #angstroms
    Gline = 4304.0*(1+redshift) #angstroms

    HKnoise_region_min = Kline - 550
    HKnoise_region_max = Kline - 250
    Gnoise_region_max = Gline - 20
    Gnoise_region_min = Gline - 150

    HKnoise = np.std((gal_flux)[np.where((wave > HKnoise_region_min)&(wave < HKnoise_region_max))])
    Gnoise = np.std((gal_flux)[np.where((wave > Gnoise_region_min)&(wave < Gnoise_region_max))])

    Hamps = (gal_flux)[np.where((wave>Hline-line_halfwidth)&(wave<Hline+line_halfwidth))]
    Hamp = 1.0 - np.min(Hamps)
    Kamps = (gal_flux)[np.where((wave>Kline-line_halfwidth)&(wave<Kline+line_halfwidth))]
    Kamp = 1.0 - np.min(Kamps)
    Gamps = (gal_flux)[np.where((wave>Gline-line_halfwidth)&(wave<Gline+line_halfwidth))]
    Gamp = 1.0 - np.min(Gamps)

    HSN = Hamp/Gnoise
    KSN = Kamp/Gnoise
    GSN = Gamp/Gnoise

    print 'H S/N:',HSN
    print 'K S/N:',KSN
    print 'G S/N:',GSN

    return (HSN,KSN,GSN)

    
