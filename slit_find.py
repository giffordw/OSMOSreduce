'''
Next to do: Figure out how to keep consistant slits in each slice
'''


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")


def _quadfit(x,a,b,c):
    '''define quadratic galaxy fitting function'''
    return a*(x-2032)**2 + b*(x-2032) + c

def _gaus(x,a,x0,scat,c):
    if a <= 0: a = np.inf
    return a*np.exp(-(x-x0)**2/(2*scat**2)) + c

def chip_background(pixels,flux):
    """
    Params:
    ------
    pixels (array-like): a vertical slice through the chip. Values likely range between 0 - 4064

    flux (array-like): the corresponding flux values to pixels in the vertical slice

    Returns:
    -------
    background (array-like): An array with background values with the same length as the input arrays.
                                values are binned and interpolated.
    """
    '''
    binsize = 10
    bins = np.arange(0,len(pixels)+binsize,binsize)
    
    #shifting median-filter
    medvals = []
    for i in range(len(bins)):
        if bins[i] >= 80: lower = binsize*i - 80
        else: lower = 0
        upper = binsize*i+80
        medvals.append(np.median(np.sort(flux[lower:upper])[:40]))
    I = interp1d(bins,medvals)
    return I(pixels)
    '''
    return np.median(np.sort(flux)[:10])

def identify_slits(pixels,flux,slit_y,good_detect=True):
    """
    """
    diff = flux[5:] - flux[:-5]
    diffpix = pixels[2:][:diff.size]
    maxdiff = np.max(diff[diffpix<flux.shape[0]/2.0])
    rmaxdiff = np.min(diff[diffpix>flux.shape[0]/2.0])
    start = []
    end = []
    for i in range(len(pixels)-5):
        j = i+1
        if diff[i] > maxdiff*0.80:
            if len(start) > 0:
                if pixels[j]+2 > 10+start[-1]:
                    start.append(pixels[j]+2)
                else: pass
            else: start.append(pixels[j]+2)
        elif diff[i] < rmaxdiff*0.80:
            if len(end) > 0:
                if pixels[j]+2 > 10+end[-1]:
                    end.append(pixels[j]+2)
                else: pass
            else: end.append(pixels[j]+2)
    start = np.array(start)[np.array(start) < len(pixels) - 40]
    end = np.array(end)[np.array(end)> start[0]+35]
    if len(start) > len(end):
        if slit_y < 2032:
            startf = start[:1]
        else:
            startf = start[1:]
        #else:
        #    startf = start[1:]
        endf = end
    elif len(end) > len(start):
        startf = start
        if end[0] < startf[0]:
            endf = np.array(end)[end>startf[0]+35]
        else:
            endf = end[:1]
    else:
        startf = start
        endf = end
    try:
        assert len(startf) == 1 and len(endf) == 1, 'Bad slit bounds'
    except:
        if len(startf) > len(endf) and len(endf) == 1:
            diff = np.abs(40 - (endf[0] - np.array(startf)))
            return np.array(startf)[diff == np.min(diff)],endf
        elif len(endf) > len(startf) and len(startf) == 1:
            diff = np.abs(40 - (np.array(endf) - startf[0]))
            return startf,np.array(endf)[diff == np.min(diff)]
        else:
            return [0],[0]
    if startf[0] > endf[0]:
        endf = [startf[0] + 40]

    return startf,endf


def slit_find(flux,science_flux,arc_flux):

    ##
    #Idenfity slit position as function of x
    ##
    first = []
    last = []
    pixels = np.arange(flux.shape[1])
    flux = np.log(flux)
    plt.imshow(flux - chip_background(pixels,flux),aspect=25)
    for i in range(200):
        flux2 = np.sum(flux[:,50+i*20:70+i*20],axis=1)
        pixels2 = np.arange(len(flux2))
        start,end = identify_slits(pixels2,flux2-chip_background(pixels2,flux2),300)
        first.extend(start)
        last.extend(end)
        #plt.plot(pixels,flux - chip_background(pixels,flux))
        #plt.plot(start,np.zeros(len(start)),'ro',ms=4)
        #plt.plot(end,np.zeros(len(end)),'bo',ms=4)
        #plt.show()
    xpix = np.arange(50,50+200*20,20)
    last = np.array(last)
    last = np.ma.masked_where((last<35)|(last>=flux.shape[0]),last)
    first = np.array(first)
    first = np.ma.masked_where((first<=0)|(first>=flux.shape[0]-40),first)
    plt.plot(xpix,first,'b')
    plt.plot(xpix,last,'r')
    
    
    ##
    #Fit quadratic
    ##
    #popt,pcov = curve_fit(_quadfit,xpix[:100],last[:100],p0=[1e-4,50])
    #popt2,pcov = curve_fit(_quadfit,xpix[:100],first[:100],p0=[1e-4,50])
    for i in range(3):
        popt = np.ma.polyfit(xpix[:120]-2032,last[:120],2)
        popt2 = np.ma.polyfit(xpix[:120]-2032,first[:120],2)
        first = np.ma.masked_where(np.abs(first - (popt2[0]*(xpix-2032)**2 + popt2[1]*(xpix-2032) + popt2[2])) >= 10,first)

    popt_avg = [np.average([popt2[0]]),np.average([popt2[1]]),popt2[2]]
    #plt.imshow(flux - chip_background(pixels,flux),aspect=25)
    #plt.plot(xpix,first,'b')
    #plt.plot(xpix,last,'r')
    plt.plot(xpix,_quadfit(xpix,*popt_avg),'g',lw=2)
    plt.plot(xpix,popt_avg[0]*(xpix-2032)**2 + popt_avg[1]*(xpix-2032) + popt_avg[2]+40,'g',lw=2)
    plt.show()
    
    
    ##
    #cut out slit
    ##
    d2_spectra_s = np.zeros((science_flux.shape[1],40))
    d2_spectra_a = np.zeros((arc_flux.shape[1],40))
    for i in range(science_flux.shape[1]):
        yvals = np.arange(0,science_flux.shape[0],1)
        d2_spectra_s[i] = science_flux[:,i][np.where((yvals>=popt_avg[0]*(i-2032)**2 + popt_avg[1]*(i-2032) + popt_avg[2])&(yvals<=popt_avg[0]*(i-2032)**2 + popt_avg[1]*(i-2032) + popt_avg[2]+45))][:40]
        d2_spectra_a[i] = arc_flux[:,i][np.where((yvals>=popt_avg[0]*(i-2032)**2 + popt_avg[1]*(i-2032) + popt_avg[2])&(yvals<=popt_avg[0]*(i-2032)**2 + popt_avg[1]*(i-2032) + popt_avg[2]+45))][:40]

    ##
    #Identify and cut out galaxy light
    ##
    gal_guess = np.arange(0,40,1)[np.median(d2_spectra_s.T/np.max(d2_spectra_s),axis=1)==np.max(np.median(d2_spectra_s.T/np.max(d2_spectra_s),axis=1))][0]
    popt_g,pcov_g = curve_fit(_gaus,np.arange(0,40,1),np.median(d2_spectra_s.T/np.max(d2_spectra_s),axis=1),p0=[1,gal_guess,5.0,0])
    gal_pos = popt_g[1]
    gal_wid = popt_g[2]
    if gal_wid > 5: gal_wid=5
    
    upper_gal = gal_pos + gal_wid*1.5
    lower_gal = gal_pos - gal_wid*1.5
    if upper_gal >= 40: upper_gal = 39
    if lower_gal <= 0: lower_gal = 0
    raw_gal = d2_spectra_s.T[lower_gal:upper_gal,:]
    sky = np.append(d2_spectra_s.T[:lower_gal,:],d2_spectra_s.T[upper_gal:,:],axis=0)
    sky_sub = np.zeros(raw_gal.shape) + np.median(sky,axis=0)
    sky_sub_tot = np.zeros(d2_spectra_s.T.shape) + np.median(sky,axis=0)
    
    plt.imshow(np.log(d2_spectra_s.T),aspect=35,origin='lower')
    plt.axhline(lower_gal,color='k',ls='--')
    plt.axhline(upper_gal,color='k',ls='--')
    plt.xlim(0,4064)
    plt.show()

    plt.plot(np.arange(0,40,1),_gaus(np.arange(0,40,1),*popt_g))
    plt.plot(np.arange(0,40,1),np.median(d2_spectra_s.T/np.max(d2_spectra_s),axis=1))
    plt.show()
    
    print 'gal dim:',raw_gal.shape
    print 'sky dim:',sky.shape

    plt.imshow(np.log(d2_spectra_s.T-sky_sub_tot),aspect=35,origin='lower')
    plt.show()

    plt.plot(np.arange(raw_gal.shape[1]),np.sum(raw_gal-sky_sub,axis=0)[::-1])
    plt.show()

    return d2_spectra_s.T,d2_spectra_a.T,raw_gal-sky_sub,[lower_gal,upper_gal]

if __name__ == '__main__':
    '''
    for i in range(2):
        hdu = pyfits.open('C4_0199/flats/flat590813.000'+str(i+1)+'b.fits')
        hdu2 = pyfits.open('C4_0199/science/C4_0199_science.000'+str(i+1)+'b.fits')
        hdu3 = pyfits.open('C4_0199/arcs/arc590813.000'+str(i+1)+'b.fits')
        if i == 0:
            X = slit_find(hdu[0].data[1470:1540,:],hdu2[0].data[1470:1540,:])
        else:
            X += slit_find(hdu[0].data[1470:1540,:],hdu2[0].data[1470:1540,:])
    plt.imshow(np.log(X),aspect=35)
    plt.show()
    '''

    hdu = pyfits.open('C4_0199/flats/C4_0199_flat.cr.fits')
    hdu2 = pyfits.open('C4_0199/science/C4_0199_science.cr.fits')
    hdu3 = pyfits.open('C4_0199/arcs/C4_0199_arc.cr.fits')
    X,gal,gal_bounds = slit_find(hdu[0].data[1470:1540,:],hdu2[0].data[1470:1540,:],hdu3[0].data[1470:1540,:])
    #plt.imshow(gal,aspect=35)
    #plt.show()
    #plt.plot(np.arange(gal.shape[1]),np.sum(gal,axis=0)[::-1])
    #plt.show()
