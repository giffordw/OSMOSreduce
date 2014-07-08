import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import emcee
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.stats import spearmanr
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import pdb
import pandas as pd
import time
from scipy import signal

def air_to_vacuum(airwl,nouvconv=True):
    """
    Returns vacuum wavelength of the provided air wavelength array or scalar.
    Good to ~ .0005 angstroms.

    If nouvconv is True, does nothing for air wavelength < 2000 angstroms.
    
    Input must be in angstroms.
    
    Adapted from idlutils airtovac.pro, based on the IAU standard 
    for conversion in Morton (1991 Ap.J. Suppl. 77, 119)
    """
    airwl = np.array(airwl,copy=False,dtype=float,ndmin=1)
    isscal = airwl.shape == tuple()
    if isscal:
        airwl = airwl.ravel()
    
    #wavenumber squared
    sig2 = (1e4/airwl)**2
    
    convfact = 1. + 6.4328e-5 + 2.94981e-2/(146. - sig2) +  2.5540e-4/( 41. - sig2)
    newwl = airwl.copy() 
    if nouvconv:
        convmask = newwl>=2000
        newwl[convmask] *= convfact[convmask]
    else:
        newwl[:] *= convfact
    return newwl[0] if isscal else newwl

def gaussian_lines(line_x,line_a,xgrid,width=2.0):
    '''
    Creates ideal Xenon spectrum
    '''
    #print 'Creating ideal calibration spectrum'
    temp = np.zeros(xgrid.size)
    for i in range(line_a.size):
        gauss = line_a[i]*np.exp(-(xgrid-line_x[i])**2/(2*width**2))
        temp += gauss
    return temp

def wavecalibrate(px,fx,slit_x,stretch_est=None,shift_est=None,qu_es=None):
    def prob2(x,x_p,F_p,w_m,F_m,st_es,sh_es,qu_es,interp,st_width=0.03,sh_width=75.0):
        #interp = interp1d(w_m,F_m,bounds_error=False,fill_value=0)
        new_wave = x[4]*(x_p-slit_x)**4 + x[3]*(x_p-slit_x)**3 + x[2]*(x_p-slit_x)**2+(x_p)*x[0]+x[1]
        #interp = interpolate.splrep(x_p*x[0]+x[1],F_p,s=0)
        if x[0] < st_es - st_width or x[0] > st_es + st_width: P0 = -np.inf
        else: P0 = 0.0
        if x[1] < sh_es - sh_width or x[1] > sh_es + sh_width: P1 = -np.inf
        else: P1 = 0.0
        if x[2] < -2e-5 or x[2] > 2e-5: P2 = -np.inf
        else: P2 = 0.0
        if x[3] < -1e-10 or x[3] > 1e-10: P3 = -np.inf
        else: P3 = 0.0
        if x[4] < -9e-12 or x[4] > 9e-12: P4 = -np.inf
        else: P4 = 0.0
        iwave = interp(new_wave)
        corr =  spearmanr(F_p[np.where((new_wave>=3900)&(new_wave<=6000))],iwave[np.where((new_wave>=3900)&(new_wave<=6000))])[0] + P0 + P1 + P2 + P3 + P4
        if np.isnan(corr): return -np.inf
        else: return -0.5 * (1.0 - corr) 

    #flip and normalize flux
    fx = fx - np.min(fx)
    fx = fx[::-1]
    fx = fx/signal.medfilt(fx,201)

    #prep calibration lines into 1d spectra
    wm,fm = np.loadtxt('osmos_Xenon.dat',usecols=(0,2),unpack=True)
    wm = air_to_vacuum(wm)
    xgrid = np.arange(0.0,6800.0,0.01)
    lines_gauss = gaussian_lines(wm,fm,xgrid)
    interp = interp1d(xgrid,lines_gauss,bounds_error=False,fill_value=0)

    if stretch_est is None:
        stretch_est = 0.68
        shift_est = 0.0
        qu_es = 1e-6
    
    #MCMC
    ndim,nwalkers = 5,100
    sstart = time.time()
    
    #First Pass
    p0 = np.vstack((np.random.uniform(stretch_est-0.01,stretch_est+0.01,nwalkers),np.random.uniform(-50,50,nwalkers)+shift_est,np.random.uniform(-1e-6,1e-6,nwalkers),np.random.uniform(-5e-12,5e-12,nwalkers),np.random.uniform(-5e-12,5e-12,nwalkers))).T
    sampler = emcee.EnsembleSampler(nwalkers,ndim,prob2,args=[px,fx,xgrid,lines_gauss,stretch_est,shift_est,qu_es,interp])
    print 'Stepping MCMC'
    start = time.time()
    pos, prob, state = sampler.run_mcmc(p0,100)
    end = time.time()
    print 'Burn in time:',end - start
    sampler.reset()
    print 'Starting Main MCMC'
    start = time.time()
    sampler.run_mcmc(pos,500,rstate0=state)
    end = time.time()
    print 'MCMC time:',end - start
    total_chain = sampler.flatchain
    total_lnprob = sampler.flatlnprobability
    sorted_chain = sampler.flatchain[np.argsort(sampler.flatlnprobability)[::-1]]
    max_stretch,max_shift,max_quad,max_cube,max_fourth = sorted_chain[0]
    print 'First Pass'
    print 'Max_stretch: %.4f   Max_shift: %.2f   Max_quad: %e    Max_cube: %e   Max_fourth: %e'%(max_stretch,max_shift,max_quad,max_cube,max_fourth)
    wave_new =  max_fourth*(px-slit_x)**4 + max_cube*(px-slit_x)**3 + max_quad*(px-slit_x)**2 + (px)*max_stretch + max_shift
    
    #Second Pass
    p0 = np.vstack((np.random.uniform(max_stretch-0.005,max_stretch+0.005,nwalkers),np.random.uniform(-10,10,nwalkers)+max_shift,np.random.uniform(-1e-6,1e-6,nwalkers),np.random.uniform(-5e-12,5e-12,nwalkers),np.random.uniform(-5e-12,5e-12,nwalkers))).T
    sampler = emcee.EnsembleSampler(nwalkers,ndim,prob2,args=[px,fx,xgrid,lines_gauss,max_stretch,max_shift,max_quad,interp,0.01,10.0])
    print 'Starting Main MCMC'
    start = time.time()
    sampler.run_mcmc(p0,500)
    end = time.time()
    print 'MCMC time:',end - start
    print 'Total Time:%.2f minutes'%(time.time() - sstart)
    total_chain = np.append(total_chain,sampler.flatchain,axis=0)
    total_lnprob = np.append(total_lnprob,sampler.flatlnprobability)
    sorted_chain = total_chain[np.argsort(total_lnprob)[::-1]]
    max_stretch,max_shift,max_quad,max_cube,max_fourth = sorted_chain[0]
    print 'Second Pass'
    print 'Max_stretch: %.4f   Max_shift: %.2f   Max_quad: %e    Max_cube: %e   Max_fourth: %e'%(max_stretch,max_shift,max_quad,max_cube,max_fourth)
    wave_new =  max_fourth*(px-slit_x)**4 + max_cube*(px-slit_x)**3 + max_quad*(px-slit_x)**2 + (px)*max_stretch + max_shift
    pdb.set_trace()
    '''
    #Third Pass
    p0 = np.vstack((np.random.uniform(max_stretch-0.002,max_stretch+0.002,nwalkers),np.random.uniform(-3,3,nwalkers)+max_shift,np.random.uniform(-1e-6,1e-6,nwalkers),np.random.uniform(-1e-11,1e-11,nwalkers))).T
    sampler = emcee.EnsembleSampler(nwalkers,ndim,prob2,args=[px,fx,xgrid,lines_gauss,stretch_est,shift_est,qu_es,interp,0.005,10.0])
    print 'Starting Main MCMC'
    start = time.time()
    sampler.run_mcmc(p0,1000)
    end = time.time()
    print 'MCMC time:',end - start
    total_chain = np.append(total_chain,sampler.flatchain,axis=0)
    total_lnprob = np.append(total_lnprob,sampler.flatlnprobability)
    sorted_chain = total_chain[np.argsort(total_lnprob)[::-1]]
    max_stretch,max_shift,max_quad,max_cube = sorted_chain[0]
    print 'Third Pass'
    print 'Max_stretch: %.4f   Max_shift: %.2f   Max_quad: %e   Max_cube: %e'%(max_stretch,max_shift,max_quad,max_cube)
    wave_new =  max_cube*(px-slit_x)**3 + max_quad*(px-slit_x)**2 + (px)*max_stretch + max_shift
    
    pdb.set_trace()
    '''
    return (wave_new,fx,max_cube,max_quad,max_stretch,max_shift)

def interactive_plot(px,fx,stretch_0,shift_0,quad_0,slit_x):
    #flip and normalize flux
    fx = fx - np.min(fx)
    fx = fx[::-1]

    #prep calibration lines into 1d spectra
    wm,fm = np.loadtxt('osmos_Xenon.dat',usecols=(0,2),unpack=True)
    wm = air_to_vacuum(wm)
    
    fig,ax = plt.subplots()
    plt.subplots_adjust(left=0.25,bottom=0.30)
    l, = plt.plot(quad_0*(px-slit_x)**2 + stretch_0*(px) + shift_0,fx/10.0,'b')
    plt.plot(wm,fm/2.0,'ro')
    for i in range(wm.size): plt.axvline(wm[i],color='r')
    plt.xlim(4000,6000)
    plt.ylim(0,3500)

    
    axstretch = plt.axes([0.25,0.17,0.65,0.03])
    axshift = plt.axes([0.25,0.22,0.65,0.03])
    fn_quad_0 = 0.0
    fn_stretch_0 = 0.0
    fn_shift_0 = 0.0
    fn_axquad = plt.axes([0.25,0.03,0.65,0.03])
    fn_axstretch = plt.axes([0.25,0.07,0.65,0.03])
    fn_axshift = plt.axes([0.25,0.12,0.65,0.03])
    close_ax = plt.axes([0.05,0.5,0.13,0.1])

    slide_stretch = Slider(axstretch, 'Stretch',0.4,1.3,valinit=stretch_0)
    slide_shift = Slider(axshift,'Shift',-2000.0,6000.0,valinit=shift_0)
    fn_slide_stretch = Slider(fn_axstretch, 'Fine Stretch',-0.05,0.05,valinit=fn_stretch_0)
    fn_slide_shift = Slider(fn_axshift,'Fine Shift',-200.0,200.0,valinit=fn_shift_0)
    fn_slide_quad = Slider(fn_axquad,'Fine Quad',-4e-5,4e-5,valinit=fn_quad_0)
    close_button = Button(close_ax,'Close Plots', hovercolor='0.80')

    def update(val):
        l.set_xdata((quad_0+fn_slide_quad.val)*(px-slit_x)**2+(slide_stretch.val+fn_slide_stretch.val)*(px)+(slide_shift.val+fn_slide_shift.val))
        fig.canvas.draw_idle()
    def fineupdate(val):
        l.set_xdata((quad_0+fn_slide_quad.val)*(px-slit_x)**2+(slide_stretch.val+fn_slide_stretch.val)*(px)+(slide_shift.val+fn_slide_shift.val))
        #slide_stretch.val = slide_stretch.val + fn_slide_stretch.val
        #slide_shift.val = slide_shift.val + fn_slide_shift.val
        fig.canvas.draw_idle()
    def close_plots(event):
        plt.close()
    slide_stretch.on_changed(update)
    slide_shift.on_changed(update)
    fn_slide_stretch.on_changed(fineupdate)
    fn_slide_shift.on_changed(fineupdate)
    fn_slide_quad.on_changed(fineupdate)
    close_button.on_clicked(close_plots)
    plt.show()
    shift_est = slide_shift.val+fn_slide_shift.val
    stretch_est = slide_stretch.val+fn_slide_stretch.val
    quad_est = quad_0 + fn_slide_quad.val
    print 'quad_est:',quad_est, 'stretch est:',stretch_est, 'shift est:',shift_est
    return stretch_est,shift_est,quad_est

def interactive_plot_plus(px,fx,wm,fm,stretch_0,shift_0,quad_0):
    #main plot
    fig,ax = plt.subplots()
    plt.subplots_adjust(left=0.25,bottom=0.30)
    l, = plt.plot(quad_0*(px-2032.0)**2+stretch_0*px+shift_0,fx/10.0,'b')
    plt.plot(wm,fm/2.0,'ro')
    for i in range(wm.size): plt.axvline(wm[i],color='r')
    plt.xlim(4000,6000)
    plt.ylim(0,3500)

    axstretch = plt.axes([0.25,0.17,0.65,0.03])
    axshift = plt.axes([0.25,0.22,0.65,0.03])
    fn_stretch_0 = 0.0
    fn_shift_0 = 0.0
    fn_axstretch = plt.axes([0.25,0.07,0.65,0.03])
    fn_axshift = plt.axes([0.25,0.12,0.65,0.03])
    close_ax = plt.axes([0.05,0.5,0.13,0.1])

    slide_stretch = Slider(axstretch, 'Stretch',0.4,1.3,valinit=stretch_0)
    slide_shift = Slider(axshift,'Shift',-4000.0,4000.0,valinit=shift_0)
    fn_slide_stretch = Slider(fn_axstretch, 'Fine Stretch',-0.05,0.05,valinit=fn_stretch_0)
    fn_slide_shift = Slider(fn_axshift,'Fine Shift',-200.0,200.0,valinit=fn_shift_0)
    close_button = Button(close_ax,'Close Plots', hovercolor='0.80')

    #secondary 'zoom' plots
    s = plt.figure()
    ax2 = s.add_subplot(211)
    ax3 = s.add_subplot(212)
    l2, = ax2.plot(quad_0*(px-2032.0)**2+stretch_0*px+shift_0,fx/10.0,'b')
    ax2.plot(wm,fm/2.0,'ro')
    for i in range(wm.size): ax2.axvline(wm[i],color='r')
    ax2.set_xlim(4490,4600)
    ax2.set_ylim(0,1000)
    l3, = ax3.plot(quad_0*(px-2032.0)**2+stretch_0*px+shift_0,fx/10.0,'b')
    ax3.plot(wm,fm/2.0,'ro')
    for i in range(wm.size): ax3.axvline(wm[i],color='r')
    ax3.set_xlim(4900,5100)
    ax3.set_ylim(0,1500)

    def update(val):
        l.set_xdata(quad_0*(px-2032.0)**2+(slide_stretch.val+fn_slide_stretch.val)*px+(slide_shift.val+fn_slide_shift.val))
        l2.set_xdata(quad_0*(px-2032.0)**2+(slide_stretch.val+fn_slide_stretch.val)*px+(slide_shift.val+fn_slide_shift.val))
        l3.set_xdata(quad_0*(px-2032.0)**2+(slide_stretch.val+fn_slide_stretch.val)*px+(slide_shift.val+fn_slide_shift.val))
        fig.canvas.draw_idle()
        s.canvas.draw_idle()
    def fineupdate(val):
        l.set_xdata(quad_0*(px-2032.0)**2+(slide_stretch.val+fn_slide_stretch.val)*px+(slide_shift.val+fn_slide_shift.val))
        l2.set_xdata(quad_0*(px-2032.0)**2+(slide_stretch.val+fn_slide_stretch.val)*px+(slide_shift.val+fn_slide_shift.val))
        l3.set_xdata(quad_0*(px-2032.0)**2+(slide_stretch.val+fn_slide_stretch.val)*px+(slide_shift.val+fn_slide_shift.val))
        #slide_stretch.val = slide_stretch.val + fn_slide_stretch.val
        #slide_shift.val = slide_shift.val + fn_slide_shift.val
        fig.canvas.draw_idle()
        s.canvas.draw_idle()
    def close_plots(event):
        plt.close()
        plt.close()
    slide_stretch.on_changed(update)
    slide_shift.on_changed(update)
    fn_slide_stretch.on_changed(fineupdate)
    fn_slide_shift.on_changed(fineupdate)
    close_button.on_clicked(close_plots)
    plt.show()
    shift_est = slide_shift.val+fn_slide_shift.val
    stretch_est = slide_stretch.val+fn_slide_stretch.val
    print 'quad_0:',quad_0,'stretch_0:',stretch_est,'shift_0:',shift_est
    return (quad_0*(px-2032.0)**2+px*stretch_est+shift_est,fx,stretch_est,shift_est)

if __name__ == '__main__':
    from astropy.io import fits as pyfits
    arcfits = pyfits.open('C4_0199/arcs/arc590813.0001.xo.fits')
    data = arcfits[0].data
    xpos = 500.0
    xpos2 = 1500.0
    p_x = np.arange(0,4064,1)
    f_x = np.sum(data[1670:1705,:],axis=0)
    wave,Flux,cube,quad,stretch,shift = wavecalibrate(p_x,f_x)
    #p_x2 = np.arange(0,4064,1) + 1000.0
    #wave2,Flux2,cube2,quad2,stretch2,shift2 = wavecalibrate(p_x2,f_x,stretch,shift-(xpos2*stretch-xpos*stretch),quad)



