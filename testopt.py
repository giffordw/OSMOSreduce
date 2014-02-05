import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import emcee
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.stats import spearmanr
from scipy.optimize import minimize
from scipy import interpolate
import pdb

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

def wavecalibrate(px,fx,stretch_est=None,shift_est=None,quad_est=None,parnum=2):
    def prob1(x,x_p,F_p,w_m,F_m,st_es,sh_es,qu_es):
        interp = interp1d(qu_es*x_p**2+x_p*x[0]+x[1],F_p,bounds_error=False,fill_value=0)
        #interp = interpolate.splrep(x_p*x[0]+x[1],F_p,s=0)
        if np.abs(sh_es - x[1]) > 10: P0 = -np.inf
        else: P0 = 0.0
        if np.abs(st_es - x[0]) > 0.010: P1 = -np.inf
        else: P1 = 0.0
        return np.sum(F_m[np.where((w_m>4000)&(w_m<5300))]+interp(w_m[np.where((w_m>4000)&(w_m<5300))])) + P0 + P1
        #return np.sum(interp(w_m[np.where((w_m>4300)&(w_m<5300))])/F_m[np.where((w_m>4300)&(w_m<5300))])
        #return np.sum(F_m[np.where((w_m>4300)&(w_m<5300))]+interpolate.splev(w_m[np.where((w_m>4300)&(w_m<5300))],interp,der=0))
        #return spearmanr(F_m[np.where((w_m>4300)&(w_m<5300))],interp(w_m[np.where((w_m>4300)&(w_m<5300))]))[0] + P0 + P1 #+ np.sum(F_m[np.where((w_m>4300)&(w_m<5300))]/np.max(F_m[np.where((w_m>4300)&(w_m<5300))])+interp(w_m[np.where((w_m>4300)&(w_m<5300))])/np.max(interp(w_m[np.where((w_m>4300)&(w_m<5300))])))
    def prob2(x,x_p,F_p,w_m,F_m,st_es,sh_es,qu_es):
        interp = interp1d(x[0]*x_p**2+x_p*x[1]+x[2],F_p,bounds_error=False,fill_value=0)
        #interp = interpolate.splrep(x_p*x[0]+x[1],F_p,s=0)
        if np.abs(sh_es - x[1]) > 10: P0 = -np.inf
        else: P0 = 0.0
        if np.abs(st_es - x[0]) > 0.010: P1 = -np.inf
        else: P1 = 0.0
        return np.sum(F_m[np.where((w_m>4000)&(w_m<5300))]+interp(w_m[np.where((w_m>4000)&(w_m<5300))])) + P0 + P1

    fx = fx[::-1]
    fx = fx - np.min(fx)
    wm,fm = np.loadtxt('osmos_Xenon.dat',usecols=(0,2),unpack=True)
    wm = air_to_vacuum(wm)

    if stretch_est is None or stretch_est is not None:
        if stretch_est is not None:
            stretch_0 = stretch_est
            shift_0 = shift_est
            quad_0 = quad_est
        else:
            stretch_0 = 0.68
            shift_0 = 0.0
            quad_0 = 1e-5
        print stretch_0,shift_0
        stretch_est,shift_est,quad_est = interactive_plot(px,fx,wm,fm,stretch_0,shift_0,quad_0)
    
    if parnum == 2:
        print 'Running linear fit'
        ndim,nwalkers = 2,10
        p0 = np.vstack((np.random.uniform(stretch_est-0.04,stretch_est+0.04,nwalkers),np.random.uniform(-10,10,nwalkers)+shift_est)).T
        sampler = emcee.EnsembleSampler(nwalkers,ndim,prob1,args=[px,fx,wm,fm,stretch_est,shift_est,quad_est])
        print 'Stepping MCMC'
        pos, prob, state = sampler.run_mcmc(p0,500)
        sampler.reset()
        sampler.run_mcmc(pos,3000,rstate0=state)
        (n_stretch,bins_stretch) = np.histogram(sampler.flatchain[:,0],100)
        midbins_stretch = (bins_stretch[:-1]+bins_stretch[1:])/2.0
        max_stretch = midbins_stretch[np.where(n_stretch==np.max(n_stretch))]
        (n_shift,bins_shift) = np.histogram(sampler.flatchain[:,1],100)
        midbins_shift = (bins_shift[:-1]+bins_shift[1:])/2.0
        max_shift = midbins_shift[np.where(n_shift==np.max(n_shift))]
        print 'Shift:',max_shift,'Stretch:',max_stretch

    if parnum == 3:
        print 'Running quadratic fit'
        ndim,nwalkers = 3,10
        p0 = np.vstack((np.random.uniform(1e-6,1e-5,nwalkers),np.random.uniform(stretch_est-0.005,stretch_est+0.005,nwalkers),np.random.uniform(-5,5,nwalkers)+shift_est)).T
        print 'stretch_est',stretch_est
        sampler = emcee.EnsembleSampler(nwalkers,ndim,prob2,args=[px,fx,wm,fm,stretch_est,shift_est,quad_est])
        print 'Stepping MCMC'
        pos, prob, state = sampler.run_mcmc(p0,1000)
        sampler.reset()
        sampler.run_mcmc(pos,6000,rstate0=state)
        (n_quad,bins_quad) = np.histogram(sampler.flatchain[:,0],100)
        midbins_quad = (bins_quad[:-1]+bins_quad[1:])/2.0
        max_quad = midbins_quad[np.where(n_quad==np.max(n_quad))]
        (n_stretch,bins_stretch) = np.histogram(sampler.flatchain[:,1],100)
        midbins_stretch = (bins_stretch[:-1]+bins_stretch[1:])/2.0
        max_stretch = midbins_stretch[np.where(n_stretch==np.max(n_stretch))]
        (n_shift,bins_shift) = np.histogram(sampler.flatchain[:,2],100)
        midbins_shift = (bins_shift[:-1]+bins_shift[1:])/2.0
        max_shift = midbins_shift[np.where(n_shift==np.max(n_shift))]
        plt.hist(sampler.flatchain[:,1],100)
        plt.show()
        print 'Quad:',max_quad,'Shift:',max_shift,'Stretch:',max_stretch
    
    xs = np.linspace(shift_est - 100.0,shift_est + 100.0,1000)
    xs2 = np.linspace(stretch_est - 0.05, stretch_est + 0.05,1000)
    test = np.zeros(1000)
    test2 = np.zeros(1000)
    for i in range(1000):
        interp = interp1d(px*stretch_est+xs[i],fx,bounds_error=False,fill_value=0)
        test[i] = np.sum(interp(wm[np.where((wm>4300)&(wm<5300))])+fm[np.where((wm>4300)&(wm<5300))])
        #test[i] = spearmanr(fm[np.where((wm>4300)&(wm<5300))],interp(wm[np.where((wm>4300)&(wm<5300))]))[0]
    #s = plt.figure()
    #ax = s.add_subplot(211)
    #ax.plot(xs,test)
    #ax.axvline(max_shift,color='r')
    #ax2 = s.add_subplot(212)
    #ax2.plot(xs2,test)
    #ax2.axvline(max_stretch,color='r')
    #plt.axvline(scimin['x'],color='g')
    #plt.show()
    #pdb.set_trace()
    
    #if parnum == 3:
    #    if max_quad.size > 1: max_quad = max_quad[np.floor(max_quad.size/2.0)]
    if max_stretch.size > 1: max_stretch = max_stretch[np.floor(max_stretch.size/2.0)]
    if max_shift.size > 1: max_shift = max_shift[np.floor(max_shift.size/2.0)]

    #if parnum == 2:
    #    return (px*max_stretch+max_shift,fx,max_stretch,max_shift)#(px*max_stretch+scimin['x'],fx,max_stretch,scimin['x'])
    #if parnum == 3:
    return (quad_est*px**2 + px*max_stretch + max_shift,fx,quad_est,max_stretch,max_shift)#(px*max_stretch+scimin['x'],fx,max_stretch,scimin['x'])

def interactive_plot(px,fx,wm,fm,stretch_0,shift_0,quad_0):
    fig,ax = plt.subplots()
    plt.subplots_adjust(left=0.25,bottom=0.30)
    l, = plt.plot(quad_0*px**2+stretch_0*px+shift_0,fx/10.0,'b')
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
    slide_shift = Slider(axshift,'Shift',-4000.0,4000.0,valinit=shift_0)
    fn_slide_stretch = Slider(fn_axstretch, 'Fine Stretch',-0.05,0.05,valinit=fn_stretch_0)
    fn_slide_shift = Slider(fn_axshift,'Fine Shift',-200.0,200.0,valinit=fn_shift_0)
    fn_slide_quad = Slider(fn_axquad,'Fine Quad',-2e-5,2e-5,valinit=fn_quad_0)
    close_button = Button(close_ax,'Close Plots', hovercolor='0.80')

    def update(val):
        l.set_xdata(quad_0*px**2+(slide_stretch.val+fn_slide_stretch.val)*px+(slide_shift.val+fn_slide_shift.val))
        fig.canvas.draw_idle()
    def fineupdate(val):
        l.set_xdata((quad_0+fn_slide_quad.val)*px**2+(slide_stretch.val+fn_slide_stretch.val)*px+(slide_shift.val+fn_slide_shift.val))
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
    quad_est = 1e-5 + fn_slide_quad.val
    return stretch_est,shift_est,quad_est

def interactive_plot_plus(px,fx,wm,fm,stretch_0,shift_0,quad_0):
    #main plot
    fig,ax = plt.subplots()
    plt.subplots_adjust(left=0.25,bottom=0.30)
    l, = plt.plot(quad_0*px**2+stretch_0*px+shift_0,fx/10.0,'b')
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
    l2, = ax2.plot(quad_0*px**2+stretch_0*px+shift_0,fx/10.0,'b')
    ax2.plot(wm,fm/2.0,'ro')
    for i in range(wm.size): ax2.axvline(wm[i],color='r')
    ax2.set_xlim(4490,4600)
    ax2.set_ylim(0,1000)
    l3, = ax3.plot(quad_0*px**2+stretch_0*px+shift_0,fx/10.0,'b')
    ax3.plot(wm,fm/2.0,'ro')
    for i in range(wm.size): ax3.axvline(wm[i],color='r')
    ax3.set_xlim(4900,5100)
    ax3.set_ylim(0,1500)

    def update(val):
        l.set_xdata(quad_0*px**2+(slide_stretch.val+fn_slide_stretch.val)*px+(slide_shift.val+fn_slide_shift.val))
        l2.set_xdata(quad_0*px**2+(slide_stretch.val+fn_slide_stretch.val)*px+(slide_shift.val+fn_slide_shift.val))
        l3.set_xdata(quad_0*px**2+(slide_stretch.val+fn_slide_stretch.val)*px+(slide_shift.val+fn_slide_shift.val))
        fig.canvas.draw_idle()
        s.canvas.draw_idle()
    def fineupdate(val):
        l.set_xdata(quad_0*px**2+(slide_stretch.val+fn_slide_stretch.val)*px+(slide_shift.val+fn_slide_shift.val))
        l2.set_xdata(quad_0*px**2+(slide_stretch.val+fn_slide_stretch.val)*px+(slide_shift.val+fn_slide_shift.val))
        l3.set_xdata(quad_0*px**2+(slide_stretch.val+fn_slide_stretch.val)*px+(slide_shift.val+fn_slide_shift.val))
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
    return (quad_0*px**2+px*stretch_est+shift_est,fx,stretch_est,shift_est)

if __name__ == '__main__':
    from astropy.io import fits as pyfits
    arcfits = pyfits.open('C4_0199/arcs/arc590813.0001.xo.fits')
    data = arcfits[0].data
    xpos = 500.0
    xpos2 = 1500.0
    p_x = np.arange(0,4064,1)
    f_x = np.sum(data[1670:1705,:],axis=0)
    wave,Flux,stretch,shift = wavecalibrate(p_x,f_x)
    p_x2 = np.arange(0,4064,1) + 1000.0
    wave2,Flux2,stretch2,shift2 = wavecalibrate(p_x2,f_x,stretch,shift-(xpos2*stretch-xpos*stretch))



