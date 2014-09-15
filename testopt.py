import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.lines import Line2D
from scipy.stats import spearmanr,pearsonr,kendalltau
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d,UnivariateSpline
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

def polyfour(x,a,b,c,d,e,f):
    return a + b*x + c*x**2.0 + d*x**3.0 + e*x**4.0 + f*x**5.0

def wavecalibrate(px,fx,slit_x,stretch_est=0.0,shift_est=0.0,quad_est=0.0,cube_est=0.0,fourth_est=0.0,fifth_est=0.0):
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
    #interp = UnivariateSpline(xgrid,lines_gauss)

    wave_est = fifth_est*(px-slit_x)**5 + fourth_est*(px-slit_x)**4 + cube_est*(px-slit_x)**3 + quad_est*(px-slit_x)**2 + (px-slit_x)*stretch_est + shift_est #don't subtract the slit pos because interactive plot doesn't (easier)
    wm_in = wm[np.where((wm<wave_est.max())&(wm>wave_est.min()))]
    #wm_in = wm[np.where((wm<5000.0)&(wm>wave_est.min()))]
    px_max = np.zeros(wm_in.size)
    for i in range(wm_in.size):
        px_in = px[np.where((wave_est<wm_in[i]+5.0)&(wave_est>wm_in[i]-5))]
        px_max[i] = px_in[fx[np.where((wave_est<wm_in[i]+5.0)&(wave_est>wm_in[i]-5))].argmax()]

    params,pcov = curve_fit(polyfour,(px_max-slit_x),wm_in,p0=[shift_est,stretch_est,quad_est,cube_est,fourth_est,fifth_est])
    #return (wave_new,fx,max_fourth,max_cube,max_quad,max_stretch,max_shift)
    
    return (params[0]+params[1]*(px-slit_x)+params[2]*(px-slit_x)**2+params[3]*(px-slit_x)**3.0+params[4]*(px-slit_x)**4.0+params[5]*(px-slit_x)**5.0,fx,params[5],params[4],params[3],params[2],params[1],params[0])
    #return (param0+param1*(px-slit_x)+param2*(px-slit_x)**2+param3*(px-slit_x)**3.0+param4*(px-slit_x)**4.0+param5*(px-slit_x)**5.0,fx,params[4],params[3],params[2],params[1],params[0])



def interactive_plot(px,fx,stretch_0,shift_0,quad_0,cube_0,fourth_0,fifth_0,slit_x):
    #flip and normalize flux
    fx = fx - np.min(fx)
    fx = fx[::-1]

    #prep calibration lines into 1d spectra
    wm,fm = np.loadtxt('osmos_Xenon.dat',usecols=(0,2),unpack=True)
    wm = air_to_vacuum(wm)
    
    fig,ax = plt.subplots()
    plt.subplots_adjust(left=0.25,bottom=0.30)
    l, = plt.plot(fifth_0*(px-slit_x)**5 + fourth_0*(px-slit_x)**4 + cube_0*(px-slit_x)**3 + quad_0*(px-slit_x)**2 + stretch_0*(px-slit_x) + shift_0,fx/10.0,'b')
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
        l.set_xdata((quad_0+fn_slide_quad.val)*(px-slit_x)**2+(slide_stretch.val+fn_slide_stretch.val)*(px-slit_x)+(slide_shift.val+fn_slide_shift.val))
        fig.canvas.draw_idle()
    def fineupdate(val):
        l.set_xdata((quad_0+fn_slide_quad.val)*(px-slit_x)**2+(slide_stretch.val+fn_slide_stretch.val)*(px-slit_x)+(slide_shift.val+fn_slide_shift.val))
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

class LineBrowser:
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """
    def __init__(self,fig,ax,line,wm,px,fline,line_matches):
        self.lastind = 0

        self.px = px
        self.fig = fig
        self.ax = ax
        self.wm = wm
        self.line = line
        self.fline = fline
        self.line_matches = line_matches
        self.radio_label = 'Select Line'

        self.text = ax.text(0.05, 0.95, 'Pick red reference line',transform=ax.transAxes, va='top')
        #self.selected,  = ax.plot([xs[0]], [ys[0]], 'o', ms=12, alpha=0.4,color='yellow', visible=False)
        self.selected  = self.ax.axvline(self.wm[0],lw=2,alpha=0.7,color='red', visible=False)
        self.selected_peak, = self.ax.plot(np.zeros(1),np.zeros(1),'bo',markersize=4,alpha=0.6,visible=False)

    def onpress(self, event):
        if event.key not in ('x'): return
        if event.key=='x':
            print 'added '+self.radio_label[-4:]
            self.add_line(True)
            return

    def onpick(self, event):
        if self.radio_label == 'Select Line':
            if event.artist!=self.line: return True

            N = len(event.ind)
            if not N: return True

            # the click locations
            x = event.mouseevent.xdata
            y = event.mouseevent.ydata

            distances = x-self.wm[event.ind]
            indmin = distances.argmin()
            self.dataind = event.ind[indmin]

            self.lastind = self.dataind
            self.update()
        if self.radio_label == 'Select Peak':
            if event.artist!=self.fline: return True

            N = len(event.ind)
            if not N: return True

            # the click locations
            x = event.mouseevent.xdata
            y = event.mouseevent.ydata
            
            self.chopped_ind, = np.where(np.abs(self.fline.get_xdata()-x) <= 10)
            self.max_chopped = self.chopped_ind[self.fline.get_ydata()[self.chopped_ind] == np.max(self.fline.get_ydata()[self.chopped_ind])]
            self.update()

    def update(self):
        if self.radio_label == 'Select Line':
            if self.lastind is None: return

            self.dataind = self.lastind

            #ax2.cla()
            #ax2.plot(X[dataind])

            #ax2.text(0.05, 0.9, 'mu=%1.3f\nsigma=%1.3f'%(xs[dataind], ys[dataind]),transform=ax2.transAxes, va='top')
            #ax2.set_ylim(-0.5, 1.5)
            self.selected.set_visible(True)
            self.selected.set_xdata(self.wm[self.dataind])

            self.fig.canvas.draw()
        if self.radio_label == 'Select Peak':
            self.selected_peak.set_visible(True)
            self.selected_peak.set_xdata(self.fline.get_xdata()[self.max_chopped])
            self.selected_peak.set_ydata(self.fline.get_ydata()[self.max_chopped])
            self.fig.canvas.draw()

    def add_line(self,event):
        if self.radio_label == 'Select Line':
            if self.wm[self.dataind] not in self.line_matches['lines']: #don't allow duplicates
                print 'Adding line'
                self.line_matches['lines'].append(self.wm[self.dataind])
                self.text.set_text('Pick corresponding Peak')
                self.radioset('Select Peak')
                self.fig.canvas.draw()
                return
        if self.radio_label == 'Select Peak':
            if self.px[self.max_chopped][0] not in self.line_matches['peaks']:
                print 'Adding peak'
                self.line_matches['peaks'].append(self.px[self.max_chopped][0])
                self.text.set_text('Pick red reference line')
                self.radioset('Select Line')
                self.fig.canvas.draw()
                return

    def radioset(self,label):
        self.radio_label = label

    def undo(self,event):
        if self.radio_label == 'Select Line': #then undo last peak addition
            self.line_matches['peaks'].pop()
            self.text.set_text('Re-pick corresponding Peak')
            self.radioset('Select Peak')
            self.fig.canvas.draw()
            return
        if self.radio_label == 'Select Peak': #then undo last line addition
            self.line_matches['lines'].pop()
            self.text.set_text('Re-pick red reference line')
            self.radioset('Select Line')
            self.selected.set_xdata(self.line_matches['lines'][-1])
            self.fig.canvas.draw()
            return



if __name__ == '__main__':
    from astropy.io import fits as pyfits
    wm,fm = np.loadtxt('osmos_Xenon.dat',usecols=(0,2),unpack=True)
    wm = air_to_vacuum(wm)
    arcfits = pyfits.open('C4_0199/arcs/arc590813.0001b.fits')
    data = arcfits[0].data
    xpos = 500.0
    xpos2 = 1500.0
    p_x = np.arange(0,4064,1)
    f_x = np.sum(data[1670:1705,:],axis=0)
    stretch_est,shift_est,quad_est = interactive_plot(p_x,f_x,0.70,0.0,0.0,0.0,0.0,0.0,2000)
    line_matches = {'lines':[],'peaks':[]}
    fig,ax = plt.subplots(1)
    plt.subplots_adjust(right=0.8)
    for j in range(wm.size):
        ax.axvline(wm[j],color='r')
    line, = ax.plot(wm,fm/2.0,'ro',picker=5)# 5 points tolerance
    fline, = plt.plot(quad_est*(p_x-2000)**2 + stretch_est*(p_x-2000) + shift_est,(f_x[::-1]-f_x.min())/10.0,'b',picker=5)
    closeax = plt.axes([0.83, 0.3, 0.15, 0.1])
    button = Button(closeax, 'Add Line', hovercolor='0.975')
    #rax = plt.axes([0.85, 0.5, 0.1, 0.2])
    #radio = RadioButtons(rax, ('Select Line', 'Select Peak'))
    browser = LineBrowser(fig,ax,line,wm,p_x,fline,line_matches)
    fig.canvas.mpl_connect('pick_event', browser.onpick)
    fig.canvas.mpl_connect('key_press_event',browser.onpress)
    button.on_clicked(browser.add_line)
    #radio.on_clicked(browser.radioset)
    plt.show()
    params,pcov = curve_fit(polyfour,np.sort(browser.line_matches['peaks']),np.sort(browser.line_matches['lines']),p0=[shift_est,stretch_est,quad_est,1e-8,1e-12,1e-12])
    print params
    wave,Flux,fifth,fourth,cube,quad,stretch,shift = wavecalibrate(p_x,f_x,1679.1503,0.7122818,2778.431)
    #p_x2 = np.arange(0,4064,1) + 1000.0
    #wave2,Flux2,cube2,quad2,stretch2,shift2 = wavecalibrate(p_x2,f_x,stretch,shift-(xpos2*stretch-xpos*stretch),quad)



