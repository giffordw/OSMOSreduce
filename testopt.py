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
from scipy.signal import argrelextrema
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
    def __init__(self,fig,ax,line,wm,fm,px,fline,xspectra,yspectra,peaks_w,peaks_p,peaks_h,line_matches,cal_states):
        #load calibration files
        self.wm_Xe,self.fm_Xe = np.loadtxt('osmos_Xenon.dat',usecols=(0,2),unpack=True)
        self.wm_Xe = air_to_vacuum(self.wm_Xe)
        self.wm_Ar,self.fm_Ar = np.loadtxt('osmos_Argon.dat',usecols=(0,2),unpack=True)
        self.wm_Ar = air_to_vacuum(self.wm_Ar)
        self.wm_HgNe,self.fm_HgNe = np.loadtxt('osmos_HgNe.dat',usecols=(0,2),unpack=True)
        self.wm_HgNe = air_to_vacuum(self.wm_HgNe)
        self.wm_Ne,self.fm_Ne = np.loadtxt('osmos_Ne.dat',usecols=(0,2),unpack=True)
        self.wm_Ne = air_to_vacuum(self.wm_Ne)

        self.lastind = 0

        self.j = 0
        self.px = px
        self.fig = fig
        self.ax = ax
        self.wm = wm
        self.line = line
        self.fline = fline
        self.xspectra = xspectra
        self.yspectra = yspectra
        self.peaks_w = peaks_w
        self.peaks_p = peaks_p
        self.peaks_h = peaks_h
        self.line_matches = line_matches
        self.cal_states = cal_states
        self.mindist_el, = np.where(self.peaks_w == self.line_matches['peaks_w'][self.j])
        
        #self.text = ax.text(0.05, 0.95, 'Pick red reference line',transform=ax.transAxes, va='top')
        #self.selected,  = ax.plot([xs[0]], [ys[0]], 'o', ms=12, alpha=0.4,color='yellow', visible=False)
        self.selected  = self.ax.axvline(self.line_matches['lines'][self.j],lw=3,alpha=0.5,color='red',ymin=0.5)
        self.selected_peak, = self.ax.plot(self.line_matches['peaks_w'][self.j],self.line_matches['peaks_h'][self.j],'o',mec='orange',markersize=8,alpha=0.7,mfc='None',mew=3,visible=True)
        self.selected_peak_line = self.ax.axvline(self.line_matches['peaks_w'][self.j],color='cyan',lw=4,alpha=0.3,ymax=0.5,visible=True)
        self.reset_lims()

    def update_current(self):
        if self.j >= len(self.line_matches['peaks_w']):
            print 'done with plot'
            plt.close()
            return
        self.selected_peak.set_xdata(self.line_matches['peaks_w'][self.j])
        self.selected_peak.set_ydata(self.line_matches['peaks_h'][self.j])
        self.selected.set_xdata(self.line_matches['lines'][self.j])
        self.selected_peak_line.set_xdata(self.line_matches['peaks_w'][self.j])
        
        xlim = self.ax.xaxis.get_view_interval()
        ylim = self.ax.yaxis.get_view_interval()
        if self.line_matches['lines'][self.j] > xlim[1]:
            print xlim
            print self.line_matches['lines'][self.j]
            print 'resetting axis'
            self.reset_lims()    
        self.fig.canvas.draw()
    
    def reset_lims(self):
        self.ax.set_xlim(self.line_matches['peaks_w'][self.j] - 100, self.line_matches['peaks_w'][self.j] + 500.0)
        xlims = self.ax.xaxis.get_view_interval()
        y_in = self.yspectra[np.where((self.xspectra>xlims[0])&(self.xspectra<xlims[1]))]
        self.ax.set_ylim(top=np.max(y_in)*1.1)

    def onpress(self, event):
        if event.key not in ('n','m','j','b'): return
        if event.key=='n':
            self.next_line()
        if event.key=='m':
            self.replace()
        if event.key=='j':
            self.delete()
        if event.key=='b':
            self.back_line()
        return


    def onclick(self, event):
        if event.inaxes == self.ax:
            if event.button == 1:

                # the click locations
                x = event.xdata
                y = event.ydata
        
                self.mindist_el = np.argsort(np.abs(self.peaks_w-x))[0]
                self.update_circle()

    def update_circle(self):
        self.selected_peak.set_xdata([self.peaks_w[self.mindist_el]])
        self.selected_peak.set_ydata([self.peaks_h[self.mindist_el]])
        self.fig.canvas.draw()

    def replace_b(self,event):
        self.replace()

    def replace(self):
        self.line_matches['peaks_p'][self.j] = self.peaks_p[self.mindist_el]
        self.line_matches['peaks_w'][self.j] = self.peaks_w[self.mindist_el]
        self.line_matches['peaks_h'][self.j] = self.peaks_h[self.mindist_el]
        self.next_line()
        return
    
    def back_line(self):
        if self.j >= 1:
            self.j -= 1
            self.update_current()
        else: return

    def next_go(self,event):
        self.next_line()

    def next_line(self):
        self.j += 1
        self.update_current()

    def finish(self,event):
        self.line_matches['peaks_p'] = self.line_matches['peaks_p'][:self.j]
        self.line_matches['peaks_w'] = self.line_matches['peaks_w'][:self.j]
        self.line_matches['peaks_h'] = self.line_matches['peaks_h'][:self.j]
        self.line_matches['lines'] = self.line_matches['lines'][:self.j]
        print 'FINISHED GALAXY CALIBRATION'
        plt.close()
        return

    def set_calib_lines(self,label):
        self.cal_states[label] = not self.cal_states[label]
        xl = self.ax.get_xlim()
        yl = self.ax.get_ylim()
        self.ax.cla()
        self.wm = []
        self.fm = []
        
        if self.cal_states['Xe']: 
            self.wm.extend(self.wm_Xe)
            self.fm.extend(self.fm_Xe)
        if self.cal_states['Ar']:
            self.wm.extend(self.wm_Ar)
            self.fm.extend(self.fm_Ar)
        if self.cal_states['HgNe']:
            self.wm.extend(self.wm_HgNe)
            self.fm.extend(self.fm_HgNe)
        if self.cal_states['Ne']:
            self.wm.extend(self.wm_Ne)
            self.fm.extend(self.fm_Ne)
        self.wm = np.array(self.wm)
        self.fm = np.array(self.fm)
        print self.wm
        for j in range(self.wm.size):
            self.ax.axvline(self.wm[j],color='r')
        self.line, = self.ax.plot(np.array(self.wm),np.array(self.fm)/2.0,'ro',picker=5)# 5 points tolerance
        self.selected = self.ax.axvline(self.wm[0],lw=2,alpha=0.7,color='red', visible=False)
        self.selected_peak, = self.ax.plot(np.zeros(1),np.zeros(1),'bo',markersize=4,alpha=0.6,visible=False)
        self.fline, = self.ax.plot(self.xspectra,self.yspectra,'b',picker=5)
        #self.ax.set_xlim(xl)
        #self.ax.set_ylim(yl)
        self.fig.canvas.draw()
        
    def delete_b(self,event):
        self.delete()

    def delete(self):
        self.line_matches['lines'].pop(self.j)
        self.line_matches['peaks_p'].pop(self.j)
        self.line_matches['peaks_w'].pop(self.j)
        self.line_matches['peaks_h'].pop(self.j)
        self.update_current()
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



