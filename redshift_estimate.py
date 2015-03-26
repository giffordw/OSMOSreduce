import numpy as np
import pdb
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.stats import norm
import scipy.signal as signal
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, CheckButtons
import sys
from types import *

class Estimateline:
    '''Class to manually estimate where lines are located'''
    def __init__(self,pspec,ax5,uline):
        print 'If redshift calibration appears correct, hit "Accept and Close". '\
        'Otherwise, "right click" approx. where the '+uline+' line is in the plotted spectrum. '\
        'The program will re-correlate based on this guess.'
        self.ax5 = ax5
        self.cid3 = pspec.figure.canvas.mpl_connect('button_press_event',self.onclick)

    def on_key_press(self,event):
        if event.key == 'shift':
            self.shift_is_held = True

    def on_key_release(self, event):
        if event.key == 'shift':
            self.shift_is_held = False

    def onclick(self,event):
        if event.inaxes == self.ax5:
            if event.button == 3:
                print 'xdata=%f, ydata%f'%(event.xdata, event.ydata)
                self.lam = event.xdata
                plt.close()
            '''
            if event.button == 1:
                #if self.shift_is_held:
                #    print 'xdata=%f, ydata%f'%(event.xdata, event.ydata)
                #    self.lam = event.xdata
                #    plt.close()
                #else:
                plt.close()
            '''
        else: return

class z_est:
    def __init__(self,lower_w=3900.0,upper_w=5500.0,lower_z=0.01,upper_z=0.35,z_res=3.0e-5):
        '''
        Initialize redshift estimate parameters
        '''

        #preconditions
        assert lower_w and upper_w, "wavelength bounds must have values"
        assert lower_z and upper_z, "redshift bounds must have values"
        assert (type(lower_w) == IntType or type(lower_w) == FloatType) and (type(upper_w) == IntType or \
                    type(upper_w) == FloatType), "wavelength bounds must be integers or floats"
        assert (type(lower_z) == IntType or type(lower_z) == FloatType) and (type(upper_z) == IntType or \
                    type(upper_z) == FloatType), "wavelength bounds must be integers or floats"
        assert lower_w < upper_w, "lower_w must be < upper_w"
        assert lower_z < upper_z, "lower_z must be < upper_z"

        #set class attributes
        self.lower_w = lower_w
        self.upper_w = upper_w
        self.lower_z = lower_z
        self.upper_z = upper_z
        self.z_res = z_res
        
        #create redshift array and initialize correlation value array
        self.ztest = np.arange(self.lower_z,self.upper_z,self.z_res)
        self.corr_val_i = np.zeros(self.ztest.size)

        #set redshift prior flag
        self.est_pre_z = raw_input('(1) Use a known prior [Examples: median of known redshifts. Galaxy photoz measurements] \n'\
                            '(2) View spectrum and specify a redshift prior \n'\
                            '(3) No prior\n')

        #catch and correct false entry
        _est_enter = False
        self.uline_n = raw_input('What is the name of a spectral line you wish to use to identify redshift priors? '\
                                            '[Default: HK]: ')
        if not self.uline_n:
            self.uline_n = 'HK'
        self.uline = raw_input('Please list the approx. rest wavelength (in angstroms) of that line you seek to identify in your spectra '\
                                            '[Default: HK lines are at about 3950]: ')
        if self.uline:
            self.uline = np.float(self.uline)
        else:
            self.uline = 3950.0
        while not _est_enter:
            if self.est_pre_z == '1':
                self.z_prior_width = 0.06
                print 'redshift prior width has been set to',self.z_prior_width
                _est_enter = True
            elif self.est_pre_z == '2':
                self.z_prior_width = 0.06
                print 'redshift prior width has been set to',self.z_prior_width
                _est_enter = True
            elif self.est_pre_z == '3':
                self.z_prior_width = 0.06
                _est_enter = True
            else:
                self.est_pre_z = raw_input('Incorrect entry: Please enter either (1), (2), or (3).')

        #remind user to set the correct values in next step
        if self.est_pre_z == '1':
            print 'Make sure to set the gal_prior argument to the value of the known redshift prior: '\
                '[Example: z_est.redshift_estimate(gal_prior=0.1)]'

        #postconditions
        assert self.est_pre_z, "Must define redshift prior flag"
        assert self.est_pre_z == '1' or self.est_pre_z == '2' or self.est_pre_z == '3', \
                         "Incorrect string value for prior"


    def redshift_estimate(self,early_type_wave,early_type_flux,wave,Flux_science,gal_prior=None):
        '''
        estimate redshift for object
        '''
        #manage redshift prior
        self.gal_prior = gal_prior

        #continuum subtract
        Flux_sc = Flux_science - signal.medfilt(Flux_science,171)
        early_type_flux_sc = early_type_flux - signal.medfilt(early_type_flux,171)

        #handle single redshift prior flag
        if self.est_pre_z == '1':
            if self.gal_prior:
                self.pre_z_est = self.gal_prior
            else:
                nospec = raw_input('You said you are either using a spectroscopic or photometric redshift prior. '\
                                        'You need to specify a prior value! Either enter a number in now or type (q) to exit')
                if nospec == 'q':
                    sys.exit()
                elif not nospec:
                    sys.exit()
                else:
                    self.gal_prior = np.float(nospec)
                    self.pre_z_est = self.gal_prior

        #handle user prior flag
        if self.est_pre_z == '2':
            print 'Take a look at the plotted galaxy spectrum and note, approximately, at what wavelength do you see the '+self.uline_n+' line. '\
                    'Then close the plot and enter that wavelength in angstroms.'
            plt.plot(wave,Flux_science)
            plt.xlim(self.lower_w,self.upper_w)
            plt.show()
            line_init = raw_input(self.uline_n+' approx. wavelength (A): ')
            self.pre_z_est = np.float(line_init)/self.uline - 1

        #handle no prior flag
        if self.est_pre_z == '3':
            self.pre_z_est = None

        redshift_est,cor,ztest,corr_val = self._cross_cor(self.pre_z_est,self.z_prior_width,early_type_wave,early_type_flux_sc,wave,Flux_sc)

        self.qualityval = 0
        self.first_pass = True
        self._GUI_display(redshift_est,ztest,corr_val,wave,Flux_science)
        #self.line_est = Estimateline(self.pspec,ax)
        #plt.show()
        try:
            self.pre_lam_est = self.line_est.lam
            self.pre_z_est = self.pre_lam_est/3950.0 - 1.0
            self.first_pass = False
            redshift_est,cor,ztest,corr_val = self._cross_cor(self.pre_z_est,self.z_prior_width,early_type_wave,early_type_flux_sc,wave,Flux_sc)
            print 'redshift est:',redshift_est
            self._GUI_display(redshift_est,ztest,corr_val,wave,Flux_science)
            total_new_shift = self.spectra2.dx_tot
            redshift_est = (self.uline*(1+redshift_est) - total_new_shift)/self.uline - 1
        except AttributeError:
            pass
        return redshift_est,cor,ztest,corr_val,self.qualityval

    def _cross_cor(self,z_est,unc,early_type_wave,early_type_flux,wave,Flux_sc):
        '''
        This function cross-correlates a continuum subtracted template spectrum with a continuum subtracted observed spectrum.
        It then returns an estimate of the redshift, the correlation value at that redshift, the array of redshifts tested,
        and the unnormalized correlation value.
        '''
        
        #loop over each possible redshift to compute correlation values
        for i in range(self.ztest.size):
            z = self.ztest[i]
            #redshift the template wavelengths
            wshift = early_type_wave*(1+z)
            #identify the wavelength diff between the lower wave limit and the redshifted template spectrum
            wavediff = np.min(wshift - self.lower_w)

            #if the limit is above the minimum wavelength of the redshifted template spectrum...
            if wavediff < 0:
                wave_range = wave[np.where((wave<self.upper_w)&(wave>self.lower_w))]
                Flux_range = Flux_sc[np.where((wave<self.upper_w)&(wave>self.lower_w))]
            #if the limit is below the minimum wavelength of the redshifted template spectrum...
            else:
                wave_range = wave[np.where((wave<self.upper_w+wavediff)&(wave>self.lower_w+wavediff))]
                Flux_range = Flux_sc[np.where((wave<self.upper_w+wavediff)&(wave>self.lower_w+wavediff))]
            
            #interpolate the redshifted template spectrum and estimate the flux at the observed spectrum wavelengths
            inter = interp1d(wshift,early_type_flux)
            et_flux_range = inter(wave_range)

            #calculate the pearson r correlation value between the observed and template flux
            self.corr_val_i[i] = pearsonr(et_flux_range,Flux_range)[0]

        #normalize the correlation values as a function of redshift
        corr_val = (self.corr_val_i[np.isfinite(self.corr_val_i)]+1)/np.trapz((self.corr_val_i[np.isfinite(self.corr_val_i)]+1),self.ztest[np.isfinite(self.corr_val_i)])
        self.ztest = self.ztest[np.isfinite(self.corr_val_i)]
        
        #multiply in prior to likelihood if specified
        self.corr_prior = np.zeros(self.ztest.size)
        if z_est:
            rv = norm(z_est,unc)
            corr_val = corr_val * rv.pdf(self.ztest)
            self.corr_prior = rv.pdf(self.ztest)
        
        #make redshift estimate
        redshift_est = (self.ztest[np.where((self.ztest>self.lower_z)&(self.ztest<self.upper_z))])[np.where(corr_val[np.where((self.ztest>self.lower_z)&(self.ztest<self.upper_z))] == np.max(corr_val[np.where((self.ztest>self.lower_z)&(self.ztest<self.upper_z))]))]
        
        #save correlation value at maximum redshift likelihood
        cor = (self.corr_val_i[np.where((self.ztest>self.lower_z)&(self.ztest<self.upper_z))])[np.where(corr_val[np.where((self.ztest>self.lower_z)&(self.ztest<self.upper_z))] == np.max(corr_val[np.where((self.ztest>self.lower_z)&(self.ztest<self.upper_z))]))]
        
        return redshift_est[0], cor, self.ztest,corr_val

    def _GUI_display(self,redshift_est,ztest,corr_val,wave,flux_sc):
        '''Display the spectrum and reference lines.'''
        self.fig = plt.figure(figsize=(10, 8)) 
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax2 = plt.subplot(gs[0])
        ax = plt.subplot(gs[1])
        plt.subplots_adjust(right=0.8)

        ax.plot(ztest,corr_val,'b')
        ax.axvline(redshift_est,color='k',ls='--')
        ax.fill_between(self.ztest,np.zeros(self.ztest.size),self.corr_prior,facecolor='grey',alpha=0.6)
        ax.set_xlabel('Redshift')
        ax.set_ylabel('Correlation')

        self.pspec, = ax2.plot(wave,flux_sc)
        ax2.axvline(3725.0*(1+redshift_est),ls='--',alpha=0.7,c='blue')
        ax2.axvline(3968.5*(1+redshift_est),ls='--',alpha=0.7,c='red')
        ax2.axvline(3933.7*(1+redshift_est),ls='--',alpha=0.7,c='red')
        ax2.axvline(4102.9*(1+redshift_est),ls='--',alpha=0.7,c='orange')
        ax2.axvline(4304.0*(1+redshift_est),ls='--',alpha=0.7,c='orange')
        ax2.axvline(4862.0*(1+redshift_est),ls='--',alpha=0.7,c='orange')
        ax2.axvline(4959.0*(1+redshift_est),ls='--',alpha=0.7,c='blue')
        ax2.axvline(5007.0*(1+redshift_est),ls='--',alpha=0.7,c='blue')
        ax2.axvline(5175.0*(1+redshift_est),ls='--',alpha=0.7,c='orange')
        if self.first_pass:
            self.line_est = Estimateline(self.pspec,ax2,self.uline_n)
        rax = plt.axes([0.85, 0.5, 0.1, 0.2])
        if self.qualityval == 0:
            radio = RadioButtons(rax, ('Unclear', 'Clear'))
        else:
            radio = RadioButtons(rax, ('Unclear', 'Clear'),active=1)
        def qualfunc(label):
            if label == 'Clear':
                self.qualityval = 1
            else:
                self.qualityval = 0
        radio.on_clicked(qualfunc)
        closeax = plt.axes([0.83, 0.3, 0.15, 0.1])
        button = Button(closeax, 'Accept & Close', hovercolor='0.975')
        def closeplot(event):
            plt.close()
        button.on_clicked(closeplot)
        ax2.set_xlim(self.lower_w,self.upper_w)
        ax2.set_xlabel('Wavelength (A)')
        ax2.set_ylabel('Counts')
        if not self.first_pass:
            self.spectra2 = DragSpectra(self.pspec,flux_sc,ax2,self.fig)
            self.fig.canvas.mpl_connect('motion_notify_event',self.spectra2.on_motion)
            self.fig.canvas.mpl_connect('button_press_event',self.spectra2.on_press)
            self.fig.canvas.mpl_connect('button_release_event',self.spectra2.on_release)
        plt.show()

class DragSpectra:
    '''Class to drage the spectra back and forth to match lines of interest'''
    def __init__(self,spectra,ydata,ax5,fig):
        self.ax5 = ax5
        self.fig = fig
        self.tb = plt.get_current_fig_manager().toolbar
        print 'begin shift'
        self.spectra = spectra
        self.ydata = ydata
        self.pressed = False
        self.dx_tot = 0.0
        #figure.canvas.mpl_connect('motion_notify_event',self.on_motion)
        #figure.canvas.mpl_connect('button_press_event',self.on_press)
        #figure.canvas.mpl_connect('button_release_event',self.on_release)

    def on_motion(self,evt):
        if self.pressed:
            dx = evt.xdata - self.mouse_x
            self.spectra.set_data(self.spectra_x + dx,self.ydata)
            plt.draw()

    def on_press(self,evt):
        if evt.inaxes == self.ax5 and self.tb.mode == '':
            self.mouse_x = evt.xdata
            self.spectra_x = self.spectra.get_xdata()
            self.pressed = True
        else: return

    def on_release(self,evt):
        if evt.inaxes == self.ax5 and self.tb.mode == '':
            self.pressed = False
            try:
                self.dx_tot += evt.xdata - self.mouse_x
            except AttributeError:
                self.dx_tot = 0.0
        else: return

if __name__ == '__main__':
    R = z_est()
