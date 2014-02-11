import numpy as np
import pdb
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.stats import norm

def redshift_estimate(z_est,early_type_wave,early_type_flux,wave,Flux_sc):
    #Flux_sc = signal.medfilt(Flux_science[],171)
    ztest = np.linspace(0.02,0.35,5000)
    corr_val_i = np.zeros(ztest.size)
    for i in range(ztest.size):
        ztest = np.linspace(0.02,0.35,5000)
        z = ztest[i]
        wshift = early_type_wave*(1+z)
        #wshift2 = normal_type_wave*(1+z)
        #wshift3 = normal2_type_wave*(1+z)
        wavediff = np.min(wshift - 3900)
        #wavediff2 = np.min(wshift2 - 3900)
        #wavediff3 = np.min(wshift3 - 3900)
        if wavediff < 0:
            wave_range = wave[np.where((wave<4800)&(wave>3900))]
            Flux_range = Flux_sc[np.where((wave<4800)&(wave>3900))]
        else:
            wave_range = wave[np.where((wave<4800+wavediff)&(wave>3900+wavediff))]
            Flux_range = Flux_sc[np.where((wave<4800+wavediff)&(wave>3900+wavediff))]
        #wave_range =  wave[][np.where((wave[]<np.max(early_type_wave*(1+z)))&(wave[]>np.min(w2))&(wave[]>np.min(early_type_wave*(1+z)))&(wave[]<np.max(w2)))]
        #Flux_range = Flux_science[][np.where((wave[]<np.max(early_type_wave*(1+z)))&(wave[]>np.min(w2))&(wave[]>np.min(early_type_wave*(1+z)))&(wave[]<np.max(w2)))]
        #Flux_range_corr = flux_corr(wave_range)
        inter = interp1d(wshift,early_type_flux)
        #inter2 = interp1d(wshift2,normal_type_flux)
        #inter3 = interp1d(wshift2,normal2_type_flux)
        et_flux_range = inter(wave_range)
        #nt_flux_range = inter2(wave_range)
        #nt2_flux_range = inter3(wave_range)
        corr_val_i[i] = pearsonr(et_flux_range,Flux_range)[0]
        #corr_val2[i] = pearsonr(nt_flux_range,Flux_range)[0]
        #corr_val3[i] = pearsonr(nt2_flux_range,Flux_range)[0]
        '''
        s = plt.figure()
        ax = s.add_subplot(211)
        ax1 = s.add_subplot(212)
        ax.plot(wave_range,et_flux_range,'r',alpha=0.4)
        ax.plot(wave_range,Flux_range,'b',alpha=0.4)
        ax1.plot(ztest[:i+1],corr_val_i[:i+1])
        plt.show()
        '''

    corr_val = (corr_val_i[np.isfinite(corr_val_i)]+1)/np.trapz((corr_val_i[np.isfinite(corr_val_i)]+1),ztest[np.isfinite(corr_val_i)])
    ztest = ztest[np.isfinite(corr_val_i)]
    rv = norm(z_est,0.04)
    corr_val = corr_val * rv.pdf(ztest)
    redshift_est = (ztest[np.where((ztest>0.02)&(ztest<0.35))])[np.where(corr_val[np.where((ztest>0.02)&(ztest<0.35))] == np.max(corr_val[np.where((ztest>0.02)&(ztest<0.35))]))]
    #redshift_est2[] = (ztest[np.where((ztest>0.05)&(ztest<0.15))])[np.where(corr_val2[np.where((ztest>0.05)&(ztest<0.15))] == np.max(corr_val2[np.where((ztest>0.05)&(ztest<0.15))]))]
    #redshift_est3[] = (ztest[np.where((ztest>0.05)&(ztest<0.15))])[np.where(corr_val3[np.where((ztest>0.05)&(ztest<0.15))] == np.max(corr_val3[np.where((ztest>0.05)&(ztest<0.15))]))]
    cor = (corr_val_i[np.where((ztest>0.02)&(ztest<0.35))])[np.where(corr_val[np.where((ztest>0.02)&(ztest<0.35))] == np.max(corr_val[np.where((ztest>0.02)&(ztest<0.35))]))]
    #cor2[] = (corr_val2[np.where((ztest>0.05)&(ztest<0.15))])[np.where(corr_val2[np.where((ztest>0.05)&(ztest<0.15))] == np.max(corr_val2[np.where((ztest>0.05)&(ztest<0.15))]))]
    #cor3[] = (corr_val3[np.where((ztest>0.05)&(ztest<0.15))])[np.where(corr_val3[np.where((ztest>0.05)&(ztest<0.15))] == np.max(corr_val3[np.where((ztest>0.05)&(ztest<0.15))]))]
    #plt.plot(ztest,corr_val)
    #plt.show()
    return redshift_est, cor
