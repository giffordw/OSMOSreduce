import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from testopt import *
from scipy.interpolate import interp1d
import scipy.signal as signal
import emcee
import pdb
import triangle
import time

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

def downsample_spectra(x,y,N):
    '''
    This downsamples the created spectra to have the same elements as the MDM chip
    '''
    interp = interp1d(x,y)
    down_grid = np.linspace(3600.0,6500.0,N)
    down = interp(down_grid)
    return (down_grid,down)

def prob1(x,x_p,F_p,w_m,F_m,st_es,sh_es,qu_es,interp):
    #interp = interp1d(w_m,F_m,bounds_error=False,fill_value=0)
    new_wave = qu_es*(x_p-2032.0)**2+x_p*x[0]+x[1]
    #interp = interpolate.splrep(x_p*x[0]+x[1],F_p,s=0)
    if x[0] < st_es - 0.05 or x[0] > st_es + 0.05: P0 = -np.inf
    else: P0 = 0.0
    if x[1] < sh_es - 300.0 or x[1] > sh_es + 300.0: P1 = -np.inf
    else: P1 = 0.0
    #P2 = np.log(np.exp(-(x[0]-st_es)**2/(2*0.02**2)))
    #return np.sum(F_m[np.where((w_m>4000)&(w_m<5300))]+interp(w_m[np.where((w_m>4000)&(w_m<5300))])) + P0 + P1
    #return np.sum(interp(w_m[np.where((w_m>4300)&(w_m<5300))])/F_m[np.where((w_m>4300)&(w_m<5300))])
    #return np.sum(F_m[np.where((w_m>4300)&(w_m<5300))]+interpolate.splev(w_m[np.where((w_m>4300)&(w_m<5300))],interp,der=0)
    #print 'corr',spearmanr(F_p,interp(new_wave))[0] + P0 + P1,x[0],x[1]
    iwave = interp(new_wave[np.where((new_wave>4300.0)&(new_wave<5300))])
    corr =  spearmanr(F_p[np.where((new_wave>4300.0)&(new_wave<5300))],iwave)[0] + P0 + P1 #+ P2
    if np.isnan(corr): return -np.inf
    else: return -0.5 * (1.0 - corr)**2#+ np.sum(F_m[np.where((w_m>4300)&(w_m<5300))]/np.max(F_m[np.where((w_m>4300)&(w_m<5300))])+interp(w_m[np.where((w_m>4300)&(w_m<5300))])/np.max(interp(w_m[np.where((w_m>4300)&(w_m<5300))])))

def prob2(x,x_p,F_p,w_m,F_m,st_es,sh_es,qu_es,interp,st_width=0.03,sh_width=300.0):
    #interp = interp1d(w_m,F_m,bounds_error=False,fill_value=0)
    new_wave = x[3]*x_p**3 + x[2]*(x_p-2032.0)**2+x_p*x[0]+x[1]
    #interp = interpolate.splrep(x_p*x[0]+x[1],F_p,s=0)
    if x[0] < st_es - st_width or x[0] > st_es + st_width: P0 = -np.inf
    else: P0 = 0.0
    if x[1] < sh_es - sh_width or x[1] > sh_es + sh_width: P1 = -np.inf
    else: P1 = 0.0
    if x[2] < -1e-5 or x[2] > 1e-5: P2 = -np.inf
    else: P2 = 0.0
    if x[3] < -1e-11 or x[3] > 1e-11: P3 = -np.inf
    else: P3 = 0.0
    #P2 = np.log(np.exp(-(x[0]-st_es)**2/(2*0.02**2)))
    #return np.sum(F_m[np.where((w_m>4000)&(w_m<5300))]+interp(w_m[np.where((w_m>4000)&(w_m<5300))])) + P0 + P1
    #return np.sum(interp(w_m[np.where((w_m>4300)&(w_m<5300))])/F_m[np.where((w_m>4300)&(w_m<5300))])
    #return np.sum(F_m[np.where((w_m>4300)&(w_m<5300))]+interpolate.splev(w_m[np.where((w_m>4300)&(w_m<5300))],interp,der=0)
    #print 'corr',spearmanr(F_p,interp(new_wave))[0] + P0 + P1,x[0],x[1]
    iwave = interp(new_wave)
    corr =  spearmanr(F_p,iwave)[0] + P0 + P1 + P2 + P3 
    if np.isnan(corr): return -np.inf
    else: return -0.5 * (1.0 - corr)#+ np.sum(F_m[np.where((w_m>4300)&(w_m<5300))]/np.max(F_m[np.where((w_m>4300)&(w_m<5300))])+interp(w_m[np.where((w_m>4300)&(w_m<5300))])/np.max(interp(w_m[np.where((w_m>4300)&(w_m<5300))])))

sensor_size = 4064 #px
quad = True

Xenon_lines = pd.read_csv('osmos_Xenon.dat',header=None,delim_whitespace=True,names=['wavelength','element','intensity'])
xgrid = np.arange(0.0,6800.0,0.01)

lines_gauss = gaussian_lines(Xenon_lines['wavelength'].values,Xenon_lines['intensity'],xgrid)

#plt.plot(Xenon_lines[0].values,Xenon_lines[2],'ro')
plt.plot(xgrid,lines_gauss)
plt.xlim(3500,6000)
plt.show()

x_obs,flux_obs = downsample_spectra(xgrid,lines_gauss,sensor_size)

#redo gaussian_lines with larger line width
lines_gauss = gaussian_lines(Xenon_lines['wavelength'].values,Xenon_lines['intensity'],xgrid)


########################
#Wavelength Calibration#
########################
reassign = 'n'
wave = np.zeros((1,4064))
quad,stretch,shift = np.zeros(1),np.zeros(1),np.zeros(1)
Flux = np.zeros((1,4064))
p_x = np.arange(0,4064,1)
f_x = signal.medfilt(flux_obs,5)
wave[0],Flux[0],quad[0],stretch[0],shift[0] = wavecalibrate(p_x,f_x[::-1],parnum=2)
wave[0],Flux[0],stretch[0],shift[0] = interactive_plot_plus(p_x,f_x-np.min(f_x),Xenon_lines['wavelength'],Xenon_lines['intensity'],stretch[0],shift[0],quad[0])

interp = interp1d(xgrid,lines_gauss,bounds_error=False,fill_value=0)
stretch_est,shift_est,qu_es = stretch[0],shift[0],quad[0]
#stretch_est,shift_est,qu_es = 0.70,2800.0,0.0

#ndim,nwalkers = 2,100
#p0 = np.vstack((np.random.uniform(stretch_est-0.01,stretch_est+0.01,nwalkers),np.random.uniform(-50,50,nwalkers)+shift_est)).T

if quad: pass
ndim,nwalkers = 4,100
sstart = time.time()
#First Pass
p0 = np.vstack((np.random.uniform(stretch_est-0.01,stretch_est+0.01,nwalkers),np.random.uniform(-50,50,nwalkers)+shift_est,np.random.uniform(-1e-6,1e-6,nwalkers),np.random.uniform(-5e-12,5e-12,nwalkers))).T
sampler = emcee.EnsembleSampler(nwalkers,ndim,prob2,args=[p_x,f_x,xgrid,lines_gauss,stretch_est,shift_est,qu_es,interp])
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
max_stretch,max_shift,max_quad,max_cube = sorted_chain[0]
print 'First Pass'
print 'Max_stretch: %.4f   Max_shift: %.2f   Max_quad: %e    Max_cude: %e'%(max_stretch,max_shift,max_quad,max_cube)
wave_new =  max_cube*p_x**3 + max_quad*(p_x-2032.0)**2+p_x*max_stretch + max_shift
print wave_new - x_obs

#Second Pass
p0 = np.vstack((np.random.uniform(max_stretch-0.005,max_stretch+0.005,nwalkers),np.random.uniform(-10,10,nwalkers)+max_shift,np.random.uniform(-1e-6,1e-6,nwalkers),np.random.uniform(-5e-12,5e-12,nwalkers))).T
sampler = emcee.EnsembleSampler(nwalkers,ndim,prob2,args=[p_x,f_x,xgrid,lines_gauss,max_stretch,max_shift,max_quad,interp,0.01,10.0])
print 'Starting Main MCMC'
start = time.time()
sampler.run_mcmc(p0,500)
end = time.time()
print 'MCMC time:',end - start
print 'Total Time:%.2f minutes'%(time.time() - sstart)

total_chain = np.append(total_chain,sampler.flatchain,axis=0)
total_lnprob = np.append(total_lnprob,sampler.flatlnprobability)
sorted_chain = total_chain[np.argsort(total_lnprob)[::-1]]
max_stretch,max_shift,max_quad,max_cube = sorted_chain[0]
print 'Second Pass'
print 'Max_stretch: %.4f   Max_shift: %.2f   Max_quad: %e    Max_cude: %e'%(max_stretch,max_shift,max_quad,max_cube)
wave_new =  max_cube*p_x**3 + max_quad*(p_x-2032.0)**2+p_x*max_stretch + max_shift
print wave_new - x_obs

#Third Pass
p0 = np.vstack((np.random.uniform(max_stretch-0.002,max_stretch+0.002,nwalkers),np.random.uniform(-3,3,nwalkers)+max_shift,np.random.uniform(-1e-6,1e-6,nwalkers),np.random.uniform(-1e-11,1e-11,nwalkers))).T
sampler = emcee.EnsembleSampler(nwalkers,ndim,prob2,args=[p_x,f_x,xgrid,lines_gauss,stretch_est,shift_est,qu_es,interp,0.005,10.0])
print 'Starting Main MCMC'
start = time.time()
sampler.run_mcmc(p0,500)
end = time.time()
print 'MCMC time:',end - start

total_chain = np.append(total_chain,sampler.flatchain,axis=0)
total_lnprob = np.append(total_lnprob,sampler.flatlnprobability)
sorted_chain = total_chain[np.argsort(total_lnprob)[::-1]]
max_stretch,max_shift,max_quad,max_cube = sorted_chain[0]
print 'Third Pass'
print 'Max_stretch: %.4f   Max_shift: %.2f   Max_quad: %e   Max_cube: %e'%(max_stretch,max_shift,max_quad,max_cube)
wave_new =  max_cube*p_x**3 + max_quad*(p_x-2032.0)**2+p_x*max_stretch + max_shift
print wave_new - x_obs

print 'Acceptance Fraction:',np.average(sampler.acceptance_fraction)
'''
figure = triangle.corner(sampler.flatchain)
figure.show()
'''
