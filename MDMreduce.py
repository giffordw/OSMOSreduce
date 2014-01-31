'''Build like you want to distribute'''

import numpy as np
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
import scipy.signal as signal
from ds9 import *
import sys
import re
import subprocess
#import curses
import copy
import os
import fnmatch
import time
from testopt import *
import pickle
import pdb
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.stats import norm
from get_photoz import *
from redshift_estimate import *

def getch():
    import tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd,termios.TCSADRAIN,old_settings)
    return ch

def filter_image(img):
    img_sm = signal.medfilt(img,9)
    sigma = 2.0
    bad = np.abs(img-img_sm) / sigma > 8.0
    img_cr = img.copy()
    img_cr[bad] = img_sm[bad]
    return img_cr

class EstimateHK:
    def __init__(self,pspec):
        print 'Select redshift'
        #self.pspec = pspec
        #self.cid = fig.canvas.mpl_connect('button_press_event',self.onclick)
        #self.cid1 = fig.canvas.mpl_connect('key_press_event',self.on_key_press)
        #self.cid2 = fig.canvas.mpl_connect('key_release_event',self.on_key_release)
        self.cid3 = fig.canvas.mpl_connect('button_press_event',self.onclick)
        #self.shift_is_held = False

    def on_key_press(self,event):
        if event.key == 'shift':
            self.shift_is_held = True

    def on_key_release(self, event):
        if event.key == 'shift':
            self.shift_is_held = False

    def onclick(self,event):
        if event.button == 1:
            #if self.shift_is_held:
            #    print 'xdata=%f, ydata%f'%(event.xdata, event.ydata)
            #    self.lam = event.xdata
            #    plt.close()
            #else:
            plt.close()
        if event.button == 3:
            print 'xdata=%f, ydata%f'%(event.xdata, event.ydata)
            self.lam = event.xdata
            plt.close()

pixscale = 0.273 #pixel scale at for OSMOS
xbin = 1
ybin = 1
yshift = 13.0
wm,fm = np.loadtxt('osmos_Xenon.dat',usecols=(0,2),unpack=True)
wm = air_to_vacuum(wm)

###################
#Define Cluster ID#
###################
try:
    id_import = str(sys.argv[1])
    clus_id = id_import
except:
    print "Cluster Name Error: You must enter a cluster name to perform reduction"
    print ' '
    idnew = str(raw_input("Cluster ID: C4_"))
    if len(idnew) < 4:
        if len(idnew) == 3: clus_id = 'C4_0'+idnew
        if len(idnew) == 2: clus_id = 'C4_00'+idnew
        if len(idnew) == 1: clus_id == 'C4_000'+idnew
    else:
        clus_id = 'C4_'+idnew

print 'Reducing cluster: ',clus_id
###############################################################

############################
#Import Cluster .fits files#
############################
image_file = 'mosaic_r_C4_'+clus_id[-4:].lstrip('0')+'_image.fits' #define mosaic image filename

#import, clean, and add science fits files
sciencefiles = np.array([])
hdulists_science = np.array([])
for file in os.listdir('./'+clus_id+'/science/'): #search and import all science filenames
    if fnmatch.fnmatch(file, '*xo.fits'):
        sciencefiles = np.append(sciencefiles,file)
        scifits = pyfits.open(clus_id+'/science/'+file)
        hdulists_science = np.append(hdulists_science,scifits)
science_file = sciencefiles[0]
hdulist_science = pyfits.open(clus_id+'/science/'+science_file)
naxis1 = hdulist_science[0].header['NAXIS1']
naxis2 = hdulist_science[0].header['NAXIS2']

#import sky data
hdulist_sky = pyfits.open(clus_id+'/offset_sky/'+clus_id+'_offset.0001.xo.fits')

#import flat data
flatfiles = np.array([])
hdulists_flat = np.array([])
for file in os.listdir('./'+clus_id+'/flats/'): #search and import all science filenames
    if fnmatch.fnmatch(file, '*xo.fits'):
        flatfiles = np.append(flatfiles,file)
        flatfits = pyfits.open(clus_id+'/flats/'+file)
        hdulists_flat = np.append(hdulists_flat,flatfits)

#import arc data
arcfiles = np.array([])
hdulists_arc = np.array([])
for file in os.listdir('./'+clus_id+'/arcs/'): #search and import all science filenames
    if fnmatch.fnmatch(file, '*xo.fits'):
        arcfiles = np.append(arcfiles,file)
        arcfits = pyfits.open(clus_id+'/arcs/'+file)
        hdulists_arc = np.append(hdulists_arc,arcfits)
###############################################################

#########################################################
#Need to parse .oms file for ra,dec and slit information#
#########################################################
RA = np.array([])
DEC = np.array([])
SLIT_NUM = np.array([])
SLIT_WIDTH = np.array([])
SLIT_LENGTH = np.array([])
SLIT_X = np.array([])
SLIT_Y = np.array([])
for file in os.listdir('./'+clus_id+'/'):
    if fnmatch.fnmatch(file, '*.oms'):
        omsfile = file
inputfile = open(clus_id+'/'+omsfile)
alltext = inputfile.readlines()
for line in alltext:
    RAmatch = re.search('TARG(.*)\.ALPHA\s*(..)(..)(.*)',line)
    DECmatch = re.search('DELTA\s*(...)(..)(.*)',line)
    WIDmatch = re.search('WID\s\s*(.*)',line)
    LENmatch = re.search('LEN\s\s*(.*)',line)
    Xmatch = re.search('XMM\s\s*(.*)',line)
    Ymatch = re.search('YMM\s\s*(.*)',line)
    if RAmatch:
        SLIT_NUM = np.append(SLIT_NUM,RAmatch.group(1))
        RA = np.append(RA,RAmatch.group(2)+':'+RAmatch.group(3)+':'+RAmatch.group(4))
    if DECmatch:
        DEC = np.append(DEC,DECmatch.group(1)+':'+DECmatch.group(2)+':'+DECmatch.group(3))
    if WIDmatch:
        SLIT_WIDTH = np.append(SLIT_WIDTH,WIDmatch.group(1))
    if LENmatch:
        SLIT_LENGTH = np.append(SLIT_LENGTH,LENmatch.group(1))
    if Xmatch:
        SLIT_X = np.append(SLIT_X,0.5*naxis1+np.float(Xmatch.group(1))*(11.528)/(pixscale))
    if Ymatch:
        SLIT_Y = np.append(SLIT_Y,0.5*naxis2+np.float(Ymatch.group(1))*(11.528)/(pixscale)+yshift)

###############################################################

############################
#Query SDSS for galaxy data#
############################
#returns a Pandas dataframe with columns
#objID','SpecObjID','ra','dec','umag','gmag','rmag','imag','zmag','redshift','photo_z','extra'
Gal_dat = query_galaxies(RA[1:],DEC[1:])

####################
#Open images in ds9#
####################
p = subprocess.Popen('ds9 '+clus_id+'/'+image_file+' -geometry 1200x900 -scale sqrt -scale mode zscale -fits '+clus_id+'/arcs/'+arcfiles[0],shell=True)
#p = subprocess.Popen('ds9 '+clus_id+'/'+image_file+' -geometry 1200x900 -scale sqrt -scale mode zscale -fits '+clus_id+'/arcs/'+arcfiles[0],shell=True)
time.sleep(3)
print "Have the images loaded? (y/n)"
while True: #check to see if images have loaded correctly
    char = getch()
    if char.lower() in ("y", "n"):
        if char.lower() == "y":
            print 'Image has been loaded'
            break
        else:
            sys.exit('Check to make sure file '+image_file+' exists in '+clus_id+'/')

d = ds9() #start pyds9 and set parameters
d.set('frame 1')
d.set('single')
d.set('zscale contrast 9.04')
d.set('zscale bias 0.055')
d.set('zoom 2')
d.set('cmap Heat')
d.set('regions sky fk5')
#################################################################


####################################################################################
#Loop through mosaic image and decide if objects are galaxies, stars, sky, or other#
####################################################################################
reassign = 'n'
if os.path.isfile(clus_id+'/'+clus_id+'_slittypes.pkl'):
    reassign = raw_input('Detected slit types file in path. Do you wish to use this (y) or remove and re-assign slit types (n)? ')
if reassign == 'n':
    keys = np.arange(0,SLIT_WIDTH.size,1).astype('string')
    slit_type = {}
    print 'Is this a galaxy (g), a star (r), sky (s), or center (x)?'
    for i in range(SLIT_WIDTH.size):
        d.set('pan to '+RA[i]+' '+DEC[i]+' wcs fk5')
        if SLIT_WIDTH[i] == '1.0':
            d.set('regions command {box('+RA[i]+' '+DEC[i]+' 3 24) #color=green}')
        else:
            d.set('regions command {box('+RA[i]+' '+DEC[i]+' 12 12) #color=green}')
        while True:
            char = getch()
            if char.lower() in ("g", "r", "s", "x"):
                break

        slit_type[keys[i]] = char.lower()
    pickle.dump(slit_type,open(clus_id+'/'+clus_id+'_slittypes.pkl','wb'))
else:
    slit_type = pickle.load(open(clus_id+'/'+clus_id+'_slittypes.pkl','rb'))
##################################################################


d.set('frame 2')
d.set('zscale contrast 0.25')
d.set('zoom 0.40')

##################################################################
#Loop through regions and shift regions for maximum effectiveness#
##################################################################
reassign = 'n'
if os.path.isfile(clus_id+'/'+clus_id+'_slit_pos_qual.tab'):
    reassign = raw_input('Detected slit position and quality file in path. Do you wish to use this (y) or remove and re-adjust (n)? ')
if reassign == 'n':
    good_spectra = np.array([])
    FINAL_SLIT_X = np.zeros(SLIT_X.size)
    FINAL_SLIT_Y = np.zeros(SLIT_Y.size)
    SLIT_WIDTH = np.zeros(SLIT_X.size)
    print 'If needed, move region box to desired location. To increase the size, drag on corners'
    for i in range(SLIT_WIDTH.size):
        d.set('pan to 1150.0 '+str(SLIT_Y[i])+' physical')
        print 'Galaxy at ',RA[i],DEC[i]
        d.set('regions command {box(2000 '+str(SLIT_Y[i])+' 4500 40) #color=green highlite=1}')
        #raw_input('Once done: hit ENTER')
        print 'Is this spectra good (y) or bad (n)?'
        while True:
            char = getch()
            if char.lower() in ("y","n"):
                break
        good_spectra = np.append(good_spectra,char.lower())
        newpos_str = d.get('regions').split('\n')[4]
        newpos = re.search('box\(.*,(.*),.*,(.*),.*\)',newpos_str)
        FINAL_SLIT_X[i] = SLIT_X[i]
        FINAL_SLIT_Y[i] = newpos.group(1)
        SLIT_WIDTH[i] = newpos.group(2)
        print FINAL_SLIT_X[i],FINAL_SLIT_Y[i],SLIT_WIDTH[i]
        d.set('regions delete all')
    np.savetxt(clus_id+'/'+clus_id+'_slit_pos_qual.tab',np.array(zip(FINAL_SLIT_X,FINAL_SLIT_Y,SLIT_WIDTH,good_spectra),dtype=[('float',float),('float2',float),('int',int),('str','|S1')]),delimiter='\t',fmt='%10.2f %10.2f %3d %s')
else:
    FINAL_SLIT_X,FINAL_SLIT_Y,SLIT_WIDTH = np.loadtxt(clus_id+'/'+clus_id+'_slit_pos_qual.tab',dtype='float',usecols=(0,1,2),unpack=True)
    good_spectra = np.loadtxt(clus_id+'/'+clus_id+'_slit_pos_qual.tab',dtype='string',usecols=(3,),unpack=True)
####################################################################
#######################################
#Reduction steps to prep science image#
#######################################
redo = 'n'
if os.path.isfile(clus_id+'/science/'+clus_id+'_science.cr.fits'):
    redo = raw_input('Detected cosmic ray filtered file exists. Do you wish to use this (y) or remove and re-calculate (n)?')
if redo == 'n':
    scifits_c = copy.copy(hdulists_science[0]) #copy I will use to hold the smoothed and added results
    scifits_c.data *= 0.0
    print 'SCIENCE REDUCTION'
    for scifits in hdulists_science:
        filt = filter_image(scifits.data)
        scifits_c.data += filt + np.abs(np.nanmin(filt))
    scifits_c.writeto(clus_id+'/science/'+clus_id+'_science.cr.fits')
else: 
    scifits_c = pyfits.open(clus_id+'/science/'+clus_id+'_science.cr.fits')[0]
    print 'loading pre-prepared cosmic ray filtered files...'

print 'SKY REDUCTION'
if redo == 'n':
    skyfits_c = copy.copy(hdulist_sky)
    filt = filter_image(hdulist_sky[0].data)
    skyfits_c[0].data = filt + np.abs(np.nanmin(filt))
    skyfits_c.writeto(clus_id+'/offset_sky/'+clus_id+'_offset.cr.fits')
else: skyfits_c = pyfits.open(clus_id+'/offset_sky/'+clus_id+'_offset.cr.fits')

print 'FLAT REDUCTION'
if redo == 'n':
    flatfits_c = copy.copy(hdulists_flat[0]) #copy I will use to hold the smoothed and added results
    flat_data = np.zeros((hdulists_flat.size,naxis1,naxis2))
    i = 0
    for flatfits in hdulists_flat:
        filt = filter_image(flatfits.data)
        flat_data[i] = (filt+np.abs(np.nanmin(filt)))/np.max(filt+np.abs(np.nanmin(filt)))
        i += 1
    flatfits_c.data = np.median(flat_data,axis=0)
    flatfits_c.writeto(clus_id+'/flats/'+clus_id+'_flat.cr.fits')
else: flatfits_c = pyfits.open(clus_id+'/flats/'+clus_id+'_flat.cr.fits')[0]

print 'ARC REDUCTION'
if redo == 'n':
    arcfits_c = copy.copy(hdulists_arc[0]) #copy I will use to hold the smoothed and added results
    arcfits_c.data *= 0.0
    for arcfits in hdulists_arc:
        filt = filter_image(arcfits.data)
        arcfits_c.data += filt + np.abs(np.nanmin(filt))
    arcfits_c.writeto(clus_id+'/arcs/'+clus_id+'_arc.cr.fits')
else: arcfits_c = pyfits.open(clus_id+'/arcs/'+clus_id+'_arc.cr.fits')[0]

##############################
#divide science image by flat#
##############################
scifits_c2 = copy.copy(scifits_c)

#remove skies via least residual and apply master flat
sky_step = hdulists_science.size - np.linspace(-1.0,1.0,20)
total_resid = np.array([])
for ss in sky_step:
    scifits_c2.data = np.ma.masked_invalid((scifits_c.data - skyfits_c[0].data*ss) / flatfits_c.data)
    each_resid = 0
    skies = np.array(slit_type.keys())[np.where(slit_type.values()=='s')]
    for i in range(len(slit_type.values())):
        if slit_type[str(i)] == 's':
            each_resid += np.sum(np.abs(scifits_c2.data[np.int(np.floor(FINAL_SLIT_Y[i]-SLIT_WIDTH[i])):np.int(np.ceil(FINAL_SLIT_Y[i]+SLIT_WIDTH[i]))]))
    total_resid = np.append(total_resid,each_resid)
best_sub = sky_step[np.where(total_resid == np.min(total_resid))][0]
scifits_c2.data = np.ma.masked_invalid((scifits_c.data - skyfits_c[0].data*best_sub) / flatfits_c.data)
scifits_c2.data = np.ma.filled(scifits_c2.data,0.0)
if os.path.isfile(clus_id+'/science/'+clus_id+'_science.reduced.fits'):
    print 'WARNING: Overwriting pre-existing reduction file %s'%(clus_id+'/science/'+clus_id+'_science.reduced.fits')
    os.remove(clus_id+'/science/'+clus_id+'_science.reduced.fits')
scifits_c2.writeto(clus_id+'/science/'+clus_id+'_science.reduced.fits')


########################
#Wavelength Calibration#
########################
reassign = 'n'
wave = np.zeros((FINAL_SLIT_X.size-1,4064))
if os.path.isfile(clus_id+'/'+clus_id+'_stretchshift.tab'):
    reassign = raw_input('Detected file with stretch and shift parameters for each spectra. Do you wish to use this (y) or remove and re-adjust (n)? ')
if reassign == 'n':
    f = open(clus_id+'/'+clus_id+'_stretchshift.tab','w')
    f.write('#X_SLIT      Y_SLIT      SHIFT       STRETCH     WIDTH \n')
    stretch,shift = np.zeros(FINAL_SLIT_X.size-1),np.zeros(FINAL_SLIT_X.size-1)
    Flux = np.zeros((FINAL_SLIT_X.size-1,4064))
    calib_data = arcfits_c.data
    p_x = np.arange(0,4064,1)
    f_x = signal.medfilt(np.sum(calib_data[FINAL_SLIT_Y[1]-SLIT_WIDTH[1]:FINAL_SLIT_Y[1]+SLIT_WIDTH[1]/2.0,:],axis=0),5)
    d.set('pan to 1150.0 '+str(FINAL_SLIT_Y[1])+' physical')
    d.set('regions command {box(2000 '+str(FINAL_SLIT_Y[1])+' 4500 '+str(SLIT_WIDTH[1])+') #color=green highlite=1}')
    wave[0],Flux[0],stretch[0],shift[0] = wavecalibrate(p_x,f_x)
    '''
    plt.plot(wave[0],Flux[0])
    plt.plot(wm,fm/2.0,'ro')
    for j in range(wm.size):
        plt.axvline(wm[j],color='r')
    plt.xlim(4200,5200)
    plt.show()
    '''
    wave[0],Flux[0],stretch[0],shift[0] = interactive_plot_plus(p_x,f_x[::-1]-np.min(f_x),wm,fm,stretch[0],shift[0])
    f.write(str(FINAL_SLIT_X[1])+'\t')
    f.write(str(FINAL_SLIT_Y[1])+'\t')
    f.write(str(shift[0])+'\t')
    f.write(str(stretch[0])+'\t')
    f.write(str(SLIT_WIDTH[1])+'\t')
    f.write('\n')

    for i in range(stretch.size-1):
        i += 1
        p_x = np.arange(0,4064,1)
        f_x = signal.medfilt(np.sum(calib_data[FINAL_SLIT_Y[i+1]-SLIT_WIDTH[i+1]/2.0:FINAL_SLIT_Y[i+1]+SLIT_WIDTH[i+1]/2.0,:],axis=0),5)
        d.set('pan to 1150.0 '+str(FINAL_SLIT_Y[i+1])+' physical')
        d.set('regions command {box(2000 '+str(FINAL_SLIT_Y[i+1])+' 4500 '+str(SLIT_WIDTH[i+1])+') #color=green highlite=1}')
        print 'test',shift[i-1],stretch[i-1]
        wave[i],Flux[i],stretch[i],shift[i] = wavecalibrate(p_x,f_x,stretch[i-1],shift[i-1]+(FINAL_SLIT_X[i+1]*stretch[0]-FINAL_SLIT_X[i]*stretch[i-1]))
        '''
        plt.plot(wave[i],Flux[i])
        plt.plot(wm,fm/2.0,'ro')
        for j in range(wm.size):
            plt.axvline(wm[j],color='r')
        plt.xlim(4200,5200)
        plt.show()
        '''
        wave[i],Flux[i],stretch[i],shift[i] = interactive_plot_plus(p_x,f_x[::-1]-np.min(f_x),wm,fm,stretch[i],shift[i])
        f.write(str(FINAL_SLIT_X[i+1])+'\t')
        f.write(str(FINAL_SLIT_Y[i+1])+'\t')
        f.write(str(shift[i])+'\t')
        f.write(str(stretch[i])+'\t')
        f.write(str(SLIT_WIDTH[i+1])+'\t')
        f.write('\n')
        #if auto == 'n':
        #    wave,Flux,stretch,shift,auto = fitcheck(i,stretch,shift,wave,Flux)
else:
    xslit,yslit,shift,stretch,wd = np.loadtxt(clus_id+'/'+clus_id+'_stretchshift.tab',dtype='float',usecols=(0,1,2,3,4),unpack=True)
    FINAL_SLIT_X = np.append(FINAL_SLIT_X[0],xslit)
    FINAL_SLIT_Y = np.append(FINAL_SLIT_Y[0],yslit)
    SLIT_WIDTH = np.append(SLIT_WIDTH[0],wd)
    for i in range(stretch.size):
        wave[i] = stretch[i]*np.arange(0,4064,1)+shift[i]

#summed science slits + filtering to see spectra
Flux_science = np.array([signal.medfilt(np.sum(scifits_c2.data[FINAL_SLIT_Y[i+1]-SLIT_WIDTH[i+1]/2.0:FINAL_SLIT_Y[i+1]+SLIT_WIDTH[i+1]/2.0,:],axis=0)[::-1],13) for i in range(stretch.size)])

####################
#Redshift Calibrate#
####################
early_type = pyfits.open('spDR2-023.fit')
normal_type = pyfits.open('spDR2-024.fit')
normal2_type = pyfits.open('spDR2-025.fit')
coeff0 = early_type[0].header['COEFF0']
coeff1 = early_type[0].header['COEFF1']
coeff0_2 = normal_type[0].header['COEFF0']
coeff1_2 = normal_type[0].header['COEFF1']
coeff0_3 = normal2_type[0].header['COEFF0']
coeff1_3 = normal2_type[0].header['COEFF1']
early_type_flux = early_type[0].data[0]/signal.medfilt(early_type[0].data[0],171)
#early_type_flux = signal.medfilt(early_type[0].data[0],171)
normal_type_flux = normal_type[0].data[0]
normal2_type_flux = normal2_type[0].data[0]
early_type_wave = 10**(coeff0 + coeff1*np.arange(0,early_type_flux.size,1))
normal_type_wave = 10**(coeff0 + coeff1*np.arange(0,normal_type_flux.size,1))
normal2_type_wave = 10**(coeff0 + coeff1*np.arange(0,normal2_type_flux.size,1))

#flux shape correction
w2,fc = np.loadtxt('fluxcorrect.tab',dtype='float',usecols=(0,1),unpack=True)
flux_corr = interp1d(w2,fc)

import matplotlib.pyplot as plt
redshift_est = np.zeros(shift.size)
redshift_est2 = np.zeros(shift.size)
redshift_est3 = np.zeros(shift.size)
cor = np.zeros(shift.size)
cor2 = np.zeros(shift.size)
cor3 = np.zeros(shift.size)

sdss_elem = np.where(Gal_dat.redshift > 0.0)[0]
sdss_red = Gal_dat[Gal_dat.redshift > 0.0].redshift
'''
try:
    sdss_elem,sdss_red = np.loadtxt(clus_id+'/sdssred.dat',dtype='float',usecols=(0,1),unpack=True)
except:
    for i in range(RA.size-1):
        print i,RA[i+1], DEC[i+1]
    print 'Failed to load sdss spectra file', clus_id+'/sdssred.dat'
    print 'Please visit http://cas.sdss.org/dr7/en/tools/chart/list.asp and copy the above RA/DEC into the upper left box.'
    print 'Then create the file '+clus_id+'/sdssred.dat and fill with only galaxies with SDSS spectra.'
    print 'To see which objects have spectra, check the "objects with spectra" box on the left side of the window'
    print 'Remember that the element number begins with 0 in the first box, then increases left to right like a book.'
    sys.exit()
'''
for k in range(shift.size):
    '''
    if slit_type[str(k+1)] == 'g':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pspec, = ax.plot(wave[k],Flux_science[k])
        ax.axvline(3968.5*(1+np.median(sdss_red)),ls='--',alpha=0.7,c='orange')
        ax.axvline(3933.7*(1+np.median(sdss_red)),ls='--',alpha=0.7,c='orange')
        HK_est = EstimateHK(pspec)
        ax.set_xlim(3800,5500)
        plt.show()
        pre_lam_est = HK_est.lam
        pre_z_est = pre_lam_est/3950.0 - 1.0
    
    if pre_z_est < 0.0:
        pre_z_est = np.median(sdss_red)
    '''
    pre_z_est = Gal_dat.photo_z[k]

    Flux_sc = Flux_science[k]/signal.medfilt(Flux_science[k],171)

    redshift_est[k],cor[k] = redshift_estimate(pre_z_est,early_type_wave,early_type_flux,wave[k],Flux_sc)
    '''
    def redshift_est(z_est,early_type_wave,wave,Flux_sc):
        #Flux_sc = signal.medfilt(Flux_science[k],171)
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
                wave_range = wave[k][np.where((wave[k]<4800)&(wave[k]>3900))]
                Flux_range = Flux_sc[np.where((wave[k]<4800)&(wave[k]>3900))]
            else:
                wave_range = wave[k][np.where((wave[k]<4800+wavediff)&(wave[k]>3900+wavediff))]
                Flux_range = Flux_sc[np.where((wave[k]<4800+wavediff)&(wave[k]>3900+wavediff))]
            #wave_range =  wave[k][np.where((wave[k]<np.max(early_type_wave*(1+z)))&(wave[k]>np.min(w2))&(wave[k]>np.min(early_type_wave*(1+z)))&(wave[k]<np.max(w2)))]
            #Flux_range = Flux_science[k][np.where((wave[k]<np.max(early_type_wave*(1+z)))&(wave[k]>np.min(w2))&(wave[k]>np.min(early_type_wave*(1+z)))&(wave[k]<np.max(w2)))]
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
            
            s = plt.figure()
            ax = s.add_subplot(211)
            ax1 = s.add_subplot(212)
            ax.plot(wave_range,et_flux_range,'r',alpha=0.4)
            ax.plot(wave_range,Flux_range,'b',alpha=0.4)
            ax1.plot(ztest[:i+1],corr_val_i[:i+1])
            plt.show()
            

        corr_val = (corr_val_i[np.isfinite(corr_val_i)]+1)/np.trapz((corr_val_i[np.isfinite(corr_val_i)]+1),ztest[np.isfinite(corr_val_i)])
        ztest = ztest[np.isfinite(corr_val_i)]
        rv = norm(z_est,0.04)
        corr_val = corr_val * rv.pdf(ztest) 
        redshift_est[k] = (ztest[np.where((ztest>0.02)&(ztest<0.35))])[np.where(corr_val[np.where((ztest>0.02)&(ztest<0.35))] == np.max(corr_val[np.where((ztest>0.02)&(ztest<0.35))]))]
        #redshift_est2[k] = (ztest[np.where((ztest>0.05)&(ztest<0.15))])[np.where(corr_val2[np.where((ztest>0.05)&(ztest<0.15))] == np.max(corr_val2[np.where((ztest>0.05)&(ztest<0.15))]))]
        #redshift_est3[k] = (ztest[np.where((ztest>0.05)&(ztest<0.15))])[np.where(corr_val3[np.where((ztest>0.05)&(ztest<0.15))] == np.max(corr_val3[np.where((ztest>0.05)&(ztest<0.15))]))]
        cor[k] = (corr_val_i[np.where((ztest>0.02)&(ztest<0.35))])[np.where(corr_val[np.where((ztest>0.02)&(ztest<0.35))] == np.max(corr_val[np.where((ztest>0.02)&(ztest<0.35))]))]
        #cor2[k] = (corr_val2[np.where((ztest>0.05)&(ztest<0.15))])[np.where(corr_val2[np.where((ztest>0.05)&(ztest<0.15))] == np.max(corr_val2[np.where((ztest>0.05)&(ztest<0.15))]))]
        #cor3[k] = (corr_val3[np.where((ztest>0.05)&(ztest<0.15))])[np.where(corr_val3[np.where((ztest>0.05)&(ztest<0.15))] == np.max(corr_val3[np.where((ztest>0.05)&(ztest<0.15))]))]
        #plt.plot(ztest,corr_val)
        #plt.show()
    '''
    if slit_type[str(k+1)] == 'g':
        fig = plt.figure()
        ax2 = fig.add_subplot(111)
        #plt.subplots_adjust(left=0.25)
        #keep_ax = plt.axes([0.05,0.7,0.13,0.1])
        #recalc_ax = plt.axes([0.05,0.3,0.13,0.1])
        pspec, = ax2.plot(wave[k],Flux_science[k])
        ax2.axvline(3968.5*(1+redshift_est[k]),ls='--',alpha=0.7,c='orange')
        ax2.axvline(3933.7*(1+redshift_est[k]),ls='--',alpha=0.7,c='orange')
        #keep_button = Button(keep_ax,'Keep z', hovercolor='0.80')
        #recalc_button = Button(recalc_ax,'Recalc z', hovercolor='0.80')
        #def close_plots(event):
        #    plt.close()
        #def select_red(event):
        #    print 'test'
            #ax2.annotate('Click on image where HK lines look to be',xy=(0,1),xytext=(0.01,0.99),textcoords='figure fraction',horizontalalignment='left',verticalalignment='top')
        #    HK_est = EstimateHK(pspec)
        #keep_button.on_clicked(close_plots)
        #recalc_button.on_clicked(select_red)
        HK_est = EstimateHK(pspec)
        ax2.set_xlim(3800,5500)
        plt.show()
        try:
            pre_lam_est = HK_est.lam
            pre_z_est = pre_lam_est/3950.0 - 1.0
            redshift_est[k],cor[k] = redshift_estimate(pre_z_est,early_type_wave,early_type_flux,wave[k],Flux_sc)
        except:
            pass

    if k in sdss_elem.astype('int'):
        print 'Estimate: %.3f'%(redshift_est[k]), 'SDSS: %.3f'%(sdss_red.values[np.where(sdss_elem==k)][0])
    print 'z found for galaxy '+str(k+1)+' of '+str(shift.size)

plt.plot(sdss_red,redshift_est[sdss_elem.astype('int')],'ro')
#plt.plot(sdss_red,redshift_est2[sdss_elem.astype('int')],'bo')
#plt.plot(sdss_red,redshift_est3[sdss_elem.astype('int')],'o',c='purple')
plt.plot(sdss_red,sdss_red,'k')
plt.savefig(clus_id+'/redshift_compare.png')
plt.show()
'''
f = open(clus_id+'/estimated_redshifts.tab','w')
f.write('#RA    DEC    Z_est    Z_sdss\n')
for k in range(redshift_est.size):
    f.write(RA[k+1]+'\t')
    f.write(DEC[k+1]+'\t')
    f.write(str(redshift_est[k])+'\t')
    if k in sdss_elem.astype('int'):
        f.write(str(sdss_red[np.where(sdss_elem.astype('int')==k)][0])+'\t')
    else:
        f.write(str(0.000)+'\t')
    f.write('\n')
f.close()
'''

