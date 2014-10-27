'''
IMPORTANT NOTES:
In the .oms file, the first and last RA/DEC represent a reference slit at the bottom of the mask and the center of the mask respectively.

'''

import numpy as np
from astropy.io import fits as pyfits
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, CheckButtons
import scipy.signal as signal
from ds9 import *
import sys
import re
import subprocess
import pandas as pd
import copy
import os
import fnmatch
import time
from testopt import *
import pickle
import pdb
from scipy import fftpack
from get_photoz import *
from redshift_estimate import *
from sncalc import *
from redshift_checker import *
from gal_trace import *

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
    def __init__(self,pspec,ax2):
        print 'If redshift calibration appears correct, hit "Accept and Close". Otherwise, "right click" approx. where the H and K lines are in the plotted spectrum. The program will re-correlate based on this guess.'
        self.cid3 = pspec.figure.canvas.mpl_connect('button_press_event',self.onclick)

    def on_key_press(self,event):
        if event.key == 'shift':
            self.shift_is_held = True

    def on_key_release(self, event):
        if event.key == 'shift':
            self.shift_is_held = False

    def onclick(self,event):
        if event.inaxes == ax2:
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
    idnew = str(raw_input("Cluster ID: "))
    clus_id = idnew

print 'Reducing cluster: ',clus_id
###############################################################

#ask if you want to only reduce sdss galaxies with spectra
try:
    sdss_check = str(sys.argv[2])
    if sdss_check == 'sdss':
        sdss_check = True
    else:
        raise Exception(sdss_check+' is not an accepted input. \'sdss\' is the only accepted input here.')
except IndexError:
    sdss_check = False
    '''
    print "Do you want to reduce only SDSS galaxies with spectra? (y/n)"
    while True: 
        char = getch()
        if char.lower() in ("y", "n"):
            if char.lower() == "y":
                print 'Reducing only galaxies with previous SDSS spectra'
                sdss_check = True
                break
            else:
                sdss_check = False
                print 'Reducing all galaxies'
    '''

############################
#Import Cluster .fits files#
############################
for file in os.listdir('./'+clus_id+'/'): #search and import all mosaics
    if fnmatch.fnmatch(file, 'mosaic_*'):
        image_file = file

#create reduced files if they don't exist
def reduce_files(filetype):
    for file in os.listdir('./'+clus_id+'/'+filetype+'/'):
        if fnmatch.fnmatch(file, '*.????.fits'):
            if not os.path.isfile(clus_id+'/'+filetype+'/'+file[:-5]+'b.fits'):
                print 'Creating '+clus_id+'/'+filetype+'/'+file[:-5]+'b.fits'
                p = subprocess.Popen('python proc4k.py '+clus_id+'/'+filetype+'/'+file,shell=True)
                p.wait()
            else:
                print 'Reduced '+filetype+' files exist'

filetypes = ['science','arcs','flats','offset_sky']
for filetype in filetypes:
    reduce_files(filetype)

#import, clean, and add science fits files
sciencefiles = np.array([])
hdulists_science = np.array([])
for file in os.listdir('./'+clus_id+'/science/'): #search and import all science filenames
    if fnmatch.fnmatch(file, '*b.fits'):
        sciencefiles = np.append(sciencefiles,file)
        scifits = pyfits.open(clus_id+'/science/'+file)
        hdulists_science = np.append(hdulists_science,scifits)
science_file = sciencefiles[0]
hdulist_science = pyfits.open(clus_id+'/science/'+science_file)
naxis1 = hdulist_science[0].header['NAXIS1']
naxis2 = hdulist_science[0].header['NAXIS2']

#import sky data
for file in os.listdir('./'+clus_id+'/offset_sky/'):
    if fnmatch.fnmatch(file, '*0001b.fits'):
        hdulist_sky = pyfits.open(clus_id+'/offset_sky/'+file)
try: test = hdulist_sky
except: raise Exception('proc4k.py did not detect any offset sky files')

#import flat data
flatfiles = np.array([])
hdulists_flat = np.array([])
for file in os.listdir('./'+clus_id+'/flats/'): #search and import all science filenames
    if fnmatch.fnmatch(file, '*b.fits'):
        flatfiles = np.append(flatfiles,file)
        flatfits = pyfits.open(clus_id+'/flats/'+file)
        hdulists_flat = np.append(hdulists_flat,flatfits)
if len(hdulists_flat) < 1:
    raise Exception('proc4k.py did not detect any flat files')

#import arc data
arcfiles = np.array([])
hdulists_arc = np.array([])
for file in os.listdir('./'+clus_id+'/arcs/'): #search and import all science filenames
    if fnmatch.fnmatch(file, '*b.fits'):
        arcfiles = np.append(arcfiles,file)
        arcfits = pyfits.open(clus_id+'/arcs/'+file)
        hdulists_arc = np.append(hdulists_arc,arcfits)
if len(hdulists_arc) < 1:
    raise Exception('proc4k.py did not detect any arc files')
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

#remove throw away rows and dump into Gal_dat dataframe
Gal_dat = pd.DataFrame({'RA':RA[1:SLIT_WIDTH.size],'DEC':DEC[1:SLIT_WIDTH.size],'SLIT_WIDTH':SLIT_WIDTH[1:],'SLIT_LENGTH':SLIT_LENGTH[1:],'SLIT_X':SLIT_X[1:],'SLIT_Y':SLIT_Y[1:]})

###############################################################

############################
#Query SDSS for galaxy data#
############################
#returns a Pandas dataframe with columns
#objID','SpecObjID','ra','dec','umag','gmag','rmag','imag','zmag','redshift','photo_z','extra'
redshift_dat = query_galaxies(Gal_dat.RA,Gal_dat.DEC)

#merge into Gal_dat
Gal_dat = Gal_dat.join(redshift_dat)

gal_z = Gal_dat['spec_z']
gal_gmag = Gal_dat['gmag']
gal_rmag = Gal_dat['rmag']
gal_imag = Gal_dat['imag']

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
    keys = np.arange(0,Gal_dat.SLIT_WIDTH.size,1).astype('string')
    slit_type = {}
    print 'Is this a galaxy (g), a reference star (r), or empty sky (s)?'
    for i in range(len(Gal_dat)):
        d.set('pan to '+Gal_dat.RA[i]+' '+Gal_dat.DEC[i]+' wcs fk5')
        if Gal_dat.SLIT_WIDTH[i] == '1.0':
            d.set('regions command {box('+Gal_dat.RA[i]+' '+Gal_dat.DEC[i]+' 3 24) #color=green}')
        else:
            d.set('regions command {box('+Gal_dat.RA[i]+' '+Gal_dat.DEC[i]+' 12 12) #color=green}')
        while True:
            char = getch()
            if char.lower() in ("g", "r", "s"):
                break

        slit_type[keys[i]] = char.lower()
    pickle.dump(slit_type,open(clus_id+'/'+clus_id+'_slittypes.pkl','wb'))
else:
    slit_type = pickle.load(open(clus_id+'/'+clus_id+'_slittypes.pkl','rb'))

stypes = pd.DataFrame(slit_type.values(),index=np.array(slit_type.keys()).astype('int'),columns=['slit_type'])
Gal_dat = Gal_dat.join(stypes)
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
    FINAL_SLIT_X = np.zeros(len(Gal_dat))
    FINAL_SLIT_Y = np.zeros(len(Gal_dat))
    SLIT_WIDTH = np.zeros(len(Gal_dat))
    print 'If needed, move region box to desired location. To increase the size, drag on corners'
    for i in range(SLIT_WIDTH.size):
        d.set('pan to 1150.0 '+str(Gal_dat.SLIT_Y[i])+' physical')
        print 'Galaxy at ',Gal_dat.RA[i],Gal_dat.DEC[i]
        d.set('regions command {box(2000 '+str(Gal_dat.SLIT_Y[i])+' 4500 40) #color=green highlite=1}')
        #raw_input('Once done: hit ENTER')
        if Gal_dat.slit_type[i] == 'g':
            if sdss_check:
                if Gal_dat.spec_z[i] != 0.0: skipgal = False
                else: skipgal = True
            else: skipgal = False
            if not skipgal:
                print 'Is this spectra good (y) or bad (n)?'
                while True:
                    char = getch()
                    if char.lower() in ("y","n"):
                        break
                good_spectra = np.append(good_spectra,'y')#char.lower())
                newpos_str = d.get('regions').split('\n')
                for n_string in newpos_str:
                    if n_string[:3] == 'box':
                        newpos = re.search('box\(.*,(.*),.*,(.*),.*\)',n_string)
                        FINAL_SLIT_X[i] = Gal_dat.SLIT_X[i]
                        FINAL_SLIT_Y[i] = newpos.group(1)
                        SLIT_WIDTH[i] = newpos.group(2)
                        break
            else:
                good_spectra = np.append(good_spectra,'n')
                FINAL_SLIT_X[i] = Gal_dat.SLIT_X[i]
                FINAL_SLIT_Y[i] = Gal_dat.SLIT_Y[i]
                SLIT_WIDTH[i] = 40
        else:
            good_spectra = np.append(good_spectra,'n')
            FINAL_SLIT_X[i] = Gal_dat.SLIT_X[i]
            FINAL_SLIT_Y[i] = Gal_dat.SLIT_Y[i]
            SLIT_WIDTH[i] = 40
        print FINAL_SLIT_X[i],FINAL_SLIT_Y[i],SLIT_WIDTH[i]
        d.set('regions delete all')
    print FINAL_SLIT_X
    np.savetxt(clus_id+'/'+clus_id+'_slit_pos_qual.tab',np.array(zip(FINAL_SLIT_X,FINAL_SLIT_Y,SLIT_WIDTH,good_spectra),dtype=[('float',float),('float2',float),('int',int),('str','|S1')]),delimiter='\t',fmt='%10.2f %10.2f %3d %s')
else:
    FINAL_SLIT_X,FINAL_SLIT_Y,SLIT_WIDTH = np.loadtxt(clus_id+'/'+clus_id+'_slit_pos_qual.tab',dtype='float',usecols=(0,1,2),unpack=True)
    good_spectra = np.loadtxt(clus_id+'/'+clus_id+'_slit_pos_qual.tab',dtype='string',usecols=(3,),unpack=True)

Gal_dat['FINAL_SLIT_X'],Gal_dat['FINAL_SLIT_Y'],Gal_dat['SLIT_WIDTH'],Gal_dat['good_spectra'] = FINAL_SLIT_X,FINAL_SLIT_Y,SLIT_WIDTH,good_spectra

#Need to flip FINAL_SLIT_X coords to account for reverse wavelength spectra
Gal_dat['FINAL_SLIT_X_FLIP'] = 4064 - Gal_dat.FINAL_SLIT_X
####################################################################
#######################################
#Reduction steps to prep science image#
#######################################
redo = 'n'
if os.path.isfile(clus_id+'/science/'+clus_id+'_science.cr.fits'):
    redo = raw_input('Detected cosmic ray filtered file exists. Do you wish to use this (y) or remove and re-calculate (n)?')
if redo == 'n':
    #os.remove(clus_id+'/science/'+clus_id+'_science.cr.fits')
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
    #os.remove(clus_id+'/offset_sky/'+clus_id+'_offset.cr.fits')
    skyfits_c = copy.copy(hdulist_sky)
    filt = filter_image(hdulist_sky[0].data)
    skyfits_c[0].data = filt + np.abs(np.nanmin(filt))
    skyfits_c.writeto(clus_id+'/offset_sky/'+clus_id+'_offset.cr.fits')
else: skyfits_c = pyfits.open(clus_id+'/offset_sky/'+clus_id+'_offset.cr.fits')

print 'FLAT REDUCTION'
if redo == 'n':
    #os.remove(clus_id+'/flats/'+clus_id+'_flat.cr.fits')
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
    #os.remove(clus_id+'/arcs/'+clus_id+'_arc.cr.fits')
    arcfits_c = copy.copy(hdulists_arc[0]) #copy I will use to hold the smoothed and added results
    arcfits_c.data *= 0.0
    for arcfits in hdulists_arc:
        filt = arcfits.data#filter_image(arcfits.data)
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
wave = np.zeros((len(Gal_dat),4064))
if os.path.isfile(clus_id+'/'+clus_id+'_stretchshift.tab'):
    reassign = raw_input('Detected file with stretch and shift parameters for each spectra. Do you wish to use this (y) or remove and re-adjust (n)? ')
if reassign == 'n':
    #create write file
    f = open(clus_id+'/'+clus_id+'_stretchshift.tab','w')
    f.write('#X_SLIT_FLIP     Y_SLIT     SHIFT     STRETCH     QUAD     CUBE     FOURTH    FIFTH    WIDTH \n')
    
    #initialize polynomial arrays
    fifth,fourth,cube,quad,stretch,shift =  np.zeros((6,len(Gal_dat)))
    fifth_est,fourth_est,cube_est,quad_est,shift_est,stretch_est = np.zeros((6,len(Gal_dat)))
    Flux = np.zeros((len(Gal_dat),4064))
    calib_data = arcfits_c.data
    p_x = np.arange(0,4064,1)
    ii = 0
    
    #do reduction for initial galaxy
    while ii <= stretch.size:
        if good_spectra[ii]=='y':
            f_x = np.sum(calib_data[Gal_dat.FINAL_SLIT_Y[ii]-Gal_dat.SLIT_WIDTH[ii]/2.0:Gal_dat.FINAL_SLIT_Y[ii]+Gal_dat.SLIT_WIDTH[ii]/2.0,:],axis=0)
            d.set('pan to 1150.0 '+str(Gal_dat.FINAL_SLIT_Y[ii])+' physical')
            d.set('regions command {box(2000 '+str(Gal_dat.FINAL_SLIT_Y[ii])+' 4500 '+str(Gal_dat.SLIT_WIDTH[ii])+') #color=green highlite=1}')
            stretch_est[ii],shift_est[ii],quad_est[ii] = interactive_plot(p_x,f_x,0.70,0.0,0.0,cube_est[ii],fourth_est[ii],fifth_est[ii],Gal_dat.FINAL_SLIT_X_FLIP[ii])

            #Pick lines for initial parameter fit
            line_matches = {'lines':[],'peaks':[]}
            cal_states = {'Xe':True,'Ar':False,'HgNe':False,'Ne':False}
            fig,ax = plt.subplots(1)
            plt.subplots_adjust(right=0.8)
            for j in range(wm.size):
                ax.axvline(wm[j],color='r')
            line, = ax.plot(wm,fm/2.0,'ro',picker=5)# 5 points tolerance
            xspectra = quad_est[ii]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**2 + stretch_est[ii]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii]) + shift_est[ii]
            yspectra = (f_x[::-1]-f_x.min())/10.0
            fline, = plt.plot(xspectra,yspectra,'b',picker=5)
            browser = LineBrowser(fig,ax,line,wm,fm,p_x,fline,xspectra,yspectra,line_matches,cal_states)
            fig.canvas.mpl_connect('pick_event', browser.onpick)
            fig.canvas.mpl_connect('key_press_event',browser.onpress)
            closeax = plt.axes([0.83, 0.7, 0.15, 0.1])
            button = Button(closeax, 'Add Line (x)', hovercolor='0.975')
            button.on_clicked(browser.add_line)
            undoax = plt.axes([0.83,0.5,0.15,0.1])
            undo_button = Button(undoax,'Undo',hovercolor='0.975')
            undo_button.on_clicked(browser.undo)
            stateax = plt.axes([0.83,0.3,0.15,0.1])
            states = CheckButtons(stateax,cal_states.keys(), cal_states.values())
            states.on_clicked(browser.set_calib_lines)
            plt.show()
                
            params,pcov = curve_fit(polyfour,(np.sort(browser.line_matches['peaks'])-Gal_dat.FINAL_SLIT_X_FLIP[ii]),np.sort(browser.line_matches['lines']),p0=[shift_est[ii],stretch_est[ii],quad_est[ii],1e-8,1e-12,1e-12])
            cube_est = cube_est + params[3]
            fourth_est = fourth_est + params[4]
            fifth_est = fifth_est + params[5]

            wave[ii] = params[0]+params[1]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])+params[2]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**2+params[3]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**3.0+params[4]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**4.0+params[5]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**5.0
            flu = f_x - np.min(f_x)
            flu = flu[::-1]
            Flux[ii] = flu/signal.medfilt(flu,201)
            fifth[ii],fourth[ii],cube[ii],quad[ii],stretch[ii],shift[ii] = params[5],params[4],params[3],params[2],params[1],params[0]
            plt.plot(wave[ii],Flux[ii]/np.max(Flux[ii]))
            plt.plot(wm,fm/np.max(fm),'ro')
            for j in range(browser.wm.size):
                plt.axvline(browser.wm[j],color='r')
            plt.xlim(3800,6000)
            try:
                plt.savefig(clus_id+'/figs/'+str(ii)+'.wave.png')
            except:
                os.mkdir(clus_id+'/figs')
                plt.savefig(clus_id+'/figs/'+str(ii)+'.wave.png')
            plt.show()
            f.write(str(Gal_dat.FINAL_SLIT_X_FLIP[ii])+'\t')
            f.write(str(Gal_dat.FINAL_SLIT_Y[ii])+'\t')
            f.write(str(shift[ii])+'\t')
            f.write(str(stretch[ii])+'\t')
            f.write(str(quad[ii])+'\t')
            f.write(str(cube[ii])+'\t')
            f.write(str(fourth[ii])+'\t')
            f.write(str(fifth[ii])+'\t')
            f.write(str(Gal_dat.SLIT_WIDTH[ii])+'\t')
            f.write('\n')
            print 'Wave calib',ii
            ii += 1
            break
            
        f.write(str(Gal_dat.FINAL_SLIT_X_FLIP[ii])+'\t')
        f.write(str(Gal_dat.FINAL_SLIT_Y[ii])+'\t')
        f.write(str(shift[ii])+'\t')
        f.write(str(stretch[ii])+'\t')
        f.write(str(quad[ii])+'\t')
        f.write(str(cube[ii])+'\t')
        f.write(str(fourth[ii])+'\t')
        f.write(str(fifth[ii])+'\t')
        f.write(str(Gal_dat.SLIT_WIDTH[ii])+'\t')
        f.write('\n')
        ii+=1

    #estimate stretch,shift,quad terms with sliders for 2nd - all galaxies
    for i in range(ii,len(Gal_dat)):
        print 'Calibrating',i,'of',stretch.size
        if Gal_dat.good_spectra[i] == 'y':
            if sdss_check:
                if Gal_dat.spec_z[i] != 0.0: skipgal = False
                else: skipgal = True
            else: skipgal = False
            if not skipgal:
                p_x = np.arange(0,4064,1)
                f_x = np.sum(calib_data[Gal_dat.FINAL_SLIT_Y[i]-Gal_dat.SLIT_WIDTH[i]/2.0:Gal_dat.FINAL_SLIT_Y[i]+Gal_dat.SLIT_WIDTH[i]/2.0,:],axis=0)
                d.set('pan to 1150.0 '+str(Gal_dat.FINAL_SLIT_Y[i])+' physical')
                d.set('regions command {box(2000 '+str(Gal_dat.FINAL_SLIT_Y[i])+' 4500 '+str(Gal_dat.SLIT_WIDTH[i])+') #color=green highlite=1}')
                #stretch_est[i],shift_est[i],quad_est[i] = interactive_plot(p_x,f_x,stretch_est[i-1],shift_est[i-1]-(Gal_dat.FINAL_SLIT_X_FLIP[i]*stretch_est[0]-Gal_dat.FINAL_SLIT_X_FLIP[i-1]*stretch_est[i-1]),quad[i-1],cube[i-1],fourth[i-1],fifth[i-1],Gal_dat.FINAL_SLIT_X_FLIP[i])
                reduced_slits = np.where(stretch != 0.0)
                stretch_est[i],shift_est[i],quad_est[i] = interactive_plot(p_x,f_x,stretch[reduced_slits][-1],shift[reduced_slits][-1]+(Gal_dat.FINAL_SLIT_X_FLIP.values[reduced_slits][-1]-Gal_dat.FINAL_SLIT_X_FLIP[i]),quad[reduced_slits][-1],cube[reduced_slits][-1],fourth[reduced_slits][-1],fifth[reduced_slits][-1],Gal_dat.FINAL_SLIT_X_FLIP[i])

                line_matches = {'lines':[],'peaks':[]}
                cal_states = {'Xe':True,'Ar':False,'HgNe':False,'Ne':False}
                fig,ax = plt.subplots(1)
                plt.subplots_adjust(right=0.8)
                for j in range(wm.size):
                    ax.axvline(wm[j],color='r')
                line, = ax.plot(wm,fm/2.0,'ro',picker=5)# 5 points tolerance
                xspectra = quad_est[i]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**2 + stretch_est[i]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i]) + shift_est[i]
                yspectra = (f_x[::-1]-f_x.min())/10.0
                fline, = plt.plot(xspectra,yspectra,'b',picker=5)
                browser = LineBrowser(fig,ax,line,wm,fm,p_x,fline,xspectra,yspectra,line_matches,cal_states)
                fig.canvas.mpl_connect('pick_event', browser.onpick)
                fig.canvas.mpl_connect('key_press_event',browser.onpress)
                closeax = plt.axes([0.83, 0.5, 0.15, 0.1])
                button = Button(closeax, 'Add Line (x)', hovercolor='0.975')
                button.on_clicked(browser.add_line)
                undoax = plt.axes([0.83,0.3,0.15,0.1])
                undo_button = Button(undoax,'Undo',hovercolor='0.975')
                undo_button.on_clicked(browser.undo)
                stateax = plt.axes([0.83,0.8,0.15,0.1])
                states = CheckButtons(stateax,cal_states.keys(), cal_states.values())
                states.on_clicked(browser.set_calib_lines)
                plt.show()
                
                params,pcov = curve_fit(polyfour,(np.sort(browser.line_matches['peaks'])-Gal_dat.FINAL_SLIT_X_FLIP[i]),np.sort(browser.line_matches['lines']),p0=[shift_est[i],stretch_est[i],quad_est[i],1e-8,1e-12,1e-12])
                cube_est[i] = params[3]
                fourth_est[i] = params[4]
                fifth_est[i] = params[5]

                wave[i] = params[0]+params[1]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])+params[2]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**2+params[3]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**3.0+params[4]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**4.0+params[5]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**5.0
                flu = f_x - np.min(f_x)
                flu = flu[::-1]
                Flux[i] = flu/signal.medfilt(flu,201)
                fifth[i],fourth[i],cube[i],quad[i],stretch[i],shift[i] = params[5],params[4],params[3],params[2],params[1],params[0]
                plt.plot(wave[i],Flux[i]/np.max(Flux[i]))
                plt.plot(wm,fm/np.max(fm),'ro')
                for j in range(browser.wm.size):
                    plt.axvline(browser.wm[j],color='r')
                plt.xlim(3800,6000)
                try:
                    plt.savefig(clus_id+'/figs/'+str(i)+'.wave.png')
                except:
                    os.mkdir(clus_id+'/figs')
                    plt.savefig(clus_id+'/figs/'+str(i)+'.wave.png')
                plt.close()

        f.write(str(Gal_dat.FINAL_SLIT_X_FLIP[i])+'\t')
        f.write(str(Gal_dat.FINAL_SLIT_Y[i])+'\t')
        f.write(str(shift[i])+'\t')
        f.write(str(stretch[i])+'\t')
        f.write(str(quad[i])+'\t')
        f.write(str(cube[i])+'\t')
        f.write(str(fourth[i])+'\t')
        f.write(str(fifth[i])+'\t')
        f.write(str(Gal_dat.SLIT_WIDTH[i])+'\t')
        f.write('\n')
    f.close()
else:
    xslit,yslit,shift,stretch,quad,cube,fourth,fifth,wd = np.loadtxt(clus_id+'/'+clus_id+'_stretchshift.tab',dtype='float',usecols=(0,1,2,3,4,5,6,7,8),unpack=True)
    #FINAL_SLIT_X = np.append(FINAL_SLIT_X[0],xslit)
    #FINAL_SLIT_Y = np.append(FINAL_SLIT_Y[0],yslit)
    #SLIT_WIDTH = np.append(SLIT_WIDTH[0],wd)
    for i in range(stretch.size):
        wave[i] = fifth[i]*(np.arange(0,4064,1)-Gal_dat.FINAL_SLIT_X_FLIP[i])**5 + fourth[i]*(np.arange(0,4064,1)-Gal_dat.FINAL_SLIT_X_FLIP[i])**4 + cube[i]*(np.arange(0,4064,1)-Gal_dat.FINAL_SLIT_X_FLIP[i])**3 + quad[i]*(np.arange(0,4064,1)-Gal_dat.FINAL_SLIT_X_FLIP[i])**2 + stretch[i]*(np.arange(0,4064,1)-Gal_dat.FINAL_SLIT_X_FLIP[i]) + shift[i]

#summed science slits + filtering to see spectra
#Flux_science_old = np.array([np.sum(scifits_c2.data[Gal_dat.FINAL_SLIT_Y[i]-Gal_dat.SLIT_WIDTH[i]/2.0:Gal_dat.FINAL_SLIT_Y[i]+Gal_dat.SLIT_WIDTH[i]/2.0,:],axis=0)[::-1] for i in range(len(Gal_dat))])
Flux_science = np.array([gal_trace(scifits_c2.data[Gal_dat.FINAL_SLIT_Y[i]-Gal_dat.SLIT_WIDTH[i]/2.0:Gal_dat.FINAL_SLIT_Y[i]+Gal_dat.SLIT_WIDTH[i]/2.0,:])[::-1] for i in range(len(Gal_dat))])

#Add parameters to Dataframe
Gal_dat['shift'],Gal_dat['stretch'],Gal_dat['quad'],Gal_dat['cube'],Gal_dat['fourth'],Gal_dat['fifth'] = shift,stretch,quad,cube,fourth,fifth


####################
#Redshift Calibrate#
####################
early_type = pyfits.open('spDR2-023.fit')
#normal_type = pyfits.open('spDR2-024.fit')
#normal2_type = pyfits.open('spDR2-025.fit')
coeff0 = early_type[0].header['COEFF0']
coeff1 = early_type[0].header['COEFF1']
#coeff0_2 = normal_type[0].header['COEFF0']
#coeff1_2 = normal_type[0].header['COEFF1']
#coeff0_3 = normal2_type[0].header['COEFF0']
#coeff1_3 = normal2_type[0].header['COEFF1']
early_type_flux = early_type[0].data[0] - signal.medfilt(early_type[0].data[0],171)
#early_type_flux = signal.medfilt(early_type[0].data[0],171)
#normal_type_flux = normal_type[0].data[0]
#normal2_type_flux = normal2_type[0].data[0]
early_type_wave = 10**(coeff0 + coeff1*np.arange(0,early_type_flux.size,1))
#normal_type_wave = 10**(coeff0 + coeff1*np.arange(0,normal_type_flux.size,1))
#normal2_type_wave = 10**(coeff0 + coeff1*np.arange(0,normal2_type_flux.size,1))

redshift_est = np.zeros(len(Gal_dat))
redshift_est2 = np.zeros(len(Gal_dat))
redshift_est3 = np.zeros(len(Gal_dat))
cor = np.zeros(len(Gal_dat))
cor2 = np.zeros(len(Gal_dat))
cor3 = np.zeros(len(Gal_dat))
HSN = np.zeros(len(Gal_dat))
KSN = np.zeros(len(Gal_dat))
GSN = np.zeros(len(Gal_dat))
SNavg = np.zeros(len(Gal_dat))
SNHKmin = np.zeros(len(Gal_dat))

sdss_elem = np.where(Gal_dat.spec_z > 0.0)[0]
sdss_red = Gal_dat[Gal_dat.spec_z > 0.0].spec_z
qualityval = {'Clear':np.zeros(len(Gal_dat))}

est_pre_z = raw_input('Your sample contains '+str(Gal_dat.spec_z[Gal_dat.spec_z > 0.0].size)+' SDSS galaxies with spectra. Would you like to use a redshift prior that is the median of these galaxies (s)? If not, would you like to specify your own prior for each galaxy (q)? If not, press (p) to use the sdss photo_z as a prior: ')

#Choose redshift prior information
est_enter = False
while not est_enter:
    if est_pre_z == 's':
        z_prior_width = 0.04
        est_enter = True
    elif est_pre_z == 'p':
        z_prior_width = 0.06
        est_enter = True
    elif est_pre_z == 'q':
        z_prior_width = 0.06
        est_enter = True
    else:
        est_pre_z = raw_input('Incorrect entry: Please enter either (s), (q), or (p). Your sample contains '+str(Gal_dat.spec_z[Gal_dat.spec_z > 0.0].size)+' SDSS galaxies with spectra. Would you like to use a redshift prior that is the median of these galaxies (s)? If not, would you like to specify your own prior for each galaxy (q)? If not, press (p) to use the sdss photo_z as a prior: ')


for k in range(len(Gal_dat)):
    F1 = fftpack.rfft(Flux_science[k])
    cut = F1.copy()
    W = fftpack.rfftfreq(wave[k].size,d=wave[k][2001]-wave[k][2000])
    cut[np.where(W>0.15)] = 0
    Flux_science2 = fftpack.irfft(cut)
    '''
    plt.plot(wave[k],Flux_science2)
    plt.plot(wave[k],Flux_science[k],c='g',alpha=0.5)
    plt.xlim(3900,5000)
    plt.show()
    '''

    Flux_sc = Flux_science2 - signal.medfilt(Flux_science2,171)

    if Gal_dat.slit_type[k] == 'g':
        if sdss_check:
            if Gal_dat.spec_z[k] != 0.0: skipgal = False
            else: skipgal = True
        else: skipgal = False
        if not skipgal:
            d.set('pan to 1150.0 '+str(Gal_dat.FINAL_SLIT_Y[k])+' physical')
            d.set('regions command {box(2000 '+str(Gal_dat.FINAL_SLIT_Y[k])+' 4500 40) #color=green highlite=1}')
            #assign prior to redshift depending on photo_z or user defined.
            if est_pre_z == 's':
                pre_z_est = np.median(Gal_dat.spec_z[Gal_dat.spec_z > 0.0])
            if est_pre_z == 'p':
                pre_z_est = Gal_dat.photo_z[k]
            elif est_pre_z == 'q':
                print 'Take a look at the plotted galaxy spectrum and note, approximately, at what wavelength do the H and K lines exist? Then close the plot and enter that wavelength in angstroms.'
                plt.plot(wave[k],Flux_science2)
                plt.xlim(4000.0,7000.0)
                plt.show()
                HKinit = raw_input('HK approx. wavelength (A): ')
                pre_z_est = np.float(HKinit)/3950.0 - 1

            redshift_est[k],cor[k] = redshift_estimate(pre_z_est,z_prior_width,early_type_wave,early_type_flux,wave[k],Flux_sc)
            fig = plt.figure()
            ax2 = fig.add_subplot(111)
            plt.subplots_adjust(right=0.8)
            pspec, = ax2.plot(wave[k],Flux_science2)
            ax2.axvline(3725.0*(1+redshift_est[k]),ls='--',alpha=0.7,c='blue')
            ax2.axvline(3968.5*(1+redshift_est[k]),ls='--',alpha=0.7,c='red')
            ax2.axvline(3933.7*(1+redshift_est[k]),ls='--',alpha=0.7,c='red')
            ax2.axvline(4102.9*(1+redshift_est[k]),ls='--',alpha=0.7,c='orange')
            ax2.axvline(4304.0*(1+redshift_est[k]),ls='--',alpha=0.7,c='orange')
            ax2.axvline(4862.0*(1+redshift_est[k]),ls='--',alpha=0.7,c='orange')
            ax2.axvline(4959.0*(1+redshift_est[k]),ls='--',alpha=0.7,c='blue')
            ax2.axvline(5007.0*(1+redshift_est[k]),ls='--',alpha=0.7,c='blue')
            ax2.axvline(5175.0*(1+redshift_est[k]),ls='--',alpha=0.7,c='orange')
            HK_est = EstimateHK(pspec,ax2)
            rax = plt.axes([0.85, 0.5, 0.1, 0.2])
            radio = RadioButtons(rax, ('Unclear', 'Clear'))
            def qualfunc(label):
                if label == 'Clear':
                    qualityval['Clear'][k] = 1
                else:
                    qualityval['Clear'][k] = 0
            radio.on_clicked(qualfunc)
            closeax = plt.axes([0.83, 0.3, 0.15, 0.1])
            button = Button(closeax, 'Accept & Close', hovercolor='0.975')
            def closeplot(event):
                plt.close()
            button.on_clicked(closeplot)
            ax2.set_xlim(3800,5100)
            plt.show()
            try:
                pre_lam_est = HK_est.lam
                pre_z_est = pre_lam_est/3950.0 - 1.0
                redshift_est[k],cor[k] = redshift_estimate(pre_z_est,z_prior_width,early_type_wave,early_type_flux,wave[k],Flux_sc)
                print 'Using prior given by user'
                figure = plt.figure()
                ax = figure.add_subplot(111)
                plt.subplots_adjust(right=0.8)
                spectra, = ax.plot(wave[k]/(1+redshift_est[k]),Flux_science2)
                ax.axvline(3725.5,ls='--',alpha=0.7,c='blue')
                ax.axvline(3968.5,ls='--',alpha=0.7,c='red')
                ax.axvline(3933.7,ls='--',alpha=0.7,c='red')
                ax.axvline(4102.9,ls='--',alpha=0.7,c='orange')
                ax.axvline(4304.0,ls='--',alpha=0.7,c='orange')
                ax.axvline(4862.0,ls='--',alpha=0.7,c='orange')
                ax.axvline(4959.0,ls='--',alpha=0.7,c='blue')
                ax.axvline(5007.0,ls='--',alpha=0.7,c='blue')
                ax.axvline(5175.0,ls='--',alpha=0.7,c='orange')
                rax = plt.axes([0.85, 0.5, 0.1, 0.2])
                if qualityval['Clear'][k] == 0:
                    radio = RadioButtons(rax, ('Unclear', 'Clear'))
                else:
                    radio = RadioButtons(rax, ('Unclear', 'Clear'),active=1)
                radio.on_clicked(qualfunc)
                closeax = plt.axes([0.83, 0.3, 0.15, 0.1])
                button = Button(closeax, 'Accept & Close', hovercolor='0.975')
                button.on_clicked(closeplot)
                ax.set_xlim(3500,4600)
                print 'got to drag'
                spectra2 = DragSpectra(spectra,Flux_science2,ax)
                figure.canvas.mpl_connect('motion_notify_event',spectra2.on_motion)
                figure.canvas.mpl_connect('button_press_event',spectra2.on_press)
                figure.canvas.mpl_connect('button_release_event',spectra2.on_release)
                plt.show()
                total_new_shift = spectra2.dx_tot
                print total_new_shift
                redshift_est[k] = (3968.5*(1+redshift_est[k]) - total_new_shift)/3968.5 - 1
            except:
                print 'failed redshift fix'
                pass
            HSN[k],KSN[k],GSN[k] = sncalc(redshift_est[k],wave[k],Flux_sc)
            SNavg[k] = np.average(np.array([HSN[k],KSN[k],GSN[k]]))
            SNHKmin[k] = np.min(np.array([HSN[k],KSN[k]]))

    else:
        redshift_est[k] = 0.0
        cor[k] = 0.0

    if k in sdss_elem.astype('int'):
        print 'Estimate: %.5f'%(redshift_est[k]), 'SDSS: %.5f'%(sdss_red.values[np.where(sdss_elem==k)][0])
    print 'z found for galaxy '+str(k+1)+' of '+str(len(Gal_dat))

#Add redshift estimates, SN, Corr, and qualityflag to the Dataframe
Gal_dat['est_z'],Gal_dat['cor'],Gal_dat['HSN'],Gal_dat['KSN'],Gal_dat['GSN'],Gal_dat['quality_flag'] = redshift_est,cor,HSN,KSN,GSN,qualityval['Clear']

plt.plot(Gal_dat['spec_z'],Gal_dat['est_z'],'ro')
#plt.plot(sdss_red,redshift_est2[sdss_elem.astype('int')],'bo')
#plt.plot(sdss_red,redshift_est3[sdss_elem.astype('int')],'o',c='purple')
plt.plot(sdss_red,sdss_red,'k')
plt.savefig(clus_id+'/redshift_compare.png')
plt.show()

f = open(clus_id+'/estimated_redshifts.tab','w')
f.write('#RA    DEC    Z_est    Z_sdss  correlation   H S/N    K S/N     G S/N  gal_gmag    gal_rmag    gal_imag\n')
for k in range(redshift_est.size):
    f.write(Gal_dat.RA[k]+'\t')
    f.write(Gal_dat.DEC[k]+'\t')
    f.write(str(Gal_dat.est_z[k])+'\t')
    f.write(str(Gal_dat.spec_z[k])+'\t')
    #if k in sdss_elem.astype('int'):
    #    f.write(str(sdss_red[sdss_elem==k].values[0])+'\t')
    #else:
    #    f.write(str(0.000)+'\t')
    f.write(str(cor[k])+'\t')
    f.write(str(HSN[k])+'\t')
    f.write(str(KSN[k])+'\t')
    f.write(str(GSN[k])+'\t')
    f.write(str(gal_gmag[k])+'\t')
    f.write(str(gal_rmag[k])+'\t')
    f.write(str(gal_imag[k])+'\t')
    f.write('\n')
f.close()

#Output dataframe
Gal_dat.to_csv(clus_id+'/results.csv')
