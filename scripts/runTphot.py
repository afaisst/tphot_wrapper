## Runs TPHOT
# A. Faisst
#
# Run as runTphot jobX.json
#
#
#
# #############
import os, sys
import numpy as np
import time
import sh
import math
import subprocess
import json

import pyjs9 as js9

from astropy.io import fits, ascii
from astropy.table import Table, Column, MaskedColumn, hstack, vstack
import astropy.wcs as wcs

from astropy.convolution import Gaussian2DKernel
from astropy.modeling.models import Gaussian2D
from astropy.nddata.utils import Cutout2D

from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.convolution import convolve

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as path_effects

from scipy import ndimage
from scipy import interpolate

from photutils import (create_matching_kernel, resize_psf)
from photutils import (HanningWindow, TukeyWindow, CosineBellWindow,
                       SplitCosineBellWindow, TopHatWindow)

## Plotting stuff
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
#mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['hatch.linewidth'] = 4

def_cols = plt.rcParams['axes.prop_cycle'].by_key()['color']


### FUNCTIONS ###

## This function creates a placeholder TPHOT configuration file.
def create_TPHOT_template(filename,workdir,pixscaleratio,dilation,zeropoint):
    
    param_txt = '''## PARAMETERS FILE
# Following are all the parameters available.
# The path to the files MUST be relative to workdir. All the directories in the parameter files MUST exist.

### GENERAL ###

use_real=true							# Use real 2D profiles cut out from input High Resolution image
use_stamps=false						# Use analytical 2D profiles
use_unresolved=false						# Use psf fitting over unresolved point-like sources
pixel_ratio=*PIXSCALERATIO*							# The ratio between High Resolution and Low Resolution images
low_resolution_image=*WORKDIR*/lores.fits		# Input Low Resolution image
low_resolution_rms=*WORKDIR*/loresrms.fits		# Input Low Resolution RMS
rms_check=false							# Check RMS map
rms_check_value=1000						# The RMS value limit, mandatory if rms_check is true
real_catalog=*WORKDIR*/tphotcat.fits			# Input Catalog for use_real option
stamps_catalog=none			# Input Catalog for use_stamps option
stamps_hdf5=none				# File name which contain all the stamps to be use in use_stamps option
unresolved_catalog=none		# Input Catalog for use_unresolved option
segmentation_map=*WORKDIR*/seg.fits			# Input Segmentation Map
save_intermediate=true						# Save intermediate results. If false it saves memory usage
save_ram=false                              # Save RAM usage
final_catalog=*WORKDIR*/tphot_output.fits			# Output catalog

order=standard # ORDER KEYWORD DOESNT DO ANYTHING

### DILATE ###

dilate=*DILATION*
dilated_segmap=*WORKDIR*/dilated_segmap.fits
dilated_catalog=*WORKDIR*/dilated_input_catalog.fits
minarea_dilate=2.0
maxarea_dilate=10000.0
minfactor_dilate=4.0
maxfactor_dilate=12.0
dilation_factor=2.0
dilation_threshold=0.0

### CUTOUT ###

subtract_background=false					# Subtract measured background at centroid
culling=false							# Remove from the input catalog sources outside the boundaries of the Low Resolution image
high_resolution_image=*WORKDIR*/hires.fits		# The High Resolution image from which the sources where detected
cutout_catalog=*WORKDIR*/cutout.fits			# File name which will contain the catalog after the cutout step
cutout_hdf5=*WORKDIR*/cutout.h5				# File name which will contain all the cutouts from the High Resolution image

### CONVOLVE ###

real_single_kernel=true						# Use single or multiple kernel
real_kernel=*WORKDIR*/kernel.fits			# If real_single_kernel true a FITS file, if real_single_kernel false a HDF5 file
#stamps_single_kernel=true					# Use single or multiple kernel
#stamps_kernel=none			# If stamps_single_kernel true a FITS file, if stamps_single_kernel false a HDF5 file
#unresolved_single_psf=true					# Use single or multiple psf
#low_resolution_psf=psf.fits		# If unresolved_single_kernel true a FITS file, if unresolved_single_kernel false a HDF5 file
convolve_catalog_1st=*WORKDIR*/convolve_1st.fits		# File name for the convolved source of the first step
convolve_hdf5_1st=*WORKDIR*/convolve_1st.h5			# File name for the catalog after the first step
convolve_catalog_2nd=*WORKDIR*/convolve_2nd.fits		# File name for the convolved source of the second step
convolve_hdf5_2nd=*WORKDIR*/convolve_2nd.h5			# File name for the catalog after the second step
convolve_catalog_apply_shifts=false				# True only for TPhot_convolve standalone 2nd pass

### FITTER ###

error_type=RMS							# Kind of error to be used: RMS (use low_res_error) WEIGHT CONSTANT
rms_constant=0							# Number used to create the error if error_type is WEIGHT or CONSTANT
fitter_cell=COO							# COO: cell on object, SIN: whole image, OPT: optimized on number of sources and image size, GRD: specified by the user
x_cell_dim=0							# Cell size only for GRD
y_cell_dim=0							# Cell size only for GRD
dithercell=false						# Cell pattern is dithered
cell_overlap=0							# Overlap between cells to be used in optimized cells
fitting_method=LU						# LU CHO
fit_background=false						# Fit local background with a flat template
background_constant=0.0						# Constant background to be removed from the Low Resolution image before fitting
cellmask=false							# Use a mask to exclude saturated pixels ## Better results if TRUE
threshold=0.0							# Only fit pixels over this number
clip=false							# Clip out large negative fluxes and re-do the fit
n_sigma=3.0							# Number of sigma to accept linear system solution for the clip option
flux_priors=false						# Use an external catalog for fluxes priors for comparison
flux_priors_catalog=input_folder/flux_priors.fits		# Flux priors catalog containing SOURCEID, FLUX, FLUX_ERR
fit_catalog_1st=*WORKDIR*/fit_catalog_1st.fits		# File name for the catalog after the first fitting step
covar_file_1st=*WORKDIR*/covar_1st.txt			# File name for the covariance of the first fitting step
covar_file_2nd=*WORKDIR*/covar_2nd.txt			# File name for the covariance of the second fitting step
fitter_2nd_pass=false						# True only for TPhot_fitter standalone 2nd pass
fit_sublist=0                   # Number of source IDs to fit, when only a subsample of them has to be fitted (0 == ALL)
fit_source_ids=[1,2,3]           # List of ID to fit

### APERTURE PHOTOMETRY ###

compute_circular_aperture_flux=0				# Number of apertures on which perform photometry
circular_apertures_list=[13,20,33]				# Apertures on which perform photometry (2,3,5 arcsec apertures at 0.15 pixscale)
gain=0.0
rms_tol=1000
binsubpix=10
zero_point=*ZEROPOINT*
aperture_file_1st=*WORKDIR*/aphot_1st.fits			# File name for the aperture results after the first fitting step
aperture_file_2nd=*WORKDIR*/aphot_2nd.fits			# File name for the aperture results after the second fitting step

### COLLAGE ###

id_map=false							# Save a "finder chart" version of the collage
exclude_ids=0							# Number of sources to exclude
exclude_list=[1,2,3]						# List of sources to exclude
do_residual_stats=true						# Compute the statistics on the residual image
collage_1st=*WORKDIR*/collage_1st.fits			# The name of the FITS where to save the collage after the first iteration
id_map_name_1st=*WORKDIR*/id_map_1st.fits			# The name of the FITS where to save the id_map after the first iteration
residual_1st=*WORKDIR*/residual_1st.fits			# The name of the FITS where to save the residuals after the first iteration
residual_stats_1st=*WORKDIR*/collage_stats_1st.ini		# The name of the txt where to save the residuals statistic after the first iteration
collage_2nd_pass=true						# True only for TPhot_collage standalone 2nd pass
collage_2nd=*WORKDIR*/collage_2nd.fits			# The name of the FITS where to save the collage after the second iteration
id_map_name_2nd=*WORKDIR*/id_map_2nd.fits			# The name of the FITS where to save the id_map after the second iteration
residual_2nd=*WORKDIR*/residual_2nd.fits			# The name of the FITS where to save the residuals after the second iteration
residual_stats_2nd=*WORKDIR*/collage_stats_2nd.ini		# The name of the txt where to save the residuals statistic after the second iteration

### DANCE ###

d_zone_size=10							# Pixels size of region in which kernel/psf shift is computed
x_shift=5							# Maximum allowed shift in LRI pixelsize on x coordinates
y_shift=5							# Maximum allowed shift in LRI pixelsize on y coordinates
max_shift=5							# Maximum allowed shift in LRI pixelsize
n_neigh_interp=0						# Number of neighbors over which to smooth the shifts. 0 means "all neighbors within R"; < 0 means no smoothing
background_constant=0.						# Normalization
dance_file=*WORKDIR*/ddiags.txt				# File containg shifts to be applied

'''
    param_txt = param_txt.replace("*WORKDIR*",workdir)
    param_txt = param_txt.replace("*ZEROPOINT*",str(zeropoint))
    param_txt = param_txt.replace("*PIXSCALERATIO*",str(int(pixscaleratio)))
    
    if dilation:
        dilation = "true"
    else:
        dilation = "false"
    param_txt = param_txt.replace("*DILATION*",dilation)
    
    with open(filename,"w") as f:
        f.write(param_txt)
        
    return(True)



## Interpolates an image with a fraction "frac" of it's pixel size
def interpolate_image(img , frac , verbose):
    
    x = np.arange(0+0.5,img.shape[0]+0.5,1) # put the coordinates in the center of pixels.
    y = np.arange(0+0.5,img.shape[1]+0.5,1)
    #fimg = interpolate.interp2d(x, y, img, kind='cubic') # this is 2d function of image
    fimg = interpolate.RectBivariateSpline(x , y , img , kx=3,ky=3)
    
    if verbose:
        print("original: ", img.shape)
        print(" expected center original: ", np.asarray(img.shape) // 2)
        print(" center original: ", ndimage.maximum_position(img))
        print(" sum original: " , np.nansum(img))

    center = np.asarray(img.shape) // 2 + 0.5
    xy = np.asarray( [ np.arange(0+0.5,img.shape[ii]+0.5,1) for ii in range(2)] )
    nsteps = np.floor( (np.asarray(img.shape) // 2) / frac ).astype("int")
    delta = np.asarray( [np.arange(frac,(nsteps[ii]+1)*frac,frac) for ii in range(2)] )
    xynew1 = np.asarray( [center[ii] + delta[ii] for ii in range(2)] )
    xynew2 = np.asarray( [center[ii] - delta[ii] for ii in range(2)] )
    xynew = np.asarray( [np.concatenate( (np.flip(xynew2[ii]) , np.asarray([center[ii]]) , xynew1[ii]) ) for ii in range(2) ] )
        
    img_interp = fimg(xynew[0], xynew[1]) * frac**2 # adjust sum
    
    if verbose:
        print("interpolated: ", img_interp.shape)
        print(" expected center interpolated: ", np.asarray(img_interp.shape) // 2)
        print(" center interpolated: ", ndimage.maximum_position(img_interp))
        print(" sum interpolated: " , np.nansum(img_interp))
        
    if not list(ndimage.maximum_position(img_interp)) == list( np.asarray(img_interp.shape) // 2 ):
        print("Warning: shift in center position possible")
        print("  expected center interpolated: ", np.asarray(img_interp.shape) // 2)
        print("  center interpolated: ", ndimage.maximum_position(img_interp))
        
    return(img_interp)
    


## Get SExtractor default parameters
def get_default_parfile():
    
    PARS_default = {"CATALOG_NAME": "output.cat", # UPDATE
        "CATALOG_TYPE": "ASCII_HEAD",
       "PARAMETERS_NAME": "./input/sex.par", # UPDATE
        "DETECT_TYPE": "CCD",
        "DETECT_MINAREA": 5, #5
        "THRESH_TYPE": "RELATIVE",
        "DETECT_THRESH": 3,
        "ANALYSIS_THRESH": 1.5,
        "FILTER": "Y",
        "FILTER_NAME": "./input/g2.8.conv", # UPDATE
        "FILTER_THRESH": "",
        "DEBLEND_NTHRESH": 32,
        "DEBLEND_MINCONT": 0.01,
        "CLEAN": "Y",
        "CLEAN_PARAM": 1.0,
        "MASK_TYPE": "CORRECT",
        "WEIGHT_TYPE": "NONE", # UPDATE (this is in command line)
        "WEIGHT_IMAGE": "weight.fits", # UPDATE (this is in command line)
        "WEIGHT_GAIN": "Y",
        "WEIGHT_THRESH": "",
        "FLAG_IMAGE":"NONE",
        "FLAG_TYPE":"OR",
        "PHOT_APERTURES": "30", # UPDATE
        "PHOT_AUTOPARAMS": "2.5,3.5",
        "PHOT_PETROPARAMS": "2.0,3.5",
        "PHOT_AUTOAPERS": "0.0,0.0",
        "PHOT_FLUXFRAC": 0.5,
        "SATUR_LEVEL": 50000.0, # UPDATE
        "MAG_ZEROPOINT": 0.0, # UPDATE
        "MAG_GAMMA": 4.0,
        "GAIN": 0.0, # UPDATE
        "PIXEL_SCALE": 0, # UPDATE
        "SEEING_FWHM": 0.1, # UPDATE
        "STARNNW_NAME": "./input/default.nnw", # UPDATE
        "BACK_TYPE": "AUTO",
        "BACK_VALUE": 0.0,
        "BACK_SIZE": 64,
        "BACK_FILTERSIZE": 3,
        "BACKPHOTO_TYPE": "GLOBAL",
        "BACKPHOTO_THICK": 24,
        "BACK_FILTTHRESH": 0.0,
        "CHECKIMAGE_TYPE": "APERTURES,SEGMENTATION,BACKGROUND", # UPDATE
        "CHECKIMAGE_NAME": "aper.fits, seg.fits, back.fits", # UPDATE
        "NTHREADS":1 # UPDATE?
       }
    
    return(PARS_default)


def get_apertures(pixscale,aperturelist_arcsec=[0.5 , 2.0 , 3.0]):
    
    # apertures in arcsec
    apertures_arcsec = np.asarray(aperturelist_arcsec)
    
    # aperture in pixels (using pixel scale)
    apertures_pix = apertures_arcsec / pixscale
    
    return(apertures_pix)



## Clipping image data to get noise level
def clip(dat,n,niter):
    dat = dat.ravel()
    
    for jj in range(niter):
        med = np.nanmedian(dat)
        stdev = np.nanstd(dat)
    
        index = np.where( (dat < (med+n*stdev)) & (dat > (med-n*stdev)) )[0]
        
        out = {"med": med,
              "stdev": stdev}

        if len(index) == len(dat):
            return(out)
        
        if len(index) > 0:
            dat = dat[index]
        
    return(out)

def replace_in_file(filename, old_string, new_string):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        if old_string not in s:
            print("Could not find " + old_string)
            return

    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:
        s = s.replace(old_string, new_string)
        f.write(s)
        
# Returns modeled PSFex PSF for a 1D polynomial (e.g., magnitude)
# input loaded psfcube (psfcube) and its header (psfcube_h) together with parameter (e.g., magnitude)
def modelPSF_1D(psfcube,psfcube_h,param):
    X = psfcube[0][0][0].copy() # constant
    if psfcube_h["POLNAXIS"] > 0:
        x_scaled = (param - psfcube_h["POLZERO1"]) / psfcube_h["POLSCAL1"]
        for iii in range(psfcube_h["POLDEG1"]):
            X += psfcube[0][0][iii+1].copy() * x_scaled**(iii+1)
    return(X)


# Write the log file to disk
def write_log(LOG , file_name):
    with open(file_name,"w") as f:
	for ii in range(len(LOG)):
		f.write( "%s\n" % (LOG[ii]) )



##################################



## User input
userinput_file = sys.argv[1]
with open(userinput_file) as json_file:
    userinput = json.load(json_file)

## Log file
LOG = []
output_logfile_name = "%s.txt" % ".".join(userinput_file.split("/")[-1].split(".")[0:-1])
print("output log file name: %s" % output_logfile_name)



########## RUN #################


### 1. CREATE WORK DIRECTORY AND DEFINE OTHER NAMES ==============

## Work directory
this_work_dir = os.path.join("../work/" , userinput["lores_name"].split("/")[-1].split(".")[0])
if not os.path.exists(this_work_dir):
    print("Creating work directory %s" % this_work_dir)
    LOG.append("Creating work directory %s" % this_work_dir)
    sh.mkdir(this_work_dir)
else:
    print("Work directory %s already exists" % this_work_dir)
    LOG.append("Work directory %s already exists" % this_work_dir)

### 2. LOAD IMAGES AND SAVE THEM ==============

## high resolution image
with fits.open(userinput["hires_name"]) as hdul:
    hires_img = hdul[0].data
    hires_hdr = hdul[0].header
hires_pixscale = np.abs(hires_hdr["CD1_1"])*3600
hires_hdr["XOFF"] = 0
hires_hdr["YOFF"] = 0


## Low resolution image and variance
with fits.open(userinput["lores_name"]) as hdul:
    lores_img = hdul[0].data
    lores_hdr = hdul[0].header
    #lores_rms = np.sqrt( hdul["VARIANCE"].data ) # need to add this back to the image when doing cutouts!
    lores_rms = lores_img.copy() * 0.0 + 1.0
lores_pixscale = np.abs(lores_hdr["CD1_1"])*3600
lores_pixnoise = clip(lores_img , n=3 , niter=10)["stdev"]
lores_hdr["XOFF"] = 0
lores_hdr["YOFF"] = 0
    
lores_rms = lores_rms * (lores_pixnoise)**2




## high resolution PSF
if userinput["hires_psf_type"] == "fits":
    with fits.open(os.path.join(userinput["hires_psf_name"])) as hdul:
        hires_psf = hdul[0].data
        hires_psf = hires_psf / np.nansum(hires_psf)
elif userinput["hires_psf_type"] == "psfex":
    with fits.open(os.path.join(userinput["hires_psf_name"])) as hdul:
        hires_psf = modelPSF_1D(psfcube=hdul[1].data,psfcube_h=hdul[1].header,param=21) #22
        hires_psf = hires_psf / np.nansum(hires_psf)
else:
    print("PSF type not understood.")
    LOG.append("PSF type not understood.")
    write_log(LOG , file_name = output_logfile_name)
    quit()


## low resolution PSF
if userinput["lores_psf_type"] == "fits":
    with fits.open(os.path.join(userinput["hires_psf_name"])) as hdul:
        lores_psf = hdul[0].data
        lores_psf = lores_psf / np.nansum(lores_psf)
elif userinput["lores_psf_type"] == "psfex":
    with fits.open(os.path.join(userinput["lores_psf_name"])) as hdul:
        lores_psf = modelPSF_1D(psfcube=hdul[1].data,psfcube_h=hdul[1].header,param=21) #22
        lores_psf = lores_psf / np.nansum(lores_psf)
else:
    print("PSF type not understood.")
    LOG.append("PSF type not understood.")
    write_log(LOG , file_name = output_logfile_name)
    quit()

## Pixel scale fraction
# TPhot needs an integer pixel scale fraction (lores/hires).
# However, usually this is not the case (as here with lores=0.168 and hires=0.03 pixel scale).
# The best thing to do in the future is probably to rescale the hires image to an integer multiple
# of the lores pixel scale. Because we don't want to rescale the lores image as we are measuring
# the flux on it. The hires image is only used for creating a model, hence less prone to rescaling.
# This also has implication on the cutout size of the images fed to TPhot. Specifically, the 
# lores image has to be an integer fraction in size of the hires image.
# What I do here for now is to fix the pixel scale ratio to the closest integer by rounding. Then
# use this to create new cutouts, starting with the image that is smaller in arcminutes. Basically 
# use the smallest image and create a cutout from the large image filling the excess (if there
# is any) with zeros.
pixscale_fraction = np.round(lores_pixscale/hires_pixscale)

# The hires_pixscale_new is the new pixel scale of the HIRES image such that it is an integer fraction
# of the LORES pixel scale.
hires_pixscale_new = lores_pixscale / pixscale_fraction
print("Old HIRES pixel scale: %g" % hires_pixscale)
print("New HIRES pixel scale: %g" % hires_pixscale_new)
LOG.append("Old HIRES pixel scale: %g" % hires_pixscale)
LOG.append("New HIRES pixel scale: %g" % hires_pixscale_new)

# Now we have to resample the HIRES image to the new pixel scale
print("Rescaling HIRES image . . . " , end="")
LOG.append("Rescaling HIRES image . . . ")
start_time = time.time()
hires_img_resamp = resize_psf(hires_img ,
                              input_pixel_scale = hires_pixscale,
                              output_pixel_scale = hires_pixscale_new
                             )
print(" done (in %g seconds)" % (round((time.time()-start_time)/1,2)) )
LOG.append(" done (in %g seconds)" % (round((time.time()-start_time)/1,2)) )


# Now, let's cut the LORES image to make it smaller. Later we cut any size, probably the size of the
# LORES image itself.
lores_x_size = lores_hdr["NAXIS2"]
lores_y_size = lores_hdr["NAXIS1"]
#print(lores_x_size , lores_y_size)

# Figure out the size of the HIRES image. The size has also to be an integer fraction. Since
# we resized the HIRES image, this should now be consistent.
hires_x_size = lores_x_size * pixscale_fraction # hires_hdr["NAXIS2"]
hires_y_size = lores_y_size * pixscale_fraction # hires_hdr["NAXIS1"]
#hires_x_size = hires_hdr["NAXIS2"]
#hires_y_size = hires_hdr["NAXIS1"]
#print(hires_x_size , hires_y_size)

# First cut out the LORES image. We need to change some WCS keywords. I tried to use the header update function,
# however, TPHOT crashes if that is used. Therefore I decided to do this by hand instead.
lores_cutout = Cutout2D(data=lores_img.copy(),
               position=(lores_img.shape[0]//2+0,lores_img.shape[1]//2+0),
               size=(lores_x_size , lores_y_size), # currently, this is hard coded. Change later!
               mode="partial",
               copy=True,
               fill_value=0,
               wcs=wcs.WCS(lores_hdr)
              )
lores_img_cutout = lores_cutout.data.copy()
lores_hdr_cutout = fits.PrimaryHDU().header
lores_hdr_cutout.update(lores_cutout.wcs.to_header()) # this makes TPHOT crash
#lores_hdr_cutout["CRPIX1"] = lores_hdr_cutout["CRPIX1"] - 100
#lores_hdr_cutout["CRPIX2"] = lores_hdr_cutout["CRPIX2"] + 100
#print(lores_img_cutout.shape[0] , lores_img_cutout.shape[1])

# Also cutout the uncertainty image (no need to create a header here because it is the same as for the LORES image)
lores_rms_cutout = Cutout2D(data=lores_rms.copy(),
               position=(lores_rms.shape[0]//2+0,lores_rms.shape[1]//2+0),
               size=(lores_x_size , lores_y_size), # currently, this is hard coded. Change later!
               mode="partial",
               copy=True,
               fill_value=0,
               wcs=wcs.WCS(lores_hdr)
              )
lores_rms_cutout = lores_rms_cutout.data.copy()

# Finally cut out the HIRES image. We are using here the resized/resampled image. We are cutting in pixel frame
# so we don't need the WCS here, actually. Note that also here we add a WCS to the header, however, this WCS 
# is wrong because we resampled the image. 
# TODO: need to change the WCS so it reflects the resampled image (not so easy....)
hires_cutout = Cutout2D(data=hires_img_resamp.copy(),
               position=(hires_img_resamp.shape[0]//2,hires_img_resamp.shape[1]//2),
               size=(hires_x_size,hires_y_size), # currently, this is hard coded. Change later!
               mode="partial",
               copy=True,
               fill_value=0,
               wcs=wcs.WCS(hires_hdr)
              )
hires_img_cutout = hires_cutout.data.copy()
hires_hdr_cutout = fits.PrimaryHDU().header
hires_hdr_cutout.update(hires_cutout.wcs.to_header()) # this makes TPHOT crash
#print(hires_img_cutout.shape[0] , hires_img_cutout.shape[1])


## Save the images to the work directory
#hdu = fits.PrimaryHDU(data=hires_img.copy() , header=hires_hdr.copy())
hdu = fits.PrimaryHDU(data=hires_img_cutout.copy() , header=hires_hdr_cutout.copy())
#hdu = fits.PrimaryHDU(data=hires_img_cutout.copy() , header=hires_hdr.copy())
hdul = fits.HDUList([hdu])
hdul.verify("silentfix")
hdul.writeto(os.path.join(this_work_dir , "hires.fits") , overwrite=True)

#hdu = fits.PrimaryHDU(data=lores_img.copy() , header=lores_hdr.copy())
hdu = fits.PrimaryHDU(data=lores_img_cutout.copy() , header=lores_hdr_cutout.copy())
#hdu = fits.PrimaryHDU(data=lores_img_cutout.copy() , header=lores_hdr.copy())
hdul = fits.HDUList([hdu])
hdul.verify("silentfix")
hdul.writeto(os.path.join(this_work_dir , "lores.fits") , overwrite=True)

#hdu = fits.PrimaryHDU(data=lores_rms.copy() , header=lores_hdr.copy())
hdu = fits.PrimaryHDU(data=lores_rms_cutout.copy() , header=lores_hdr_cutout.copy())
#hdu = fits.PrimaryHDU(data=lores_rms_cutout.copy() , header=lores_hdr.copy())
hdul = fits.HDUList([hdu])
hdul.verify("silentfix")
hdul.writeto(os.path.join(this_work_dir , "loresrms.fits") , overwrite=True)

print("New images saved.")
LOG.append("New images saved.")


## 3. Create PSF kernel (in high-resolution pixel scale!) ###
# We want a PSF kernel K such that PSF_lowres = K * PSF_highres where * is a convolution.
# see here for an example on making matching kernels: https://photutils.readthedocs.io/en/stable/psf_matching.html


## need to convert lowres PSF to highres pixel scale
frac = hires_pixscale / lores_pixscale
print("Interpolating LORES PSF to HIRES pixelscale. Frac = %g" % frac )
LOG.append("Interpolating LORES PSF to HIRES pixelscale. Frac = %g" % frac )
lores_psf_interp2hires = resize_psf(lores_psf, input_pixel_scale=lores_pixscale , output_pixel_scale=hires_pixscale,order=3)
lores_psf_interp2hires = lores_psf_interp2hires / np.nansum(lores_psf_interp2hires)

## save
hdu = fits.PrimaryHDU(data=lores_psf_interp2hires)
hdul = fits.HDUList([hdu])
hdul.writeto(os.path.join(this_work_dir , "lores_psf_interp2hires.fits") , overwrite=True )


## prepare target PSF
# this is just the lowres PSF in the hires pixel scale
#target_psf = lores_psf_interp2hires.copy()
target_psf = Cutout2D(data=lores_psf_interp2hires.copy(),
               position=(lores_psf_interp2hires.shape[0]//2,lores_psf_interp2hires.shape[0]//2),
               size=(101,101), # currently, this is hard coded. Change later!
               mode="partial",
                copy=True
              ).data
target_psf = target_psf / np.nansum(target_psf)

## prepare the source PSF
# this should be the hires PSF. However, the cutout needs to be the same
# size as the target PSF. We therefore need to extrapolate it.
# First try by just cutting it out and replace the NaNs with 0.
source_psf = Cutout2D(data=hires_psf.copy(),
               position=(hires_psf.shape[0]//2,hires_psf.shape[0]//2),
               size=target_psf.shape,  # currently, this is hard coded. Change later!
               mode="partial",
                copy=True
              ).data
source_psf[np.isnan(source_psf)] = 0
source_psf = source_psf / np.nansum(source_psf)


## Now compute the Kernel
kernel_HIRES_to_LOWRES = create_matching_kernel(source_psf=source_psf.copy(),
                                                target_psf=target_psf.copy(),
                                               window=TopHatWindow(0.4)) # this is currently hard coded. Change later?
kernel_HIRES_to_LOWRES = kernel_HIRES_to_LOWRES / np.nansum(kernel_HIRES_to_LOWRES)


## Save
hdu = fits.PrimaryHDU(data=kernel_HIRES_to_LOWRES)
hdul = fits.HDUList([hdu])
hdul.writeto(os.path.join(this_work_dir , "kernel.fits") , overwrite=True )


## Plot the PSFs for checking
if userinput["make_plots"]:
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)

    ax1.imshow(hires_psf , norm=ImageNormalize(stretch=LogStretch()), vmax=0.03)
    ax2.imshow(lores_psf , norm=ImageNormalize(stretch=LogStretch()) , vmax=0.03)
    ax3.imshow(lores_psf_interp2hires , norm=ImageNormalize(stretch=LogStretch()) , vmax=0.01)

    ax1.set_title("HIRES PSF")
    ax2.set_title("LORES PSF")
    ax3.set_title("LORES PSF (HIRES pixel scale)")

    plt.savefig(os.path.join(this_work_dir,"psf_check.pdf") , bbox_inches="tight")
    plt.show()
    
    

## Plot the source/target PSF as well as the Kernel
if userinput["make_plots"]:
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2,2,1) # target PSF
    ax2 = fig.add_subplot(2,2,2) # source PSF
    ax3 = fig.add_subplot(2,2,3) # Kernel
    ax4 = fig.add_subplot(2,2,4) # cuts

    # target, source, and Kernel images
    ax1.imshow( target_psf , norm=ImageNormalize(stretch=LogStretch()) , vmin=0 , vmax=0.01)
    ax2.imshow( source_psf , norm=ImageNormalize(stretch=LogStretch()) , vmin=0 , vmax=0.01)
    ax3.imshow( kernel_HIRES_to_LOWRES , norm=ImageNormalize(stretch=LogStretch()) , vmin=0 , vmax=0.01)


    # Plot the cuts
    xx = np.arange(-(target_psf.shape[0]//2) , (target_psf.shape[0]//2+1) , 1)*hires_pixscale
    ax4.plot(xx, target_psf[target_psf.shape[0]//2,] , label="Target PSF")

    xx = np.arange(-(source_psf.shape[0]//2) , (source_psf.shape[0]//2+1) , 1)*hires_pixscale
    ax4.plot(xx, source_psf[source_psf.shape[0]//2,] , label="Source PSF")

    xx = np.arange(-(kernel_HIRES_to_LOWRES.shape[0]//2) , (kernel_HIRES_to_LOWRES.shape[0]//2+1) , 1)*hires_pixscale
    ax4.plot(xx,  kernel_HIRES_to_LOWRES[kernel_HIRES_to_LOWRES.shape[0]//2,] , label="Kernel")

    lores_conv_psf = convolve(source_psf,kernel_HIRES_to_LOWRES)
    lores_conv_psf = lores_conv_psf / np.nansum(lores_conv_psf)
    xx = np.arange(-(lores_conv_psf.shape[0]//2) , (lores_conv_psf.shape[0]//2+1) , 1)*hires_pixscale
    ax4.plot(xx, lores_conv_psf[lores_conv_psf.shape[0]//2,] , dashes=(5,3) , label="source x kernel")

    ax1.set_title("Target PSF")
    ax2.set_title("Source PSF")
    ax3.set_title("Kernel")

    #ax4.set_yscale("log")
    #ax4.set_ylim(1e-9,1e-1)
    ax4.set_ylim(0,0.0025)
    ax4.set_xlabel("Arcsec")
    ax4.set_ylabel("flux")
    ax4.legend(loc="best")
    ax4.set_title("Cuts")

    plt.savefig(os.path.join(this_work_dir,"kernel_check.pdf") , bbox_inches="tight")
    plt.show()


### 4. Create catalog for hires image. ######
# This catalog must include:
# id x_obj y_obj xmin ymin xmax ymax loc_bckg obj_flux
# Because the images are background subtracted, loc_bckg=0 for all sources
# The catalog can be created with SExtractor


## Run SExtractor on HIRES image

# a) Copy SExtractor config file
config_default = "../sextractor_input/default.conf.interactive"
config_this_process = os.path.join(this_work_dir,"sextractor.conv")
cmd = "cp " + config_default + " " + config_this_process
subprocess.run(cmd, shell=True)


# b) Adjust configuration file
sex_output_cat_hr_file = os.path.join(this_work_dir,"sextractor.cat")
PARS = get_default_parfile()
PARS["CATALOG_NAME"] = sex_output_cat_hr_file
PARS["PARAMETERS_NAME"] = "../sextractor_input/sex.par"
PARS["FILTER_NAME"] = "../sextractor_input/g2.8.conv"
PARS["STARNNW_NAME"] = "../sextractor_input/default.nnw"
PARS["CHECKIMAGE_NAME"] = os.path.join(this_work_dir,"seg.fits")
PARS["CHECKIMAGE_TYPE"] = "SEGMENTATION"

PARS["PHOT_APERTURES"] = ','.join(map(str, get_apertures(hires_pixscale,aperturelist_arcsec=[3.0]).tolist())) # Note: if number of apertures are changes, also change sex.par file!!!
PARS["PIXEL_SCALE"] = hires_pixscale
PARS["DEBLEND_MINCONT"] = 0.1
PARS["DEBLEND_NTHRESH"] = 32
PARS["DETECT_MINAREA"] = 20
PARS["DETECT_THRESH"] = 2.0
PARS["MAG_ZEROPOINT"] = userinput["hr_zeropoint"]
PARS["ANALYSIS_THRESH"] = 1.5
PARS["BACKPHOTO_TYPE"] = "LOCAL"  #GLOBAL
PARS["SEEING_FWHM"] = 0.1 # Doesn't really matter for SExtractor in Arcsec
PARS["CLEAN_PARAM"] = 1.0


for key in PARS.keys():
    replace_in_file(filename = config_this_process,
                   old_string = "*" + key + "*",
                   new_string = str(PARS[key]))


# c) Run SExtractor and load catalog
start_time = time.time()
print("running SExtractor . . .",end="")
LOG.append("running SExtractor . . .")
cmd = "%s %s -c %s " % (userinput["sex_command"],
                        os.path.join(this_work_dir , "hires.fits"),
                        config_this_process)
print(cmd)
LOG.append(cmd)
subprocess.run(cmd , shell=True)
print(" done (in %g minutes)" % (round((time.time()-start_time)/60,2)) )
LOG.append(" done (in %g minutes)" % (round((time.time()-start_time)/60,2)))

# d) Load catalog and create region file
sexcat = ascii.read(sex_output_cat_hr_file)
print("%g objects extracted" % len(sexcat))
LOG.append("%g objects extracted" % len(sexcat))

print("%g objects to fit" % len(sexcat))
LOG.append("%g objects to fit" % len(sexcat))


# write DS9 region file
with open( os.path.join(this_work_dir,"ds9.reg") , "w") as f:
    f.write("# Region file format: DS9 version 4.1\n")
    f.write("global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
    f.write("fk5\n")
    for iii in range(len(sexcat)):
        f.write("circle(%3.10f,%3.10f,0.5\")\n" % (sexcat["ALPHA_J2000"][iii],sexcat["DELTA_J2000"][iii]) )

# e) Load segmentation map and save
with fits.open(os.path.join(this_work_dir,"seg.fits")) as hdul:
    segmap = hdul[0].data
    segmap_hdr = hdul[0].header
segmap_hdr["XOFF"] = 0
segmap_hdr["YOFF"] = 0
hdu = fits.PrimaryHDU(data=segmap.copy() , header=segmap_hdr.copy())
hdul = fits.HDUList([hdu])
hdul.writeto( os.path.join(this_work_dir,"seg.fits") , overwrite=True)
    
## Create final catalog
print("Writing TPHOT table for HIRES image . . . " , end="")
LOG.append("Writing TPHOT table for HIRES image . . . ")
hires_cat = Table([sexcat["NUMBER"],
                   sexcat["X_IMAGE"],
                   sexcat["Y_IMAGE"],
                   sexcat["XMIN_IMAGE"],
                   sexcat["YMIN_IMAGE"],
                   sexcat["XMAX_IMAGE"],
                   sexcat["YMAX_IMAGE"],
                   np.repeat(0,len(sexcat)), # background subtracted
                   sexcat["FLUX_AUTO"],
                   sexcat["ISOAREA_IMAGE"]
                  ],
                  names=["SOURCE_ID","X_CENTER","Y_CENTER","X_MIN","Y_MIN","X_MAX","Y_MAX","BACKGROUND","FLUX_TOT","ISOAREA"],
                 dtype=["int","f","f","f","f","f","f","f","f","f"])

#hires_cat.write(os.path.join(outdir , "%s_tphot.cat" % label) , format="ascii.commented_header" , overwrite=True)
hires_cat.write(os.path.join(this_work_dir , "tphotcat.fits") , format="fits" , overwrite=True)
print("done.")
LOG.append("done.")


## Write update.sh file
#with open(os.path.join(outdir,"update.sh") , "w") as f:
#    f.write("rsync -auvr * afaisst@vmjointproc01.ipac.caltech.edu://stage/irsa-jointproc-data02/TPHOT/test_sim/%s_input/" % label)
    

## Create template for TPHOT parameter file
create_TPHOT_template(filename=os.path.join(this_work_dir,"tphot.param"),
                      workdir=this_work_dir,
                      pixscaleratio=pixscale_fraction,
                     dilation=userinput["perform_dilation"],
                      zeropoint = 27.0
                     )


## 5. RUN TPHOT

start_time = time.time()
print("running TPHOT . . .",end="")
LOG.append("running TPHOT . . .")
cmd = "TPhot --configuration %s/tphot.param --workdir ." % (this_work_dir)
print(cmd)
LOG.append(cmd)
subprocess.run(cmd , shell=True)
print(" done (in %g minutes)" % (round((time.time()-start_time)/60,2)) )
LOG.append(" done (in %g minutes)" % (round((time.time()-start_time)/60,2)) )

# --log_level DEBUG


write_log(LOG , file_name=output_logfile_name)