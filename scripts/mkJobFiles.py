### Script to create job files ###
# There is a lot hard-coded in this script. This script should just
# give you an idea on how to do it.
import os
import sh
import json
import numpy as np
import glob


## Some paths -----

# base name of simulation
base_name_of_simulation = ""

# output path (where json job files are saved)
job_files_output_path = "/stage/irsa-jointproc-data02/TPHOT/run_test_Sep25/run/jobs/"

# main tractor path
tractor_main_path = "/stage/irsa-jointproc-data03/TRACTOR/run_acs_jsp_full_SB4_Jul20/run/"

# PSF directories
hr_psf_main_path = "/stage/irsa-jointproc-data03/ACS_COSMOS/from_irsa/sextractor/psf/"
hr_psf_name_fixed = "HSC-I-9813-5_4-2812_psf.fits"
lr_psf_main_path = "/stage/irsa-jointproc-data02/PSFex/allcalexp_psfex_pipeline/output_variableMag/psfs/"
hires_psf_type = "fits"
lores_psf_type = "psfex"

# others
lr_zeropoint = 27
hr_zeropoint = 25.94734
tractor_work_main_path = "/stage/irsa-jointproc-data03/TRACTOR/run_acs_jsp_full_SB4_Jul20/run/work/"
tractor_prefix = "SB4"
sex_command = "/usr/bin/sextractor"



## Get all directories -----
dirs = glob.glob( os.path.join( tractor_main_path , "cutouts" , "calexp-HSC-I*" ) )
# '/stage/irsa-jointproc-data03/TRACTOR/run_acs_jsp_full_SB4_Jul20/run/cutouts/calexp-HSC-I-9813-8_3'

## Create the job files
cc = 1 # counter
for dd,this_dir in enumerate(dirs[0:1]): # ***** CHANGE THIS BEFORE RUNNING *****
    
    # get file in this directory
    files = glob.glob( os.path.join( this_dir , "*_calexp-HSC-I-*_acs_I_mosaic_30mas_sci.fits" ) )
    # /stage/irsa-jointproc-data03/TRACTOR/run_acs_jsp_full_SB4_Jul20/run/cutouts/calexp-HSC-I-9812-0_6/0012_calexp-HSC-I-9812-0_6-4564_acs_I_mosaic_30mas_sci.fits
    # 0013_calexp-HSC-I-9812-0_1.fits

    for ff,this_file in enumerate(files[0:1]): # ***** CHANGE THIS BEFORE RUNNING *****
        this_job = dict()

        this_job["base_name"] = base_name_of_simulation

        # get base name
        tilenbr = this_file.split("/")[-1].split("_")[0]
        acsnbr = this_file.split("/")[-1].split("-")[-1].split("_")[0]
        base_name = this_dir.split("/")[-1]
        tract = base_name.split("-")[3]

        # get image names and check
        this_job["hires_name"] = this_file
        this_job["lores_name"] = os.path.join( this_dir , "%s_%s.fits" % (tilenbr , base_name) )
        if not os.path.exists(this_job["hires_name"]):
            print("HIRES image %s does not exist. ABORT" % this_job["hires_name"])
            quit()
        if not os.path.exists(this_job["lores_name"]):
            print("LORES image %s does not exist. ABORT" % this_job["lores_name"])
            quit()

        # get PSFs and check
        this_job["hires_psf_name"] = os.path.join( hr_psf_main_path , hr_psf_name_fixed )
        this_job["hires_psf_type"] = hires_psf_type
        this_job["lores_psf_name"] = os.path.join( lr_psf_main_path , "%s-%s_%s.psf" % (base_name , tract , acsnbr) )
        this_job["lores_psf_type"] = lores_psf_type
        if not os.path.exists(this_job["hires_psf_name"]):
            print("HIRES PSF %s does not exist. ABORT" % this_job["hires_psf_name"])
            quit()
        if not os.path.exists(this_job["lores_psf_name"]):
            print("LORES PSF %s does not exist. ABORT" % this_job["lores_psf_name"])
            quit()

        # add the rest
        this_job["lr_zeropoint"] = lr_zeropoint
        this_job["hr_zeropoint"] = hr_zeropoint
        this_job["make_plots"] = "true"
        this_job["perform_dilation"] = "true"
        this_job["reuse_dilationmap"] = "true"
        this_job["compare_to_tractor"] = "true"
        this_job["tractor_main_path"] = tractor_work_main_path
        this_job["tractor_prefix"] = tractor_prefix
        this_job["sex_command"] = sex_command

        # save the json file
        with open(os.path.join(job_files_output_path , "job_%g.json" % cc ), 'w') as outfile:
            json.dump(this_job, outfile)

        cc = cc + 1

