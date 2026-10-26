import io
import os
import glob
from astropy.io import fits
from astropy.table import Table
from watroo import AtrousTransform, B3spline, utils
import csv
import copy
from astropy.time import Time
import astropy.visualization as visu
import astropy.constants
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.collections as mcol
from scipy.ndimage import label, find_objects
from scipy.signal import correlate2d
import imageio
import cv2
from rectify import rectify
from skimage.registration import phase_cross_correlation
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from geometry import Point, Line
import subprocess
from event import Event
from campfires import Image, Stack, Sequence
import time
from process_aia import process_aia
from make_carrington_stack_campfires import make_carrington_stack_campfires

include_AIA = False # set to True if AIA should be included and to False if only HRIEUV is needed

hrieuv_data_path = r'/home/francisv/CLOSE-UP/new_base_code_20250911/test_run_20200530/data/hrieuv/'
#hrieuv_data_path = r'/home/francisv/CLOSE-UP/new_base_code_20250911/test_run_20221012/data/hrieuv/'

if (include_AIA): # note: make sure that the HRIEUV files at hrieuv_data_path have been aligned to AIA
    aia_data_path  = r'/home/francisv/CLOSE-UP/new_base_code_20250911/test_run_20200530/data/aia/171/'
    aia_wavelength = 171 # can be 171, 94, 131, 193, 211, 335. In that case, process_aia and make_carrington_stack_campfires should be run for every wavelength
    data_paths     = (hrieuv_data_path, aia_data_path)
else:
    data_paths = hrieuv_data_path
print(data_paths)


def main_antoine():
    t0              = time.time()
    data_paths      = "/scratch/dolliou/test_campfire/data/2023_10_06_org_carmaps/"
    print("starting Sequence")
    seq = Sequence(data_paths, fov = None)
    print("starting extract_events")
    seq.extract_events(sigma = 3, dmin = 2, vmin = 2, compute_excluded=False)

    t              = time.time() - t0
    print(f"elapsed time {t}")

    print("starting events_totable")
    seq.events_totable(output_filename = 'EvtCatalog_20231006_excludefalse')
    print("starting plot_statistics")
    seq.plot_statistics()
    print("starting plot_events")
    seq.plot_events()
    print("starting events_tofits")
    seq.events_tofits()
    print("starting make_movies")
    seq.make_movies(first_n = 10)
    print("End.label")

def main(data_paths):
    
    # select the input parameters for your run of this program:
    hrieuv_carrington_path = r'/home/francisv/CLOSE-UP/new_base_code_20250911/test_run_20200530/carrington/hrieuv/'
   #hrieuv_carrington_path = r'/home/francisv/CLOSE-UP/new_base_code_20250911/test_run_20221012/carrington/hrieuv/'
    if (include_AIA):
        aia_carrington_path = r'/home/francisv/CLOSE-UP/new_base_code_20250911/test_run_20200530/carrington/aia/171/'
    
    # 20200530   
    start_time = r'2020-05-30T14:54:00'
    end_time   = r'2020-05-30T14:58:20'
    shape      = (2400, 2400)  # shape of FOV in Carrington coordinates
    lonlims    = (248, 289)    # check lonlims and latlims in JHelioviewer, the shape remains mostly the same
    latlims    = (-12, 26)
    
    # 20221012   
   #start_time = r'2022-10-12T05:25:00'
   #end_time   = r'2022-10-12T06:09:28'
   #shape      = (2400, 2400)      # shape of FOV in Carrington coordinates
   #lonlims    = (219.43, 240.18)  # manually finalsed vlaues using JHelioviewer
   #latlims    = (-11.50, 9.25)    # manually finalsed vlaues using JHelioviewer
   
    # generate the stack of HRIEUV images
    make_carrington_stack_campfires(r'HRIEUV', hrieuv_data_path, hrieuv_carrington_path, shape, lonlims, latlims)

    if (include_AIA):
        # get AIA level 1.5 data (can be repeated for several wavelengths)
        # Note: take into account the light travel time when defining the AIA start and end time as compared to HRIEUV!
        process_aia(aia_data_path, start_time, end_time, aia_wavelength) 

        # generate the stack of AIA images (can be repeated for several wavelengths)
        make_carrington_stack_campfires(r'AIA', aia_data_path, aia_carrington_path, shape, lonlims, latlims)

    # run EUV brightening detection on the Carrington files
    if (include_AIA):
        data_paths = (hrieuv_carrington_path, aia_carrington_path)
    else:
        data_paths = hrieuv_carrington_path
    print(data_paths)

    # detect EUV brightenings and produce output
    print("starting Sequence")
    seq = Sequence(data_paths, fov = None)
    print("starting extract_events")
    seq.extract_events(sigma = 6, dmin = 0, vmin = 1)
    print("starting events_totable")
    seq.events_totable(output_filename = 'EvtCatalog_20200530')
   #seq.events_totable(output_filename = 'EvtCatalog_20221012')
    print("starting plot_statistics")
    seq.plot_statistics()
    print("starting plot_events")
    seq.plot_events()
    print("starting events_tofits")
    seq.events_tofits()
    print("starting make_movies")
    seq.make_movies(first_n = 10)
    print("End.label")


if __name__ == '__main__':
    # main(data_paths)
    main_antoine()