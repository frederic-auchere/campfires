import glob
from rectify import rectify
from astropy.io import fits
import astropy.constants
import astropy.time
import os

outbase = r'C:\archive\CampFires'

shape = (2172, 2172)
lonlims = (210, 265)
latlims = (-70, -30)

path = r'C:\archive\eui\releases\2022\03\18'

files = glob.glob(os.path.join(path, 'solo_L2_eui-hrieuv174-image*.fits'))

solar_r = 1.004

for i, f in enumerate(files):

    euimap = fits.getdata(f)
    header = fits.getheader(f, 1)

    spherical = rectify.CarringtonTransform(header, radius_correction=solar_r)
    spherizer = rectify.Rectifier(spherical)
    carrington = spherizer(euimap, shape, lonlims, latlims, opencv=False, order=2)

    out = os.path.join(outbase, os.path.basename(f))
    out = out.replace('_L2_', '_L3_')
    out = out.replace('.fits', '_carrington.fits')

    header['MAPPINGR'] = solar_r*astropy.constants.R_sun.value
    header['CACRPIX1'] = (shape[0] + 1)/2
    header['CACRPIX2'] = (shape[1] + 1)/2
    header['CACRVAL1'] = (lonlims[1] + lonlims[0])/2
    header['CACRVAL2'] = (latlims[1] + latlims[0])/2
    header['CACDELT1'] = (lonlims[1] - lonlims[0])/(shape[0]-1)
    header['CACDELT2'] = (latlims[1] - latlims[0])/(shape[1]-1)
    fits.writeto(out, carrington, header=header, overwrite=True)
