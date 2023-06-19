import glob
import rectify
from astropy import visualization
from astropy.io import fits
import matplotlib.pyplot as plt
import sunpy.map
import aiapy.calibrate
import numpy as np
import astropy


path = 'C:\\Users\\Frédéric Auchère\\Desktop\\test\\'

files = glob.glob(path + 'solo_L2_eui-hrieuv174*.fits')
euifile = files[0]

path = 'C:\\Users\\Frédéric Auchère\\Desktop\\test\\'

files = glob.glob(path + 'aia*.fits')
aiafile = files[0]

euimap = fits.getdata(euifile)
header = fits.getheader(euifile)
euimap = np.float32(euimap)

t1 = astropy.time.Time("2020-05-30 14:30:00")
t2 = astropy.time.Time("2020-05-30 15:30:00")
pointing_table = aiapy.calibrate.util.get_pointing_table(t1, t2)
with fits.open(aiafile) as hdu:
    hdu[1].verify('fix')
    aiamap = np.float32(hdu[1].data)
    aiaheader = hdu[1].header
smap = sunpy.map.Map((aiamap, aiaheader))
smap = aiapy.calibrate.update_pointing(smap, pointing_table)

aiaheader['CRPIX1'] = smap.meta['CRPIX1'] + 1
aiaheader['CRPIX2'] = smap.meta['CRPIX2'] + 1
crln_obs = 244.13506366263948
crlt_obs = -0.8083424851434228
dsun_obs = 151659273485.41815
print(aiaheader['CRLN_OBS'] - crln_obs)
print(aiaheader['CRLT_OBS'] - crlt_obs)
print(aiaheader['DSUN_OBS'] - dsun_obs)
aiaheader['CRLN_OBS'] = crln_obs
aiaheader['CRLT_OBS'] = crlt_obs
aiaheader['DSUN_OBS'] = dsun_obs

cmap = plt.get_cmap('gray')

stretch = visualization.LinearStretch()

shape = (2400, 2400)
lonlims = (248.8, 287.8)
latlims = (-11.5, 27.5)

outpath = path

nr, nc1, nc2, ncd, nrt = 21, 21, 21, 1, 5
solar_r = np.linspace(1.0, 1.02, nr)
crval1 = np.linspace(-8.0, -2.0, nc1)
crval2 = np.linspace(-20, -16.0, nc2)
crota = np.linspace(0.02, 0.07, nrt)

corr = np.zeros((nr, nc1, nc2, ncd, nrt))
cdelt = [0]

ocrval1 = header['CRVAL1']
ocrval2 = header['CRVAL2']
ocrota = header['CROTA']
ocdelt = header['CDELT1']

for i, sr in enumerate(solar_r):

    spherical = rectify.CarringtonTransform(aiaheader, radius_correction=sr)
    spherizer = rectify.Rectifier(spherical)
    aiacarrington = spherizer(aiamap, shape, lonlims, latlims, opencv=True, order=1)

    norm = np.sqrt(aiacarrington)
    print(i)

    for j, crv1 in enumerate(crval1):
        for k, crv2 in enumerate(crval2):
            for l, cdlt in enumerate(cdelt):
                for m, crot in enumerate(crota):

                    header['CRVAL1'] = ocrval1 + crv1
                    header['CRVAL2'] = ocrval2 + crv2
                    header['CDELT1'] = ocdelt + cdlt
                    header['CDELT2'] = header['CDELT1']
                    header['CROTA'] = ocrota + crot

                    spherical = rectify.CarringtonTransform(header, radius_correction=sr)
                    spherizer = rectify.Rectifier(spherical)
                    carrington = spherizer(euimap, shape, lonlims, latlims, opencv=True, order=1)

                    diff = (aiacarrington - carrington)/norm
                    corr[i, j, k, l, m] = np.std(diff[(carrington > 0) & (carrington < 1310)])

i, j, k, l, m = np.unravel_index(np.argmin(corr, axis=None), corr.shape)
print(solar_r[i], crval1[j], crval2[k], cdelt[l], crota[m])
