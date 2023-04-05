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
from geometry import Point, Line
import subprocess
from event import Event


def parabolic(cc):
    cy, cx = np.unravel_index(np.argmax(cc, axis=None), cc.shape)
    if cx == 0 or cy == 0 or cx == cc.shape[1] - 1 or cy == cc.shape[0] - 1:
        return cx, cy
    else:
        xi = [cx - 1, cx, cx + 1]
        yi = [cy - 1, cy, cy + 1]
        ccx2 = cc[[cy, cy, cy], xi] ** 2
        ccy2 = cc[yi, [cx, cx, cx]] ** 2

        xn = ccx2[2] - ccx2[1]
        xd = ccx2[0] - 2 * ccx2[1] + ccx2[2]
        yn = ccy2[2] - ccy2[1]
        yd = ccy2[0] - 2 * ccy2[1] + ccy2[2]

        if xd != 0:
            dx = xi[2] - (xn / xd + 0.5)
        else:
            dx = cx
        if yd != 0:
            dy = yi[2] - (yn / yd + 0.5)
        else:
            dy = cy

        return dx, dy


class Stack:
    """
    A stack of images
    """

    def __init__(self, files=[]):
        self.images = [Image(self, file, idx=i) for i, file in enumerate(files)]
        self.n_images = len(self.images)
        self.events = []
        self.excluded = []
        self.min = None
        self.max = None
        self.mean = None
        self.variance = None
        self.relative_variance = None

    def __iter__(self):
        for image in self.images:
            yield image

    def __add__(self, image):

        self.images.append(image, idx=self.n_images)
        self.n_images += 1

    def __len__(self):

        return len(self.images)

    def compute_statistics(self):
        for image in self:
            data, _ = image.get()
            if self.mean is None:
                self.mean = np.copy(data)
            else:
                self.mean += data
            if self.max is None:
                self.max = np.copy(data)
            else:
                gd = data > self.max
                self.max[gd] = data[gd]
            if self.min is None:
                self.min = np.copy(data)
            else:
                gd = data < self.min
                self.min[gd] = data[gd]
        self.mean /= len(self.images)
        self.variance = np.zeros_like(self.mean)
        for image in self:
            data, _ = image.get()
            self.variance += (data - self.mean) ** 2
        gd = self.mean != 0
        self.relative_variance = np.copy(self.variance)
        self.relative_variance[gd] /= self.mean[gd]

        mask = self.min <= 0
        self.relative_variance[mask] = 0
        self.min[mask] = 0
        self.max[mask] = 0
        self.variance[mask] = 0
        self.mean[mask] = 0

    def get_variance(self):
        if self.variance is None:
            self.compute_statistics()
        return self.variance

    def get_relative_variance(self):
        if self.relative_variance is None:
            self.compute_statistics()
        return self.relative_variance

    def get_min(self):
        if self.min is None:
            self.compute_statistics()
        return self.min

    def get_max(self):
        if self.max is None:
            self.compute_statistics()
        return self.max

    def get_mean(self):
        if self.mean is None:
            self.compute_statistics()
        return self.mean

    def blobs3d(self, n_levels=2, sigma=1, detection_method='wavelets', saturation=True):
        blobs = []
        for image in self:
            blobs.append(
                image.blobs2d(n_levels=n_levels, sigma=sigma, detection_method=detection_method, saturation=saturation))
        return ma.masked_array(blobs)

    def extract_events(self, n_levels=2, sigma=1, dmin=0, vmin=0, vmax=None, detection_method='wavelets',
                       saturation=True):

        blobs = self.blobs3d(n_levels=n_levels, sigma=sigma, detection_method=detection_method, saturation=saturation)
        if vmax is None:
            vmax = blobs.size

        regions, nregions = label(~blobs.mask)
        slices = find_objects(regions)

        self.events = []

        for i, s in enumerate(slices):
            blob = ma.masked_array(blobs.data[s], mask=regions[s] != i + 1)
            if s[0].stop - s[0].start < dmin:
                self.excluded.append(Event(self, s, blob, i))
                # blobs.mask[s][~blob.mask] = True
            elif not vmin <= (~blob.mask).sum() <= vmax:
                self.excluded.append(Event(self, s, blob, i))
                # blobs.mask[s][~blob.mask] = True
            else:
                self.events.append(Event(self, s, blob, i))

    def extract_background(self, n_levels=2, sigma=1, dmin=0, vmin=0, vmax=None, detection_method='wavelets'):

        blobs = self.blobs3d(n_levels=n_levels, sigma=sigma, detection_method=detection_method)
        if vmax is None:
            vmax = blobs.size

        regions, nregions = label(~blobs.mask)
        slices = find_objects(regions)

        plt.imshow(~blobs[10].mask, origin="lower")


class Image:

    def __init__(self, parent, file, idx=None):

        self.dn_per_photoelectron = None
        self.file = file
        self.idx = idx
        self.parent_stack = parent
        self.header = None

    def aia_gain(self, wave):
        """
        Return the AIA cameras gain in e-/DN
        Input: wave in ANGSTROM (a string)
        (Boerner et al.)
        """

        aia_gain = dict()
        aia_gain[131] = aia_gain[335] = 17.6
        aia_gain[193] = aia_gain[211] = aia_gain[94] = 18.3
        aia_gain[171] = aia_gain[304] = 17.7
        return aia_gain[wave]

    def get(self, photons=True):

        with fits.open(self.file) as hdu:
            if 'EXTEND' in hdu[0].header:
                ext = 1 if hdu[0].header['EXTEND'] else 0
            else:
                ext = 0
            if self.header is None:
                self.header = hdu[ext].header
            try:
                data = np.float32(hdu[ext].data)
            except:
                hdu[ext].verify('fix')
                data = np.float32(hdu[ext].data)

        if not ("inpainted" in self.file) and photons:
            if 'HRI_EUV' in self.header['DETECTOR']:
                self.dn_per_photoelectron = 5.27
                self.header['RDNOISE'] = 1.5 / self.dn_per_photoelectron
                data *= self.header['XPOSURE']  # image remultiplied by exposure time
                data /= self.dn_per_photoelectron
            elif 'AIA' in self.header['TELESCOP']:
                self.dn_per_photoelectron = self.aia_gain(self.header["WAVELNTH"])
                data /= self.dn_per_photoelectron
                self.header['RDNOISE'] = 1.15 / self.dn_per_photoelectron

        return data, self.header

    def noise(self, image):
        img = np.copy(image)
        img[img < 0] = 0
        return np.sqrt(img + self.header['RDNOISE'] ** 2)

    def blobs2d(self, sigma=1, n_levels=2, detection_method='wavelets', saturation=True):

        img, hdr = self.get()

        data = ma.masked_array(img, mask=True)

        if detection_method == 'wavelets':

            transform = AtrousTransform(scaling_function_class=B3spline)
            coeffs = transform(img - np.median(img[img > 0]), level=n_levels)
            if saturation:
                gd = np.logical_and(img > 0, img < 620)
            else:
                gd = img > 0
            sigma_s = self.noise(img)
            sigma_s[~gd] = 0
            if sigma > 0:
                dns = [sigma,] * n_levels
                for coeff, d, se in zip(coeffs.data[0:n_levels], dns,
                                        coeffs.scaling_function.sigma_e()[0:n_levels]):
                    data.mask[coeff >= (d * sigma_s * se)] = False
                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                # bad = 1 - cv2.erode(1 - np.uint8(gd), kernel, iterations=3)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                bad = cv2.erode(np.uint8(gd), kernel, iterations=8)
                data.mask[bad == 0] = True
            else:
                dns = [np.abs(sigma)] * n_levels
                for coeff, d, se in zip(coeffs[0:n_levels], dns,
                                        transform.scaling_function_class(2).sigma_e()[0:n_levels]):
                    data.mask[coeff >= (d * sigma_s * se)] = False
                data.mask = ~data.mask

            # !TODO get rid of this big hack to avoid issues at the edges
            data.mask[0, :] = True
            data.mask[:, 0] = True
            data.mask[-1, :] = True
            data.mask[:, -1] = True

        else:
            raise ValueError("Unknown detection method")

        return data

    def make_rgb(self,
                 denoise=False,
                 events=None,
                 fov=Ellipsis,
                 flatten=False,
                 color_by=None,
                 interval=astropy.visualization.MinMaxInterval(),
                 stretch=astropy.visualization.LinearStretch()):
        data, _ = self.get()
        if denoise:
            noise = np.copy(data)
            noise = np.ones_like(data)
            gd = np.logical_and(data > 0, data < 620)
            noise[gd] = 2 * np.sqrt(data[gd] + 3 / 8)
            data = utils.enhance(data,
                                 noise,
                                 denoise=[-3, -2])
        transform = stretch + interval
        if fov is None:
            fov = (slice(0, data.shape[0]),
                   slice(0, data.shape[1]))
        data = np.uint8(transform(data[fov]) * 255)
        rgb = np.stack((data, data, data), axis=2)
        if events is not None:
            for ev in events:
                if ev.isinframe(fov, fnum=self.idx) or flatten:
                    if flatten:
                        contour = ev.rgb_outline
                    else:
                        contour = ev.rgb_contour[self.idx - ev.slc[0].start]
                    y1 = fov[0].start - ev.slc[1].start
                    if y1 < 0:
                        y1 = 0
                    y2 = fov[0].stop - ev.slc[1].start
                    if y2 > contour.shape[0]:
                        y2 = contour.shape[0]
                    x1 = fov[1].start - ev.slc[2].start
                    if x1 < 0:
                        x1 = 0
                    x2 = fov[1].stop - ev.slc[2].start
                    if x2 > contour.shape[1]:
                        x2 = contour.shape[1]

                    yp1 = y1 - (fov[0].start - ev.slc[1].start)
                    yp2 = y2 - (fov[0].start - ev.slc[1].start)
                    xp1 = x1 - (fov[1].start - ev.slc[2].start)
                    xp2 = x2 - (fov[1].start - ev.slc[2].start)

                    mask = contour[y1:y2, x1:x2, :].sum(axis=2) > 0
                    for c in range(3):
                        rgb[yp1:yp2, xp1:xp2, c][mask] = contour[y1:y2, x1:x2, c][mask]
        return rgb

class Sequence:

    def __init__(self, paths, fov=None, master=0, suffix='*.fits',
                 outpath=None, detection_method='wavelets'):

        self.dmin = None
        self.n_levels = None
        self.sigma = None
        self.master = master
        self.paths = [paths] if type(paths) is str else paths
        if outpath is None: self.outpath = self.paths[self.master]
        self.dcrval1 = None
        self.dcrval2 = None
        self.fov = fov
        self.suffix = suffix
        self.stacks = []
        self.masterstack = None
        self.detection_method = detection_method

        self.build_multiplets()

    def print_multiplets(self):
        images = [stk.images for stk in self.stacks]
        with open("multiplets.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for imgs in zip(*images):
                files = [os.path.basename(i.file) for i in imgs]
                writer.writerow(files)

    def build_multiplets(self):

        if len(self.paths) > 1:
            other_files = []
            other_dates = []
            other_distances = []
            for path in self.paths[1:]:
                files = glob.glob(os.path.join(path, self.suffix))
                other_files.append(files)
                if 'DATE-OBS' in fits.getheader(files[0]):
                    ext = 0
                else:
                    ext = 1
                other_headers = [fits.getheader(of, ext) for of in files]
                other_dates.append(Time([oh['DATE-OBS'] for oh in other_headers]))
                other_distances.append([oh['DSUN_OBS'] for oh in other_headers])

        multiplets = []
        hri_files = glob.glob(os.path.join(self.paths[0], self.suffix))
        mjds = [Time(fits.getheader(f)['DATE-OBS']).mjd for f in hri_files]
        hri_files = [f for _, f in sorted(zip(mjds, hri_files))]

        for hri_file in hri_files:
            hri_header = fits.getheader(hri_file)
            hri_distance = hri_header['DSUN_OBS']
            hri_date = Time(hri_header['DATE-OBS'])
            multiplet = [hri_file]
            if len(self.paths) > 1:
                for odates, ofiles, odists in zip(other_dates, other_files, other_distances):
                    dt = (np.array(odists) - hri_distance) / astropy.constants.c.value / 86400
                    idx = np.argmin(np.abs(odates - dt - hri_date))
                    multiplet.append(ofiles[idx])
            multiplets.append(multiplet)

        for s in range(len(multiplets[0])):
            self.stacks.append(Stack(files=[m[s] for m in multiplets]))

        self.masterstack = self.stacks[self.master]

    def extract_events(self, instruments=None, sigma=5, n_levels=2, dmin=0, vmin=0, vmax=None, saturation=True):
        self.sigma = sigma
        self.n_levels = n_levels
        self.dmin = dmin
        if instruments is None: instruments = [self.master]
        if type(instruments) is not list:
            instruments = [instruments]
        for instr in instruments:
            self.stacks[instr].extract_events(sigma=sigma, n_levels=n_levels, dmin=dmin, vmin=vmin, vmax=vmax,
                                              detection_method=self.detection_method, saturation=saturation)

    def extract_background(self, instruments=None, sigma=1, n_levels=3, dmin=0, vmin=0, vmax=None):
        if instruments is None:
            instruments = [self.master]
        if type(instruments) is not list:
            instruments = [instruments]
        for instr in instruments:
            self.stacks[instr].extract_background(sigma=sigma, n_levels=n_levels, dmin=dmin, vmin=vmin, vmax=vmax,
                                                  detection_method=self.detection_method)

    def copy(self, stack):
        self.stacks.append(copy.deepcopy(stack))

    def inpaint(self, stack, recompute=False, renoise=True):
        if recompute:
            data, header = stack.images[0].get()
            mask = np.zeros_like(data, dtype=np.uint8)
            for ev in self.masterstack.events:
                if "AIA" in stack.images[0].header["TELESCOP"]:
                    dx, dy = ev.shift
                else:
                    dx, dy = 0, 0
                slc = (slice(ev.slc[1].start - round(dy), ev.slc[1].stop - round(dy)),
                       slice(ev.slc[2].start - round(dx), ev.slc[2].stop - round(dx)))
                mask[slc][(~ev.blob.mask).sum(axis=0) > 0] = 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.dilate(mask, kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            grow = cv2.dilate(mask, kernel, iterations=1)
            rim = grow - mask
        for image in stack.images:
            splt = os.path.splitext(image.file)
            newfile = splt[0] + "_inpainted" + splt[1]
            if recompute:
                data, header = image.get()
                bound = np.copy(data)
                mini = bound.min()
                bound -= mini
                bound **= 0.5
                maxi = bound.max()
                bound /= maxi
                bound *= 255
                bound = bound.astype(np.uint8)
                cv2.inpaint(bound, mask, 3, cv2.INPAINT_NS, bound)
                cv2.inpaint(bound, rim, 3, cv2.INPAINT_NS, bound)
                bound = bound.astype(data.dtype)
                bound /= 255
                bound *= maxi
                bound **= 2
                bound += mini
                inpainted = np.copy(data)
                inpainted[mask == 1] = bound[mask == 1]
                inpainted[rim == 1] = bound[rim == 1]
                if renoise:
                    nz = np.copy(inpainted)
                    nz[inpainted < 0] = 0
                    rng = np.random.default_rng()
                    poisson = rng.poisson(lam=nz)
                    inpainted[mask == 1] = poisson[mask == 1]
                fits.writeto(newfile, inpainted * image.dn_per_photoelectron, header, overwrite=True)
            image.file = newfile

    def file_suffix(self):
        suffix = self.detection_method
        if self.detection_method == "wavelets":
            suffix = suffix + f"_sigma{self.sigma}_levels{self.n_levels}_dmin{self.dmin}"
        else:
            suffix += f"_dmin{self.dmin}"
        return suffix

    def events_tofits(self, first_n=None):

        sort = np.argsort([ev.relative_variance for ev in self.masterstack.events])[::-1]
        if first_n is not None:
            if first_n > 65535:
                first_n = 65535
            if first_n > sort.shape[0]:
                first_n = sort.shape[0]
            sort = sort[0:first_n]
        for idx, image in enumerate(self.masterstack.images):
            fi_out = image.file[:-5] + '_detected_regions_' + \
                     self.file_suffix() + ".fits.gz"
            header = fits.getheader(image.file)
            regions = np.zeros((header['NAXIS2'], header['NAXIS1']), dtype=np.uint16)
            fov = (slice(0, header['NAXIS2']),
                   slice(0, header['NAXIS1']))
            for i, s in enumerate(sort):
                ev = self.masterstack.events[s]
                if ev.isinframe(fov, fnum=idx):
                    regions[ev.slc[1].start:ev.slc[1].stop,
                    ev.slc[2].start:ev.slc[2].stop][~ev.blob.mask[idx - ev.slc[0].start]] = i

            fits.writeto(fi_out, regions, header=header, overwrite=True)

        # regions, nregions = ndi.label(mask)
        # center_of_intensity = ndi.center_of_mass(cube,labels=regions,index=range(nregions+1))
        # center_of_intensity = np.array(center_of_intensity)
        # table_center = Table([center_of_intensity[:,0],center_of_intensity[:,1],center_of_intensity[:,2]],
        #            names=('T','X','Y'))
        # table_center.write('center_of_intensity_v3.fits',format='fits')

    def events_totable(self, first_n=None, output_filename='EvtCatalog'):

        sort = np.argsort([ev.relative_variance for ev in self.masterstack.events])[::-1]
        if first_n is not None:
            if first_n > 65535:
                first_n = 65535
            if first_n > sort.shape[0]:
                first_n = sort.shape[0]
            sort = sort[0:first_n]

        output_names = (
                        'CF', 'xc', 'yc', 'tc', 'projected_area', 'xwidth', 'ywidth', 'height', 'volume', 'duration',
                        'total_intensity', 'mean_intensity', 'max_intensity', 'xmax', 'ymax', 'tmax', 'variance',
                        'xbary', 'ybary', 'tbary', 'barintensity',
                        'relative_variance', 'x_image', 'y_image', 'lon_carrington', 'lat_carrington', 'corrcoeff',
                        'LOSdist', 'x_shift', 'y_shift', 'major_axis', 'minor_axis', 'angle'
        )

        a_index = []
        a_xc = []
        a_yc = []
        a_tc = []
        a_projected_area = []
        a_xwidth = []
        a_ywidth = []
        a_height = []
        a_volume = []
        a_duration = []
        a_total_intensity = []
        a_mean_intensity = []
        a_max_intensity = []
        a_xmax = []
        a_ymax = []
        a_tmax = []
        a_variance = []
        a_xbary = []
        a_ybary = []
        a_tbary = []
        a_barintensity = []
        a_relative_variance = []
        a_x_image_coord = []
        a_y_image_coord = []
        a_lon_carrington = []
        a_lat_carrington = []
        a_corrcoeff = []
        a_los_dist = []
        a_x_shift = []
        a_y_shift = []
        a_major_axis = []
        a_minor_axis = []
        a_angle = []

        a_height_fort = []
        a_min_segment_fort = []
        a_shift_fort = []
        a_corrcoeff_fort = []
        a_image_coords_fort = []
        a_carrington_coords_fort = []

        for i, s in enumerate(sort):
            ev = self.masterstack.events[s]

            a_index.append(i)
            a_xc.append(ev.xc)
            a_yc.append(ev.yc)
            a_tc.append(ev.tc)
            a_projected_area.append(ev.projected_area)
            a_xwidth.append(ev.xwidth)
            a_ywidth.append(ev.ywidth)
            a_height.append(ev.height)
            a_volume.append(ev.volume)
            a_duration.append(ev.duration)
            a_total_intensity.append(ev.total_intensity)
            a_mean_intensity.append(ev.mean_intensity)
            a_max_intensity.append(ev.max_intensity)
            a_xmax.append(ev.xmax)
            a_ymax.append(ev.ymax)
            a_tmax.append(ev.tmax)
            a_variance.append(ev.variance)
            a_xbary.append(ev.xbary)
            a_ybary.append(ev.ybary)
            a_tbary.append(ev.tbary)
            a_barintensity.append(ev.barintensity)
            a_relative_variance.append(ev.relative_variance)
            a_x_image_coord.append(ev.image_coords[0])
            a_y_image_coord.append(ev.image_coords[1])
            a_lon_carrington.append(ev.carrington_coords[0])
            a_lat_carrington.append(ev.carrington_coords[1])
            a_corrcoeff.append(ev.corrcoeff)
            a_los_dist.append(ev.min_segment)
            a_x_shift.append(ev.shift[0])
            a_y_shift.append(ev.shift[1])
            a_major_axis.append(ev.ellipse_parameters[0])
            a_minor_axis.append(ev.ellipse_parameters[1])
            a_angle.append(ev.ellipse_parameters[2])

            a_height_fort.append(ev.height_fort)
            a_min_segment_fort.append(ev.min_segment_fort)
            a_shift_fort.append(ev.shift_fort)
            a_corrcoeff_fort.append(ev.corrcoeff_fort)
            a_image_coords_fort.append(ev.image_coords_fort)
            a_carrington_coords_fort.append(ev.carrington_coords_fort)

        output_table = Table(
            [a_index, a_xc, a_yc, a_tc, a_projected_area, a_xwidth, a_ywidth, a_height, a_volume, a_duration,
             a_total_intensity, a_mean_intensity,
             a_max_intensity, a_xmax, a_ymax, a_tmax, a_variance, a_xbary, a_ybary, a_tbary, a_barintensity,
             a_relative_variance,
             a_x_image_coord, a_y_image_coord, a_lon_carrington, a_lat_carrington, a_corrcoeff, a_los_dist, a_x_shift,
             a_y_shift,
             a_major_axis, a_minor_axis, a_angle],
            names=output_names)

        output_table.write(output_filename + '.fits', format='fits', overwrite=True)
        output_table.write(output_filename + '.csv', format='csv', overwrite=True)

        foreveryt_table = Table([a_height_fort, a_min_segment_fort, a_shift_fort],
                                names=['height_fort', 'min_segment_fort', 'shift_fort'])

        foreveryt_table.write(output_filename + '_correl_everyt.fits', format='fits', overwrite=True)

    def plot_histograms(self):
        nbins = 50
        xmin = 1e2
        xmax = 7e2
        bins = np.linspace(xmin, xmax, nbins)
        total_hist = np.zeros(nbins - 1, dtype=np.float64)
        events_hist = np.zeros(nbins - 1, dtype=np.float64)
        for i in self.masterstack.images:
            data, _ = i.get()
            total_hist += np.histogram(data, bins=bins)[0]
            events = [ev for ev in self.masterstack.events if ev.isinframe(None, fnum=i.idx)]
            for ev in events:
                events_hist += np.histogram(data[ev.slc[1].start:ev.slc[1].stop,
                                            ev.slc[2].start:ev.slc[2].stop],
                                            bins=bins)[0]

        total_hist /= (sum(total_hist) * np.diff(bins))
        events_hist /= (sum(events_hist) * np.diff(bins))

        fig = plt.figure(figsize=(10, 8))
        plt.step(bins[1:], total_hist)
        plt.step(bins[1:], events_hist)
        plt.xscale('linear')
        plt.yscale('linear')
        plt.ylim(0, 0.007)
        plt.xlabel('Intensity (photons)')
        plt.ylabel('Frequency')
        fig.savefig('histogram_' + self.file_suffix() + ".png")

    def plot_2dhistograms(self):
        nbins1 = 25
        nbins2 = 25
        bin1 = np.linspace(100, 1400, nbins1)
        bin2 = np.linspace(100, 620, nbins2)
        total_hist = np.zeros((nbins1 - 1, nbins2 - 1), dtype=np.float64)
        events_hist = np.zeros((nbins1 - 1, nbins2 - 1), dtype=np.float64)
        for i1, i2 in zip(self.masterstack.images, self.stacks[1].images):
            data1, _ = i1.get()
            data2, _ = i2.get()
            total_hist += np.histogram2d(data2.ravel(), data1.ravel(), bins=(bin1, bin2))[0]
            events = [ev for ev in self.masterstack.events if ev.isinframe(None, fnum=i1.idx)]
            for ev in events:
                events_hist += np.histogram2d(data2[ev.slc[1].start:ev.slc[1].stop,
                                              ev.slc[2].start:ev.slc[2].stop].ravel(),
                                              data1[ev.slc[1].start:ev.slc[1].stop,
                                              ev.slc[2].start:ev.slc[2].stop].ravel(),
                                              bins=(bin1, bin2))[0]

        total_hist /= total_hist.sum()
        events_hist /= events_hist.sum()

        fig = plt.figure(figsize=(10, 8))
        plt.imshow(events_hist, origin="lower", cmap="gray",
                   norm=mcolors.PowerNorm(1.0),
                   aspect=(bin2.max() - bin2.min()) / (bin1.max() - bin1.min()),
                   extent=[bin2.min(), bin2.max(), bin1.min(), bin1.max()])
        x, y = np.meshgrid(bin2[:-1], bin1[:-1])
        plt.contour(x, y, total_hist)
        plt.xscale('linear')
        plt.yscale('linear')
        plt.xlabel('Intensity HRI174 (photons)')
        plt.ylabel('HRYlya Intensity (DN.s$^{-1}$)')
        fig.savefig('histogram2d_' + self.file_suffix() + ".png")

    def compute_heights(self, method="opencv", position_type="bary"):

        if method == "phase_cross_correlation":
            halfvisu = 7  # nb of pixel in visu : 2*halfsize+1
            halfco = 7  # nb of pixel in cross-correl : 2*halfco+1
        elif method == "opencv":
            halfvisu = 14  # nb of pixel in visu : 2*halfsize+1
            halfco = 7  # nb of pixel in cross-correl : 2*halfco+1
        elif method == "correlate2d":
            halfvisu = 14  # nb of pixel in visu : 2*halfsize+1
            halfco = 7  # nb of pixel in cross-correl : 2*halfco+1

        sc = slice(halfvisu - halfco, halfvisu + halfco + 1)

        for t in set([getattr(ev, "t" + position_type) for ev in self.masterstack.events]):

            stk1data, hd1 = self.masterstack.images[t].get()
            stk2ata, hd2 = self.stacks[1].images[t].get()

            transform = rectify.CarringtonTransform(hd1,
                                                    radius_correction=hd1["MAPPINGR"] / astropy.constants.R_sun.value)

            for evt in [ev for ev in self.masterstack.events if getattr(ev, "t" + position_type) == t]:

                xevt = getattr(evt, "x" + position_type)
                yevt = getattr(evt, "y" + position_type)

                if method == "phase_cross_correlation":
                    avgdx = 0
                    avgdy = 0
                    xx = slice(xevt - halfvisu, xevt + halfvisu + 1)
                    yy = slice(yevt - halfvisu, yevt + halfvisu + 1)
                    sub1 = stk1data[yy, xx]
                    xx = slice(xevt - halfvisu + avgdx, xevt + halfvisu + 1 + avgdx)
                    yy = slice(yevt - halfvisu + avgdy, yevt + halfvisu + 1 + avgdy)
                    sub2 = stk2ata[yy, xx]
                    shift, error, diffphase = phase_cross_correlation(sub1, sub2,
                                                                      upsample_factor=10)  # precision 1/10 pixel
                    cc = error
                    shift = shift[1] - avgdx, shift[0] - avgdy
                elif method == "opencv":
                    avgdx = 0
                    avgdy = 0
                    # avgdx = -8
                    # avgdy = -2
                    xx = slice(xevt - halfvisu, xevt + halfvisu + 1)
                    yy = slice(yevt - halfvisu, yevt + halfvisu + 1)
                    sub1 = stk1data[yy, xx]
                    xx = slice(xevt - halfvisu + avgdx, xevt + halfvisu + 1 + avgdx)
                    yy = slice(yevt - halfvisu + avgdy, yevt + halfvisu + 1 + avgdy)
                    sub2 = stk2ata[yy, xx][halfvisu // 2:-halfvisu // 2, halfvisu // 2:-halfvisu // 2]
                    cc = cv2.matchTemplate(sub1, sub2, cv2.TM_CCOEFF_NORMED)
                    dx, dy = parabolic(cc)
                    dx -= halfvisu // 2
                    dy -= halfvisu // 2
                    shift = dx - avgdx, dy - avgdy
                elif method == "correlate2d":
                    xx = slice(xevt - halfvisu, xevt + halfvisu + 1)
                    yy = slice(yevt - halfvisu, yevt + halfvisu + 1)
                    sub1 = stk1data[yy, xx]
                    sub2 = stk2ata[yy, xx][halfvisu // 2:-halfvisu // 2, halfvisu // 2:-halfvisu // 2]
                    cc = correlate2d(sub1, sub2, mode='same')
                    dx, dy = parabolic(cc)
                    shift = dx, dy
                    print(shift, cc.shape)

                corr_lon = 0.0

                lon1 = (xevt - hd1["CACRPIX1"] + 1) * hd1["CACDELT1"] + hd1["CACRVAL1"]
                lat1 = (yevt - hd1["CACRPIX2"] + 1) * hd1["CACDELT2"] + hd1["CACRVAL2"]
                lon2 = (xevt - shift[0] - hd2["CACRPIX1"] + 1) * hd2["CACDELT1"] + hd2["CACRVAL1"] + corr_lon
                lat2 = (yevt - shift[1] - hd2["CACRPIX2"] + 1) * hd2["CACDELT2"] + hd2["CACRVAL2"]

                o1 = Point(np.radians(hd1["CRLN_OBS"]), np.radians(hd1["CRLT_OBS"]), hd1["DSUN_OBS"] / hd1["MAPPINGR"],
                           rect=False)
                c1 = Point(np.radians(lon1), np.radians(lat1), 1, rect=False)
                o2 = Point(np.radians(hd2["CRLN_OBS"] + corr_lon), np.radians(hd2["CRLT_OBS"]),
                           hd2["DSUN_OBS"] / hd2["MAPPINGR"], rect=False)
                c2 = Point(np.radians(lon2), np.radians(lat2), 1, rect=False)
                l1 = Line(o1, c1)
                l2 = Line(o2, c2)
                _, _, m, d = l1.findClosestPoints(l2)

                h = np.sqrt(m.x ** 2 + m.y ** 2 + m.z ** 2)

                evt.height = h * hd1["MAPPINGR"] - astropy.constants.R_sun.value
                evt.min_segment = d * hd1["MAPPINGR"]
                evt.shift = shift
                evt.corrcoeff = cc.max()

                # image_coords = transform(x=lon1, y=lat1)
                # evt.image_coords = (float(image_coords[0]), float(image_coords[1]))
                # evt.carrington_coords = (lon1, lat1)

    def compute_heights_foreveryt(self, method="opencv", position_type="bary"):

        if method == "phase_cross_correlation":
            halfvisu = 7  # nb of pixel in visu : 2*halfsize+1
            halfco = 7  # nb of pixel in cross-correl : 2*halfco+1
        elif method == "opencv":
            halfvisu = 14  # nb of pixel in visu : 2*halfsize+1
            halfco = 7  # nb of pixel in cross-correl : 2*halfco+1
        elif method == "correlate2d":
            halfvisu = 14  # nb of pixel in visu : 2*halfsize+1
            halfco = 7  # nb of pixel in cross-correl : 2*halfco+1

        sc = slice(halfvisu - halfco, halfvisu + halfco + 1)

        for t in range(self.masterstack.n_images):

            stk1data, hd1 = self.masterstack.images[t].get()
            stk2ata, hd2 = self.stacks[1].images[t].get()

            transform = rectify.CarringtonTransform(hd1,
                                                    radius_correction=hd1["MAPPINGR"] / astropy.constants.R_sun.value)

            for evt in [ev for ev in self.masterstack.events if ev.isinframe(None, t)]:

                xevt, yevt = evt.get_center_at_t(t, position_type=position_type)

                if method == "phase_cross_correlation":
                    avgdx = 0
                    avgdy = 0
                    xx = slice(xevt - halfvisu, xevt + halfvisu + 1)
                    yy = slice(yevt - halfvisu, yevt + halfvisu + 1)
                    sub1 = stk1data[yy, xx]
                    xx = slice(xevt - halfvisu + avgdx, xevt + halfvisu + 1 + avgdx)
                    yy = slice(yevt - halfvisu + avgdy, yevt + halfvisu + 1 + avgdy)
                    sub2 = stk2ata[yy, xx]
                    shift, error, diffphase = phase_cross_correlation(sub1, sub2,
                                                                      upsample_factor=10)  # precision 1/10 pixel
                    cc = error
                    shift = shift[1] - avgdx, shift[0] - avgdy
                elif method == "opencv":
                    avgdx = 0
                    avgdy = 0
                    # avgdx = -8
                    # avgdy = -2
                    xx = slice(xevt - halfvisu, xevt + halfvisu + 1)
                    yy = slice(yevt - halfvisu, yevt + halfvisu + 1)
                    sub1 = stk1data[yy, xx]
                    xx = slice(xevt - halfvisu + avgdx, xevt + halfvisu + 1 + avgdx)
                    yy = slice(yevt - halfvisu + avgdy, yevt + halfvisu + 1 + avgdy)
                    sub2 = stk2ata[yy, xx][halfvisu // 2:-halfvisu // 2, halfvisu // 2:-halfvisu // 2]
                    cc = cv2.matchTemplate(sub1, sub2, cv2.TM_CCOEFF_NORMED)
                    dx, dy = parabolic(cc)
                    dx -= halfvisu // 2
                    dy -= halfvisu // 2
                    shift = dx - avgdx, dy - avgdy
                elif method == "correlate2d":
                    xx = slice(xevt - halfvisu, xevt + halfvisu + 1)
                    yy = slice(yevt - halfvisu, yevt + halfvisu + 1)
                    sub1 = stk1data[yy, xx]
                    sub2 = stk2ata[yy, xx][halfvisu // 2:-halfvisu // 2, halfvisu // 2:-halfvisu // 2]
                    cc = correlate2d(sub1, sub2, mode='same')
                    dx, dy = parabolic(cc)
                    shift = dx, dy
                    print(shift, cc.shape)

                corr_lon = 0.0

                lon1 = (xevt - hd1["CACRPIX1"] + 1) * hd1["CACDELT1"] + hd1["CACRVAL1"]
                lat1 = (yevt - hd1["CACRPIX2"] + 1) * hd1["CACDELT2"] + hd1["CACRVAL2"]
                lon2 = (xevt - shift[0] - hd2["CACRPIX1"] + 1) * hd2["CACDELT1"] + hd2["CACRVAL1"] + corr_lon
                lat2 = (yevt - shift[1] - hd2["CACRPIX2"] + 1) * hd2["CACDELT2"] + hd2["CACRVAL2"]

                o1 = Point(np.radians(hd1["CRLN_OBS"]), np.radians(hd1["CRLT_OBS"]), hd1["DSUN_OBS"] / hd1["MAPPINGR"],
                           rect=False)
                c1 = Point(np.radians(lon1), np.radians(lat1), 1, rect=False)
                o2 = Point(np.radians(hd2["CRLN_OBS"] + corr_lon), np.radians(hd2["CRLT_OBS"]),
                           hd2["DSUN_OBS"] / hd2["MAPPINGR"], rect=False)
                c2 = Point(np.radians(lon2), np.radians(lat2), 1, rect=False)
                l1 = Line(o1, c1)
                l2 = Line(o2, c2)
                _, _, m, d = l1.findClosestPoints(l2)

                h = np.sqrt(m.x ** 2 + m.y ** 2 + m.z ** 2)
                image_coords = transform(x=lon1, y=lat1)

                evt.height_fort[t] = h * hd1["MAPPINGR"] - astropy.constants.R_sun.value
                evt.min_segment_fort[t] = d * hd1["MAPPINGR"]
                evt.shift_fort[t, :] = np.array(shift)
                evt.corrcoeff_fort[t] = cc.max()
                evt.image_coords_fort[t, :] = np.array([float(image_coords[0]), float(image_coords[1])])
                evt.carrington_coords_fort[t, :] = np.array(lon1, lat1)

    def plot_heights(self, instrument=None, full_events=False):

        if instrument is None: instrument = self.master

        if full_events == False:
            events = self.stacks[instrument].events
        else:
            events = [ev for ev in self.stacks[instrument].events if
                      ev.slc[0].start > 0 and ev.slc[0].stop < len(self.stacks[instrument])]

        sort = np.argsort([ev.relative_variance for ev in events])[::-1]
        events = [events[s] for s in sort]

        _, hdr = self.masterstack.images[0].get()

        marilena = [138, 41, 184, 405, 454, 452, 483, 724, 64, 3, 257, 74, 771, 135, 596, 758]

        heights = [ev.height / 1e6 for ev in events]

        fig, ax = plt.subplots(2, 3, figsize=(18 / 1.5, 12 / 1.5))

        nbins = 30

        xmin = 0
        xmax = 6
        # logbins = np.logspace(np.log10(xmin),np.log10(xmax), nbins)
        bins = np.linspace(xmin, xmax, nbins)

        pixlength = (np.radians(hdr['CACDELT1']) * astropy.constants.R_sun.value / 1e6)
        pixarea = pixlength ** 2

        ax[0, 0].hist(heights, bins=bins, density=True, histtype="step", color="black")
        # ax[0, 0].set_xscale('linear')
        # ax[0, 0].set_yscale('linear')
        ax[0, 0].set_xlabel('Height (Mm)')
        ax[0, 0].set_ylabel('Probability density')
        ax[0, 0].set_xlim(xmin, xmax)
        # ax[0, 0].set_ylim(2e-3, 2e2)

        #        corrcoeffs = [ev.corrcoeff for ev in events]
        # corrcoeffs = [ev.corrcoeff for ev in events]
        # colors = corrcoeffs
        # events_marilena = [events[m] for m in marilena]
        # heights_marilena = [events[m].height/1e6 for m in marilena]

        colors = "gray"
        size = 8

        cmap = "viridis"

        ax[0, 1].scatter([ev.ellipse_parameters[0] * pixlength for ev in events],
                         heights,
                         s=size,
                         c=colors, cmap=cmap)
        # ax[0, 1].scatter([ev.ellipse_parameters[0]*pixlength for ev in events_marilena],
        #                   heights_marilena,
        #                   s=size, 
        #                   c="red", cmap=cmap)
        # ax[0, 1].set_xscale('Log')
        ax[0, 1].set_xlabel('Major axis length (Mm)')
        ax[0, 1].set_ylabel('Height (Mm)')
        ax[0, 1].plot([0, 5], [0, 2.5], ':', color="gray")

        # ax[0, 1].scatter([ev.ellipse_parameters[0]*pixlength for ev in events],
        #                  [ev.ellipse_parameters[0]/ev.ellipse_parameters[1] for ev in events], s=size, 
        #                   c="gray", cmap=cmap)
        # #ax[0, 1].set_xscale('Log')
        # ax[0, 1].set_xlabel('Major axis length (Mm)')
        # ax[0, 1].set_ylabel('Aspect ratio')
        # ax[0, 1].plot([0, 5], [0, 5], ':', color="gray")

        # ax[0, 1].scatter([ev.feret_diameter*pixlength for ev in events], heights, s=size, c=corrcoeffs)
        # #ax[0, 1].set_xscale('Log')
        # ax[0, 1].set_xlabel('Maximum length (Mm)')
        # ax[0, 1].set_ylabel('Height (Mm)')
        # ax[0, 1].plot([0, 5], [0, 2.5], ':', color="gray")

        # ax[0, 1].scatter([ev.feret_diameter*pixlength for ev in events], [ev.ellipse_parameters[0]*pixlength for ev in events], s=size, c=corrcoeffs)
        # #ax[0, 1].set_xscale('Log')
        # ax[0, 1].set_xlabel('Maximum length (Mm)')
        # ax[0, 1].set_ylabel('Major axis length (Mm)')
        # ax[0, 1].plot([0, 4], [0, 4], 'b:')

        # ax[0, 1].scatter([ev.projected_area*pixarea for ev in events], heights, s=size, c=corrcoeffs)
        # ax[0, 1].set_xscale('Log')
        # ax[0, 1].set_xlabel('Projected ared (Mm$^2$)')
        # ax[0, 1].set_ylabel('Height (Mm)')

        # xmin = 0.1
        # xmax = 10
        # logbins = np.logspace(np.log10(xmin),np.log10(xmax), nbins)
        # ax[2].hist([ev.ellipse_parameters[0]*pixlength for ev in events], logbins, density=True)
        # ax[2].set_xscale('log')
        # ax[2].set_yscale('log')
        # ax[2].set_xlabel('Major axis length (Mm)')
        # ax[2].set_ylabel('Probability density')
        # ax[2].set_xlim(xmin, xmax)
        # ax[2].set_ylim(1e-3, 1e2)

        ax[0, 2].scatter([ev.total_intensity for ev in events], heights, s=size, c=colors, cmap=cmap)
        # ax[0, 2].scatter([ev.total_intensity for ev in events_marilena], heights_marilena, s=size, c="red", cmap=cmap)
        ax[0, 2].set_xscale('Log')
        ax[0, 2].set_xlabel('Total intensity (photons)')
        ax[0, 2].set_ylabel('Height (Mm)')

        ax[1, 0].scatter([-ev.shift[0] for ev in events], [ev.shift[1] for ev in events], s=size, c=colors, cmap=cmap)
        # ax[1, 0].scatter([-ev.shift[0] for ev in events_marilena], [ev.shift[1] for ev in events_marilena], s=size, c="red", cmap=cmap)
        ax[1, 0].set_xlabel('Longitude shift (pixels)')
        ax[1, 0].set_ylabel('Latitude shift (pixels)')

        dt = 5.0
        ax[1, 1].scatter([ev.duration * dt for ev in events], heights, s=size, c=colors, cmap=cmap)
        # ax[1, 1].scatter([ev.duration*dt for ev in events_marilena], heights_marilena, s=size, c="red", cmap=cmap)
        ax[1, 1].set_xlabel('Duration (s)')
        ax[1, 1].set_ylabel('Height (Mm)')

        # ax[1, 1].scatter([ev.carrington_coords[0] for ev in events], heights, s=size, c=colors, cmap=cmap)
        # ax[1, 1].scatter([ev.carrington_coords[0] for ev in events_marilena], heights_marilena, s=size, c="red", cmap=cmap)
        # ax[1, 1].set_xlabel('Longitude (degrees)')
        # ax[1, 1].set_ylabel('Height (Mm)')

        # ax[1, 2].scatter(heights, [ev.min_segment/1e6 for ev in events], s=size, c=colors, cmap=cmap)
        # ax[1, 2].scatter(heights_marilena, [ev.min_segment/1e6 for ev in events_marilena], s=size, c="red", cmap=cmap)
        # ax[1, 2].set_xlabel('Height (Mm)')
        # ax[1, 2].set_ylabel('LOS distance (Mm)')
        # ax[1, 2].plot([0, 6], [0, 6], ':', color="gray")
        # ax[1, 2].set_ylim(-0.1, 1.8)

        ax[1, 2].scatter([ev.ellipse_parameters[0] * pixlength for ev in events], [ev.duration * dt for ev in events],
                         s=size, c=colors, cmap=cmap)
        # ax[1, 2].scatter([ev.ellipse_parameters[0]*pixlength for ev in events_marilena], [ev.duration*dt for ev in events_marilena], s=size, c="red", cmap=cmap)
        ax[1, 2].set_xlabel('Major axis length (Mm)')
        ax[1, 2].set_ylabel('Duration (s)')
        ax[1, 2].plot([0, 6], [0, 6], ':', color="gray")
        ax[1, 2].set_ylim(1, 200)

        fig.tight_layout()

        filename = 'events_heights_' + self.file_suffix() + '.png'
        fig.savefig(filename, dpi=200)
        filename = 'events_heights_' + self.file_suffix() + '.eps'
        fig.savefig(filename)

    def plot_statistics(self, instrument=None, full_events=False):
        if instrument is None:
            instrument = self.master
        if len(self.stacks) == 0:
            self.extract_events(instrument=instrument)

        if full_events == False:
            events = self.stacks[instrument].events
        else:
            events = [ev for ev in self.stacks[instrument].events if
                      ev.slc[0].start > 0 and ev.slc[0].stop < len(self.stacks[instrument])]

        print(len(events))

        fig, ax = plt.subplots(3, 3, figsize=(18, 18))
        _, hdr = self.masterstack.images[0].get()
        dt = 5.0
        if 'CACDELT1' in hdr:
            pixlength = np.radians(hdr['CACDELT1']) * astropy.constants.R_sun.value / 1e6
        else:
            pixlength = np.radians(hdr['CDELT1']/3600) * hdr['DSUN_OBS'] / 1e6
        pixarea = pixlength ** 2

        nbins = 50

        xmin = 0.03
        xmax = 3
        logbins = np.logspace(np.log10(xmin), np.log10(xmax), nbins)
        areas = [ev.projected_area * pixarea for ev in events]
        ax[0, 0].hist([a for a in areas if a == min(areas)], bins=logbins, density=True, histtype="step",
                      color='#1f77ba')
        ax[0, 0].hist([a for a in areas if a > min(areas)], bins=logbins, density=True, color='#1f77ba')
        ax[0, 0].set_xscale('log')
        ax[0, 0].set_yscale('log')
        ax[0, 0].set_xlabel('Projected area (Mm$^2$)')
        ax[0, 0].set_ylabel('Probability density')
        ax[0, 0].set_xlim(xmin, xmax)
        ax[0, 0].set_ylim(3e-3, 3e2)

        voxvol = pixarea * dt

        xmin = 1
        xmax = 1e3
        logbins = np.logspace(np.log10(xmin), np.log10(xmax), nbins)
        ax[0, 1].hist([ev.volume * voxvol for ev in events], bins=logbins, density=True)
        ax[0, 1].set_xscale('log')
        ax[0, 1].set_yscale('log')
        ax[0, 1].set_xlabel('Volume (Mm$^2$.s)')
        ax[0, 1].set_ylabel('Probability density')
        ax[0, 1].set_xlim(xmin, xmax)
        ax[0, 1].set_ylim(1e-6, 1e1)

        xmin = 4
        xmax = 600
        logbins = np.logspace(np.log10(xmin), np.log10(xmax), nbins)
        durations = [ev.duration * dt for ev in events]
        ax[1, 0].hist([d for d in durations if d == min(durations)], bins=logbins, density=True, histtype="step",
                      color='#1f77ba')
        ax[1, 0].hist([d for d in durations if d > min(durations)], bins=logbins, density=True, color='#1f77ba')
        ax[1, 0].set_xscale('log')
        ax[1, 0].set_yscale('log')
        ax[1, 0].set_xlabel('Duration (s)')
        ax[1, 0].set_ylabel('Probability density')
        ax[1, 0].set_xlim(xmin, xmax)
        ax[1, 0].set_ylim(1e-6, 5)

        xmin = 1e2
        xmax = 1e6
        logbins = np.logspace(np.log10(xmin), np.log10(xmax), nbins)
        ax[1, 1].hist([ev.total_intensity for ev in events], logbins, density=True)
        ax[1, 1].set_xscale('log')
        ax[1, 1].set_yscale('log')
        ax[1, 1].set_xlabel('Total intensity (photons)')
        ax[1, 1].set_ylabel('Probability density')
        ax[1, 1].set_xlim(xmin, xmax)
        ax[1, 1].set_ylim(1e-9, 1e-2)

        xmin = 0.1
        xmax = 10
        logbins = np.logspace(np.log10(xmin), np.log10(xmax), nbins)
        ax[0, 2].hist([ev.ellipse_parameters[0] * pixlength for ev in events], logbins, density=True)
        ax[0, 2].set_xscale('log')
        ax[0, 2].set_yscale('log')
        ax[0, 2].set_xlabel('Major axis length (Mm)')
        ax[0, 2].set_ylabel('Probability density')
        ax[0, 2].set_xlim(xmin, xmax)
        ax[0, 2].set_ylim(1e-5, 1e2)

        xmin = 0.1
        xmax = 10
        logbins = np.logspace(np.log10(xmin), np.log10(xmax), nbins)
        ax[1, 2].hist([ev.ellipse_parameters[1] * pixlength for ev in events], logbins, density=True)
        ax[1, 2].set_xscale('log')
        ax[1, 2].set_yscale('log')
        ax[1, 2].set_xlabel('Minor axis length (Mm)')
        ax[1, 2].set_ylabel('Probability density')
        ax[1, 2].set_xlim(xmin, xmax)
        ax[1, 2].set_ylim(1e-4, 1e2)

        xmin = -90
        xmax = 90
        bins = np.linspace(xmin, xmax, nbins)
        ax[2, 2].hist([np.degrees(ev.ellipse_parameters[2]) for ev in events], bins, density=True)
        ax[2, 2].set_xscale('linear')
        ax[2, 2].set_yscale('log')
        ax[2, 2].set_xlabel('Angle (degrees)')
        ax[2, 2].set_ylabel('Probability density')
        ax[2, 2].set_xlim(xmin, xmax)
        ax[2, 2].set_ylim(1e-5, 0.1)

        xmin = 1
        xmax = 10
        bins = np.linspace(xmin, xmax, nbins)
        ax[2, 1].hist(
            [ev.ellipse_parameters[0] / ev.ellipse_parameters[1] for ev in events if ev.ellipse_parameters[1] > 0],
            bins, density=True)
        ax[2, 1].set_xscale('linear')
        ax[2, 1].set_yscale('log')
        ax[2, 1].set_xlabel('Aspect ratio')
        ax[2, 1].set_ylabel('Probability density')
        ax[2, 1].set_xlim(xmin, xmax)
        ax[2, 1].set_ylim(1e-4, 10)

        xmin = 1e-5
        xmax = 1e1
        logbins = np.logspace(np.log10(xmin), np.log10(xmax), nbins)
        ax[2, 0].hist([ev.relative_variance for ev in events], logbins, density=True)
        ax[2, 0].set_xscale('log')
        ax[2, 0].set_yscale('log')
        ax[2, 0].set_xlabel('Relative variance')
        ax[2, 0].set_ylabel('Probability density')
        ax[2, 0].set_xlim(xmin, xmax)
        ax[2, 0].set_ylim(1e-4, 1e3)

        filename = 'events_statistics_' + self.file_suffix() + '.png'
        fig.tight_layout()

        fig.savefig(filename, dpi=200)

    def plot_events(self, first_n=None, colorby=None):
        plt.ioff()
        sort = np.argsort([ev.relative_variance for ev in self.masterstack.events])[::-1]
        if first_n is not None:
            if first_n > 65535:
                first_n = 65535
            if first_n > sort.shape[0]:
                first_n = sort.shape[0]
            sort = sort[0:first_n]
        events = [self.masterstack.events[s] for s in sort]

        if colorby is not None:
            cmap = plt.get_cmap("plasma")
            rgbcolors = np.uint8(255 * cmap(np.linspace(0, 1, 256)))
            attr = [getattr(ev, colorby) for ev in events]
            maxi, mini = max(attr), min(attr)
            for ev in events:
                ev.rgbcolor = rgbcolors[np.uint8(255 * (getattr(ev, colorby) - mini) / (maxi - mini))]
                ev.rgb_outline = ev.make_rgb_contour(flatten=True)
        else:
            cmap = plt.get_cmap("gray")
            mini = 0
            maxi = 255

        for i, s in enumerate(self.stacks):
            f = s.images[0]
            rgb = f.make_rgb(events=events,
                             fov=None,
                             flatten=True,
                             interval=visu.ManualInterval(vmin=f.parent_stack.get_min().min(),
                                                          vmax=f.parent_stack.get_max().max()))
            # red = np.logical_and(rgb[:, :, 0] == rgb[:, :, 1], rgb[:, :, 1] == rgb[:, :, 2]) == False
            # rgb[:, :, 0][red] = 255
            # rgb[:, :, 1][red] = 0
            # rgb[:, :, 2][red] = 0

            # filename = "locations_" + self.file_suffix() + f"_{i}.png"
            # display.savepng(filename, rgb[:, :, ::-1]/255, text=self.detection_method, textpos=(10, 10), fontsize=20)
            figsize = (12, 12)
            fig, ax = plt.subplots(figsize=figsize)
            _, h = f.get()
            if "CACRPIX1" in h:
                lon1 = (0 - h["CACRPIX1"] + 1) * h["CACDELT1"] + h["CACRVAL1"]
                lat1 = (0 - h["CACRPIX2"] + 1) * h["CACDELT2"] + h["CACRVAL2"]
                lon2 = (f.parent_stack.get_min().shape[1] - 1 - h["CACRPIX1"] + 1) * h["CACDELT1"] + h["CACRVAL1"]
                lat2 = (f.parent_stack.get_min().shape[0] - 1 - h["CACRPIX2"] + 1) * h["CACDELT2"] + h["CACRVAL2"]
                x_label = 'Carrington longitude (degrees)'
                y_label = 'Carrington latitude (degrees)'
            else:
                lon1 = 0
                lon2 = h['NAXIS1']*h['CDELT1']
                lat1 = 0
                lat2 = h['NAXIS2']*h['CDELT2']
                x_label = 'Solar X (arcseconds)'
                y_label = 'Solar Y (arcseconds)'
            im = ax.imshow(rgb, origin='lower', interpolation="nearest", cmap=cmap, extent=[lon1, lon2, lat1, lat2])

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = fig.colorbar(im, cax=cax)

            cb.set_label(colorby)
            im.set_clim(mini, maxi)

            plt.tight_layout()

            bounds = ax.get_window_extent().bounds
            inches = (bounds[2] - bounds[0]) / fig.dpi
            dpi = f.parent_stack.get_min().shape[1] / inches
            fig.savefig("locations_bar_" + self.file_suffix() + f"_{i}.png", dpi=dpi)

            # xc = [ev.xc for ev in self.masterstack.events]
            # yc = [ev.yc for ev in self.masterstack.events]
            # plt.scatter(xc, yc, facecolors='none', edgecolors='r', marker='s', s=5)
        plt.ion()

    def make_movies(self, first_n=None, indices=None, width=81, height=81, denoise=False):
        events = self.masterstack.events
        sort = np.argsort([ev.relative_variance for ev in events])[::-1]
        if first_n is not None:
            if first_n > 65535:
                first_n = 65535
            if first_n > sort.shape[0]:
                first_n = sort.shape[0]
            usedsort = sort[0:first_n]
        elif indices is not None:
            usedsort = sort[indices]
        else:
            usedsort = sort
        sort = list(sort)
        for i, s in enumerate(usedsort):
            outdir = os.path.join(self.outpath, 'detections')
            if not os.path.isdir(outdir): os.mkdir(outdir)
            outname = os.path.join(outdir, '{:04d}.mp4'.format(sort.index(s)))
            self.make_movie(outname, center_event=events[s], width=width, height=height, denoise=denoise)

    def make_movie(self, outname, center_event=None, width=81, height=81, denoise=False):

        def make_frame(images, fov=None, outframe=None, denoise=False):
            plt.ioff()
            if fov is None:
                fig = plt.figure(constrained_layout=True, figsize=(8, 8), dpi=50)
                gs = fig.add_gridspec(1, 1,
                                      width_ratios=[1],
                                      height_ratios=[1])
            else:
                fig = plt.figure(constrained_layout=True, figsize=(8, 8), dpi=50)
                gs = fig.add_gridspec(2, len(images),
                                      width_ratios=[1] * len(images),
                                      height_ratios=[2, 1])

            for idx, f in enumerate(images):
                nfov = Ellipsis if fov is None else fov
                interval = visu.AsymmetricPercentileInterval(3, 99)
                # if f.header is None:
                #     f.get()
                # if "HRI" in f.header["TELESCOP"]:
                #     stack = self.masterstack
                # elif "AIA" in f.header["TELESCOP"]:
                #     stack = self.stacks[1]
                stack = f.parent_stack
                minimap = stack.get_min()[nfov]
                mini, _ = interval.get_limits(minimap[minimap > 0])
                _, maxi = interval.get_limits(stack.get_max()[nfov])
                rgb = f.make_rgb(events=f.parent_stack.events,
                                 fov=fov,
                                 denoise=denoise,
                                 interval=visu.ManualInterval(vmin=mini,
                                                              vmax=maxi))
                ax = fig.add_subplot(gs[0, idx])
                ax.set_title(f.header['TELESCOP'] + '\n' + str(f.header['WAVELNTH']), fontsize=8)
                ax.imshow(rgb / 255, origin='lower', interpolation='nearest')
                if idx == 0:
                    bounds = ax.get_window_extent().bounds
                if idx > 0:
                    ax.yaxis.set_ticklabels([])

                if center_event.isinframe(fov, fnum=images[0].idx) and center_event.shift is not None:
                    xc, yc = center_event.xmax, center_event.ymax
                    xc -= fov[1].start
                    yc -= fov[0].start
                    ax.plot(xc, yc, '+g')
                    if idx > 0:
                        ax.plot(xc - center_event.shift[0], yc - center_event.shift[1], '+r')

            if fov is not None:
                ax = fig.add_subplot(gs[1, :])
                ax2 = ax.twinx()
                inframe = [ev for ev in self.masterstack.events if ev.isinframe(fov)]  # , fnum=images[0].idx)]
                for ev in inframe:
                    linestyle = "solid" if ev is center_event else "dotted"
                    color = [c / 255 for c in ev.rgbcolor]
                    ax.plot(range(ev.slc[0].start, ev.slc[0].stop), ev.light_curve, 'o', markersize=2, color=color,
                            linestyle=linestyle)
                    if ev is center_event and ~np.isnan(ev.height_fort[0]):
                        print(ev.height_fort[0], ev.height_fort[0] == np.nan, ev is center_event)
                        ax2.plot(range(ev.slc[0].start, ev.slc[0].stop),
                                 ev.height_fort[np.isfinite(ev.height_fort)] / 1e6, '+', markersize=2, color=color,
                                 linestyle="dashed")

                ax.set_xlim(0, len(images[0].parent_stack))
                infov = [ev for ev in self.masterstack.events if ev.isinframe(fov)]
                vmin = min([ev.light_curve.min() for ev in infov])
                vmax = max([ev.light_curve.max() for ev in infov])
                ax.set_ylim(vmin, vmax)
                ax.plot([images[0].idx, images[0].idx], [vmin, vmax])
                ax.spines['top'].set_visible(False)
                ax.set_ylabel("Intensisty (e$^-$/pixel)")

                if ~np.isnan(ev.height_fort[0]):
                    ax2.set_ylim(0, 6)
                    ax2.set_ylabel("height (Mm)")

            if outframe is None:
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                return buf
            else:
                fig_width_inches, fig_height_inches = fig.get_size_inches()
                ratio = fig_height_inches / fig_width_inches
                fig_width_pixels, fig_height_pixels = int(fig_width_inches * fig.dpi), int(fig_height_inches * fig.dpi)
                new_fig_width_pixels, new_fig_height_pixels = fig_width_pixels, fig_height_pixels
                while new_fig_width_pixels % 2 == 1 or new_fig_height_pixels % 2 == 1:
                    new_fig_width_pixels += 1
                    new_fig_height_pixels = int(new_fig_width_pixels * ratio)
                dpi = new_fig_width_pixels / fig_width_inches
                fig.savefig(outframe, dpi=dpi)

            plt.close(fig)
            plt.ion()

        def encode(indir, outname, flush=True):
            if outname.endswith('.gif'):
                files = glob.glob(os.path.join(indir, '*.png'))
                frames = []
                for f in files:
                    frames.append(imageio.imread(f))
                imageio.mimsave(outname, frames)
            else:
                subprocess.run(["ffmpeg",
                                "-i", os.path.join(indir, '%05d.png'),
                                "-vcodec", "libx264",
                                "-pix_fmt", "yuv420p",
                                "-crf", "22",
                                "-r", "10",
                                "-y", outname]
                               )

        if center_event is not None:
            fov = (slice(int(center_event.yc) - height // 2, int(center_event.yc) + height // 2 + 1),
                   slice(int(center_event.xc) - width // 2, int(center_event.xc) + width // 2 + 1))
        else:
            fov = None

        tempdir = os.path.join(os.path.dirname(outname), 'temp')
        if not os.path.isdir(tempdir): os.mkdir(tempdir)

        for i, images in enumerate(zip(*self.stacks)):
            outframe = os.path.join(tempdir, '{:05d}.png'.format(i))
            make_frame(images, outframe=outframe, fov=fov, denoise=denoise)

        encode(tempdir, outname)

    def plot_profiles(self, imgstack=None, first_n=49):

        s = np.sqrt(first_n)
        nx, ny = int(s), int(s)
        if nx * ny < first_n:
            nx += 1
        if nx * ny < first_n:
            ny += 1

        fig, axes = plt.subplots(nx, ny, figsize=(18, 18))

        sort = np.argsort([ev.relative_variance for ev in self.masterstack.events])[::-1]
        sort = sort[0:first_n + 1]
        events = [self.masterstack.events[s] for s in sort]
        events = [ev for ev in events if ev.corrcoeff > 0.5]  # and ev.projected_area < 25]

        hw = 21

        if imgstack is None:
            imgstack = self.masterstack

        styles = ("solid", "dotted", "dashed")

        for ev, ax in zip(events, axes.flatten()):

            recon, hdr = imgstack.images[ev.tmax].get()

            l = [ev.xmax, ev.ymax]

            if imgstack is not self.masterstack:
                sx, sy = int(np.round(ev.shift[0])), int(np.round(ev.shift[1]))
            else:
                sx, sy = 0, 0

            ax.imshow(recon[l[1] - hw - sy:l[1] + hw - sy, l[0] - hw - sx:l[0] + hw - sx], origin='lower',
                      cmap='gray')
            ax.tick_params(labelsize=7)

            hcross = [[(hw - 2, hw), (hw + 2, hw)]]
            lc = mcol.LineCollection(hcross, colors=(0, 1, 0, 1), linewidth=1)
            ax.add_collection(lc)
            vcross = [[(hw, hw - 2), (hw, hw + 2)]]
            lc = mcol.LineCollection(vcross, colors=(1, 0, 0, 1), linewidth=1)
            ax.add_collection(lc)

            idx = np.linspace(0, 2 * hw - 1, 2 * hw)

            for stack, st in zip(self.stacks, styles):

                recon, hdr = stack.images[ev.tmax].get()

                if stack is not self.masterstack:
                    sx, sy = int(np.round(ev.shift[0])), int(np.round(ev.shift[1]))
                else:
                    sx, sy = 0, 0

                hcut = np.copy(recon[l[1] - sy, l[0] - hw - sx:l[0] + hw - sx])
                hcut *= 1.8 * hw / hcut.max()
                ax.plot(idx, hcut, 'green', linewidth=2, linestyle=st)

                vcut = np.copy(recon[l[1] - hw - sy:l[1] + hw - sy, l[0] - sx])
                vcut *= 1.8 * hw / vcut.max()
                ax.plot(vcut, idx, 'red', linewidth=2, linestyle=st)

        fig.savefig("profiles_" + self.file_suffix() + ".png", dpi=200)

    def find_closest_event(self, x, y, t, position_type="max", same_t=True, frame=None):
        dist = lambda xo, yo, to: np.sqrt((x - xo) ** 2 + (y - yo) ** 2 + (t - to) ** 2)

        if frame is not None:
            events = [ev for ev in self.masterstack.events if getattr(ev, "t" + position_type) == frame]
        else:
            events = self.masterstack.events

        if same_t is True:
            d = [dist(*(*ev.image_coords, getattr(ev, "t" + position_type))) for ev in events]
        else:
            d = [dist(*(*ev.image_coords, getattr(ev, "t" + position_type))) for ev in events if
                 getattr(ev, "t" + position_type) == t]

        return min(d), events[np.argmin(d)]


def main(paths):
    seq = Sequence(paths, fov=None)

    seq.extract_events(sigma=8.6, dmin=0, vmin=1)
    seq.events_totable(output_filename='EvtCatalog_20200530')
    seq.plot_statistics()
    seq.plot_events()
    seq.events_tofits()
    seq.make_movies(first_n=10)


if __name__ == '__main__':
    # hri_path = r'C:\archive\Campfires\katsukawa'
    # hri_path = r'C:\archive\Campfires\20200530\HRI174'
    hri_path = r'C:\archive\eui\releases\2022\03\18'
    main([hri_path])
