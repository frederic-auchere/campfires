import numpy as np
import numpy.ma as ma
import cv2
import astropy.constants
from rectify import rectify
from skimage.measure import regionprops

rgbcolors = [[255, 0, 0],
             [0, 255, 0],
             [0, 0, 255],
             [255, 255, 0],
             [255, 0, 255],
             [0, 255, 255],
             [128, 0, 0],
             [0, 128, 0],
             [0, 0, 128],
             [128, 128, 0],
             [128, 0, 128],
             [0, 128, 128]]
ncolors = len(rgbcolors)


class Event:
    """
    Defines an 'event'
    """
    @classmethod
    def from_csv(cls):
        return cls()

    def __init__(self, parent, slc, blob, index):

        """
        index: event number as returned by label
        xc, yc, tc: center of interval for each event on x, y & t axes (in carrington pixels)
        projected area: area of theproject of the vent blob on the x, y plane
        xwidth, ywidth, duration: maximum width of the event in x, y, t
        height: altitude computed by cross correlation with AIA
        volume: number of voxels
        total_intensity: hum. total intensity
        mean_intensity: hum. mean intensity
        max_intensity: hum. maximum intensity
        xmax, ymax, tmax: position of the maximum of intensity (in carrington pixels)
        variance: variance of the light curve, which is the mean intensity at each time step
        xbary, ybary, tbary: position of the intensity weighted average (barycenter)
        barintensity: intensity at the barycenter
        relative_variance: variance normalized to the mean of the light curve
        x_image_coord, y_image_coord: coordinates in the original images corresponding to xmax, ymax
        carrington_coords: hum. carrington coordinates
        corrcoeff: correlation cofficient with AIA
        min_segment: length of the minimal segment between two HRI and AIA LOS
        shift: shift between HRI & AIA (in carrington pixels)
        height_fort: height for every t
        min_segment_fort: length of the minimal segment between two HRI and AIA LOS for every t
        shift_fort: shift between HRI & AIA (in carrington pixels) for every t
        corrcoeff_fort: correlation cofficient with AIA for every t
        image_coords_fort: coordinates in the original images corresponding to xmax, ymax for every t
        carrington_coords_fort: carrington coordinates for every t
        """
        self.feret_diameter = None
        self.relative_variance = None
        self.xbary, self.ybary, self.tbary = None, None, None
        self.barintensity = None
        self.variance = None
        self.ellipse_parameters = None
        self.xmax, self.ymax, self.tmax = None, None, None
        self.index = index
        self.parent_stack = parent
        self.slc = (slc[0],
                    slice(slc[1].start - 1, slc[1].stop + 1),
                    slice(slc[2].start - 1, slc[2].stop + 1))
        data = np.full((blob.shape[0], blob.shape[1] + 2, blob.shape[2] + 2), 0)
        mask = np.full((blob.shape[0], blob.shape[1] + 2, blob.shape[2] + 2), True)
        self.blob = ma.masked_array(data, mask=mask)
        self.blob.data[0:blob.shape[0], 1:blob.shape[1] + 1, 1:blob.shape[2] + 1] = blob.data
        self.blob.mask[0:blob.shape[0], 1:blob.shape[1] + 1, 1:blob.shape[2] + 1] = blob.mask
        self.rgbcolor = rgbcolors[self.index % ncolors]
        self.rgb_contour = self.make_rgb_contour()
        self.rgb_outline = self.make_rgb_contour(flatten=True)
        self.xc, self.yc, self.tc = None, None, None
        self.projected_area = None
        self.xwidth = None
        self.ywidth = None
        self.duration = None
        self.volume = None
        self.total_intensity = None
        self.mean_intensity = None
        self.max_intensity = None
        self.light_curve = None
        self.self_variance = None
        self.image_coords = np.nan, np.nan
        self.carrington_coords = np.nan, np.nan
        self.stats()
        self.score = self.get_score()

        # Computed later by compute_heights
        self.shift = np.nan, np.nan
        self.height = np.nan
        self.corrcoeff = np.nan
        self.min_segment = np.nan

        self.height_fort = np.full(parent.n_images, np.nan)
        self.min_segment_fort = np.full(parent.n_images, np.nan)
        self.shift_fort = np.full((parent.n_images, 2), np.nan)
        self.corrcoeff_fort = np.full(parent.n_images, np.nan)
        self.image_coords_fort = np.full((parent.n_images, 2), np.nan)
        self.carrington_coords_fort = np.full((parent.n_images, 2), np.nan)

    def make_rgb_contour(self, flatten=False):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        if flatten:
            rgb_contour = np.zeros(self.blob.mask.shape[1:3] + (3,), dtype=np.uint8)
            m = np.uint8((~self.blob.mask).sum(axis=0) > 0)
            mask = (cv2.dilate(m, kernel) - m) == 1
            for c in range(3):
                rgb_contour[:, :, c][mask] = self.rgbcolor[c]
        else:
            rgb_contour = np.zeros(self.blob.mask.shape + (3,), dtype=np.uint8)

            for i, m in enumerate(self.blob.mask):
                m = np.uint8(~m)
                mask = (cv2.dilate(m, kernel) - m) == 1
                for c in range(3):
                    rgb_contour[i, :, :, c][mask] = self.rgbcolor[c]

        return rgb_contour

    def get_score(self):
        mask = (~self.blob.mask).sum(axis=0) > 0
        relative_variance = self.parent_stack.get_relative_variance()
        return np.mean(relative_variance[self.slc[1:]][mask])

    def stats(self):
        self.tc = (self.slc[0].start + self.slc[0].stop) / 2
        self.yc = (self.slc[1].start + self.slc[1].stop) / 2
        self.xc = (self.slc[2].start + self.slc[2].stop) / 2
        self.projected_area = ((~self.blob.mask).sum(axis=0) > 0).sum()
        self.xwidth = self.slc[2].stop - self.slc[2].start - 2
        self.ywidth = self.slc[1].stop - self.slc[1].start - 2
        self.volume = (~self.blob.mask).sum()
        self.duration = self.slc[0].stop - self.slc[0].start
        self.total_intensity = self.blob.sum()
        self.mean_intensity = self.blob.mean()
        self.max_intensity = self.blob.max()
        self.tmax, self.ymax, self.xmax = np.unravel_index(np.argmax(self.blob), shape=self.blob.shape)
        self.tmax += self.slc[0].start
        self.ymax += self.slc[1].start
        self.xmax += self.slc[2].start
        self.light_curve = self.blob.mean(axis=(1, 2))
        self.ellipse_parameters = self.ellipse_properties()
        self.variance = self.light_curve.var()
        t, y, x = np.indices(self.blob.shape)
        xbary = int(round(np.sum(self.blob * x) / np.sum(self.blob)))
        ybary = int(round(np.sum(self.blob * y) / np.sum(self.blob)))
        tbary = int(round(np.sum(self.blob * t) / np.sum(self.blob)))
        self.barintensity = self.blob[tbary, ybary, xbary]
        self.xbary = self.slc[2].start + xbary
        self.ybary = self.slc[1].start + ybary
        self.tbary = self.slc[0].start + tbary
        self.relative_variance = self.variance / self.light_curve.mean()
        hd1 = self.parent_stack.images[0].header
        if "MAPPINGR" in hd1:
            transform = rectify.CarringtonTransform(hd1, radius_correction=hd1["MAPPINGR"] / astropy.constants.R_sun.value)
            lon1 = (self.xmax - hd1["CACRPIX1"] + 1) * hd1["CACDELT1"] + hd1["CACRVAL1"]
            lat1 = (self.ymax - hd1["CACRPIX2"] + 1) * hd1["CACDELT2"] + hd1["CACRVAL2"]
            image_coords = transform(x=lon1, y=lat1)
            self.image_coords = float(image_coords[0]), float(image_coords[1])
            self.carrington_coords = lon1, lat1

    def ellipse_properties(self):
        mask = ~self.blob.mask
        area = mask.sum(axis=(1, 2))
        s = area.argmax()
        props = regionprops(np.uint8(mask[s]), intensity_image=self.blob.data[s])
        self.feret_diameter = props[0].feret_diameter_max
        major = props[0].major_axis_length
        if major == 0:
            major = 1
        minor = props[0].minor_axis_length
        if minor == 0:
            minor = 1
        angle = props[0].orientation

        return major, minor, angle

    def isinframe(self, fov, fnum=None):
        if fov is None:
            isinframe = True
        else:
            isinframe = (self.slc[2].stop > fov[1].start) and (self.slc[2].start < fov[1].stop) and \
                        (self.slc[1].stop > fov[0].start) and (self.slc[1].start < fov[0].stop)
        if fnum is not None:
            isinframe = isinframe and self.slc[0].start <= fnum < self.slc[0].stop
        return isinframe

    def get_center_at_t(self, t, position_type='bary'):
        """ Compute the position of the center of the event at that timestep,
        either looking at maximum intensity position or at intensity-weighted barycenter"""

        if position_type == "max":

            blobstart = self.slc[0].start
            blobend = self.slc[0]
            blobt = self.blob[t - blobstart, ...]
            y, x = np.unravel_index(np.argmax(blobt), shape=blobt.shape)

            x += self.slc[2].start
            y += self.slc[1].start

            return x, y

        else:
            print("TBD")
            return 0, 0
