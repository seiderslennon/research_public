'''
Author: Lennon Seiders
'''

import h5py
import numpy as np
from scipy import ndimage
import cv2

'''
This file contains hexrd's (https://github.com/HEXRD/hexrd) find_peaks_2d() function for filtering diffraction images, 
as well as an alternate method and a test to be used with synthetic data. 

These functions are to be used in other scripts for filtering and testing.

'''

sigma_to_fwhm = 2.*np.sqrt(2.*np.log(2.))
fwhm_to_sigma = 1. / sigma_to_fwhm # = 0.42....

# Finds peak structures by using scipy.ndimage.gaussian_laplace(). 
# gaussian_laplace() uses a gaussian filter followed by a second derivative measurement using a laplacian kernel.
def find_peaks_2d(img, method, method_kwargs):
    if method == 'label':
        # labeling mask
        structureNDI_label = ndimage.generate_binary_structure(2, 1)

        # First apply filter if specified
        filter_fwhm = method_kwargs['filter_radius']
        if filter_fwhm:
            filt_stdev = fwhm_to_sigma * filter_fwhm
            img = -ndimage.filters.gaussian_laplace(
                img, filt_stdev
            )

        labels_t, numSpots_t = ndimage.label(
            img > method_kwargs['threshold'],
            structureNDI_label
            )
        coms_t = np.atleast_2d(
            ndimage.center_of_mass(
                img,
                labels=labels_t,
                index=np.arange(1, np.amax(labels_t) + 1)
                )
            )

    return numSpots_t, coms_t

# Finds peak structures by eroding foreground with given kernel and then dilating
def find_peaks_2d_open(img, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), thres=40):
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    structureNDI_label = ndimage.generate_binary_structure(2, 1)

    labels_t, numSpots_t = ndimage.label(img > thres, structureNDI_label)
    coms_t = np.atleast_2d(ndimage.center_of_mass(
        img,
        labels=labels_t,
        index=np.arange(1, np.amax(labels_t) + 1)
        )
    )
    return numSpots_t, coms_t

# Tests find_peaks_2d() on a synthetic diffraction image. Returns peaks missed or false positives.
def find_peaks_2d_test(filename, thres=38):
    print("testing", filename)
    f = h5py.File(filename, 'r+')
    img = np.array(f['imageseries']['images'])
    if img.shape == (3,):
        img = img[0,:,:]
    img_primary_peaks = np.array(f['imageseries']['primaryPeaks'])
    img_secondary_peaks = np.array(f['imageseries']['secondaryPeaks'])
    true_pks =np.append(img_primary_peaks, img_secondary_peaks, axis=0)
    true_pks = np.sort(true_pks, axis=0)

    found_pks = find_peaks_2d(img, 'label',{'filter_radius':3, 'threshold':thres})[1]
    found_pks = np.sort(found_pks, axis=0)

    # Check for shared pairs
    try: 
        identical = np.allclose(true_pks, found_pks, atol=1)
        # if identical: print('correctly identified')
        # else: print('true peaks and peaks found not identical')
        return 0, len(true_pks)
    except: 
        identical = 0
        # if len(true_pks) > len(found_pks): 
        #     print('missing', len(true_pks) - len(found_pks), 'peaks')
        # else: 
        #     print('incorrectly classified', len(found_pks) - len(true_pks), 'peaks')
        return len(found_pks) - len(true_pks), len(true_pks)