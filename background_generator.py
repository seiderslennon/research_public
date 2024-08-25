'''
Author: Lennon Seiders
'''

import h5py
import os
import numpy as np
import scipy.stats as stats

'''
Script for obtaining accurate background noise by sampling windowed kernel desnsity estimates of lab dataset.
Run this scripy to create a file titled 'background.h5', synthetic background noise mimicking that 
of the lab dataset's images.

'''

directory = 'nobg_2024-03-13-21-19-01-scan Ti7Al_z25p5_deg360_step0p1_rot45'
num_images = 1
if os.path.isfile('gaussian_kde_full.h5'):
    os.remove('gaussian_kde_full.h5')

# Creates mask over peak radii in order to remove peaks from background data
def set_points_along_radii(img):
    size = img.shape
    center = (2063, 2059)
    radii_first = [1068, 1160, 1220, 1599, 1904]
    radii_second = [1083, 1171, 1232, 1623, 1932]
    x = np.arange(size[0])
    y = np.arange(size[1])
    xx, yy = np.meshgrid(x, y)

    distances = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    for r1, r2 in zip(radii_first, radii_second):
        mask = (distances >= r1 - 15) & (distances <= r2 + 15)
        img[mask] = -1
    
    return img

# Get lab dataset file for background sampling
def get_background(path):
    file = h5py.File(path, 'r')
    img = np.array(file['imageseries']['images'], dtype=np.int16)
    if img.shape == (3,):
        img = img[0,:,:]
    img = np.squeeze(img, axis=0)
    return set_points_along_radii(img)

# Iterate over 2^windowing_exp windows of img,
# Create and sample a gaussian kernel density estimate at each window to get new background
def background_to_gaussians(img, windowing_exp=5):
    size_x = 4096
    kde_list = []
    num_windows = pow(2, windowing_exp)
    window_size = int(size_x/num_windows)

    for window_row in range(num_windows):
        for window_col in range(num_windows):
            start_row = window_size * window_row
            end_row = start_row + window_size
            start_col = window_size * window_col
            end_col = start_col + window_size
            sample_values = img[start_row:end_row, start_col:end_col].flatten().tolist()
            weights = [1 if x != -1 else 0 for x in sample_values]

            kde = stats.gaussian_kde(sample_values, weights = weights)
            kde_list.append(kde)
            new_samples = np.array(kde.resample((window_size)*(window_size)).flatten().tolist(), dtype=np.int16)
            new_samples[new_samples < 0] = 0
            new_samples[new_samples > 100] = 0
            noise_image = new_samples.reshape(window_size, window_size)

            img[start_row:end_row, start_col:end_col] = noise_image

    return img

# Run script
def main():
    path = 'nobg_2024-03-13-21-19-01-scan Ti7Al_z25p5_deg360_step0p1_rot45/Raw_scan_0001.h5'
    img = get_background(path)
    if os.path.isfile('background.h5'):
        os.remove('background.h5')
    newImageFile = h5py.File('background.h5','a')
    newImageFile['synthetic_bg'] = background_to_gaussians(img)

    return

if __name__ == "__main__":
    main()
