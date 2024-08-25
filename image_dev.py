'''
Author: Lennon Seiders
'''

import os
import h5py
import numpy as np
import cv2
import imaging

'''
Creates deviation images by subtracting preprocessed images from raw scans. 
Resulting images are peaks and noise removed by preprocessing algorithm:
  'image_dev': img_pre - img_post
  'dev_filtered': image_dev masked to only show peaks near radii, eroded and then dilated
  'dev_peaks': list of peaks found in image_dev; peaks that have been removed by preprocessing algorithm

Set directory paths and number of images to generate prior to running this script.

'''
num_files = 20 # choose how many deviation images to create from lab dataset

raw_directory = 'nobg_2024-03-13-21-19-01-scan Ti7Al_z25p5_deg360_step0p1_rot45'
prep_directory = 'Ti7Al_deg360_step0p1_after_preprocessing\Ti7Al_z25p5_deg360_step0p1_rot45_0to3600_ver3p1_mult1p6_3600_mimg9'
result_directory = 'img_dev/'

size = (4096, 4096)
center = (2063, 2059)
radii_first = [1068, 1160, 1220, 1599, 1904]
radii_second = [1083, 1171, 1232, 1623, 1932]
mask = np.zeros(size, dtype=bool)
mask2 = np.zeros(size, dtype=bool)
x = np.arange(size[0])
y = np.arange(size[1])
xx, yy = np.meshgrid(x, y)
distances = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
for r1, r2 in zip(radii_first, radii_second):
    mask |= (distances >= r1 - 10) & (distances <= r2 + 10)

# optional function to draw dotted lines along each radius
def set_radii_elements(array, center, radii, value):
    for radius in radii:
        for angle in range(360):
            radian = np.deg2rad(angle)
            x = int(center[0] + radius * np.cos(radian))
            y = int(center[1] + radius * np.sin(radian))
            if 0 <= x < array.shape[0] and 0 <= y < array.shape[1]:
                array[x, y] = value

# Create num_files amount of img_dev images
files1 = sorted(os.listdir(raw_directory))
files2 = sorted(os.listdir(prep_directory))
count = 0
for raw_f, prep_f in zip(files1[:num_files], files2[:num_files]):
    raw = os.path.join(raw_directory, raw_f)
    prep = os.path.join(prep_directory, prep_f)

    f1 = h5py.File(raw, 'r')
    r_img = np.array(f1['imageseries']['images'][0,:,:])
    f1.close()
    f2 = h5py.File(prep, 'r')
    p_img = np.array(f2['imageseries']['images'][0,:,:])
    f2.close()

    new_filename = result_directory + 'dev_' + format(count+1, '04') + '.h5'
    img_dev = r_img - p_img
    
    if os.path.isfile(new_filename):
        os.remove(new_filename)
        
    result = h5py.File(new_filename, 'a')
    result['image_dev'] = img_dev

    count += 1

# Filter img_dev to isolate peaks
for dev_f in os.listdir(result_directory):
    f = os.path.join(result_directory, dev_f)
    f1 = h5py.File(f, 'r+')
    dev_img = np.array(f1['image_dev'])

    masked_image = np.where(mask, dev_img, 0)
    masked_image = cv2.morphologyEx(masked_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    numpks, pks = imaging.find_peaks_2d(masked_image, 'label',{'filter_radius':3, 'threshold':5})
    coords = [np.array([round(pk[1]), round(pk[0])]) for pk in pks]
    f1['dev_filtered'] = masked_image
    f1['removed_peaks'] = coords

    
