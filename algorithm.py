'''
Author: Lennon Seiders
'''

import h5py
import numpy as np
from scipy import ndimage
import skimage
import imaging
from math import isclose
import os
import math
import cv2

'''
Algorithm for second peak removal. Run this script and modify settings in Main() 
in order to segment and remove peaks from diffraction images and/or compare with lab's algorithm.

'''
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
radii_first = [1068, 1160, 1220, 1599, 1904]
radii_second = [1083, 1171, 1232, 1623, 1932]
r_ranges = [(rad1-10, rad2+10) for rad1, rad2 in zip(radii_first, radii_second)]
filter_thres = 10
filter_radius = 3
theta_margin = 0.04
rho_margin = 30

# Convert cartesian coordinates to polar coordinates with center (center_x, center_y)
def cart2pol(x, y, center_x=2063, center_y=2059):
    dx = x - center_x
    dy = y - center_y
    rho = math.sqrt(dx**2 + dy**2)
    theta = math.degrees(math.atan2(dy, dx))
    return (rho, theta)

# Convert polar coordinates with center (center_x, center_y) to cartesian coordinates
def pol2cart(radius, angle, center_x=2063, center_y=2059):
    angle_rad = math.radians(angle)
    x = center_x + radius * math.cos(angle_rad)
    y = center_y + radius * math.sin(angle_rad)
    return (x, y)

# Iterate through directory and save diffraction image data to a numpy array
def load_images_from_h5_folder(folder, num=3600):
    def load_h5_file(file_path):
        f = h5py.File(file_path, 'r')
        img = np.array(f['imageseries']['images'])
        if img.ndim == 3:
            img = img[0]
        return img
    images = []
    i = 0
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".h5"):
            file_path = os.path.join(folder, filename)
            images.append(load_h5_file(file_path))
            i += 1
            if i == num: break
            print (i, end="\r")
    return np.array(images)

# Filter a given image and return a list of peaks for each of the five radii ranges
# 'filter' parameter determines filtering method used: (see imaging.py for more info)
#   'laplace': hexrd's find_peaks_2d()
#   'open': erosion followed by dilation technique
#   'both': opening (erosion and dilation) followed by find_peaks_2d()
def get_pks(img, filter):
    if filter == 'laplace':
        numpks, pks = imaging.find_peaks_2d(img, 'label',{'filter_radius':filter_radius, 'threshold':filter_thres})
    elif filter == 'open':
        numpks, pks = imaging.find_peaks_2d_open(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    elif filter == 'both':
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        numpks, pks = imaging.find_peaks_2d(img, 'label',{'filter_radius':filter_radius, 'threshold':filter_thres})
    else:
        print(filter, 'is not a valid filtering method')
        exit(1)

    newpk_lists = [[], [], [], [], []]
    for pk in pks: 
        pcord = cart2pol(pk[1], pk[0])
        for i in range(0,5):
            if pcord[0] >= r_ranges[i][0] and pcord[0] <= r_ranges[i][1]:
                newpk_lists[i].append(pcord)
                break
    return newpk_lists

# Compares peak lists of two scans and creates a list of peaks from scan_a that should be removed
def find_2pks(scan_a, scan_b):
    # pk_a: candidate for removal
    # pk_b: comparison peak
    def is2nd(pk_a, pk_b):
        return isclose(pk_b[1], pk_a[1], abs_tol=theta_margin) and (pk_b[0] + 5 < pk_a[0])
    
    removepks = []
    for rad_a, rad_b in zip(scan_a, scan_b):
        for pk in rad_a:
            for pk_b in rad_b:
                if is2nd(pk, pk_b):
                    removepks.append(pol2cart(pk[0], pk[1]))
    return removepks    

# Adds 2d array image to file.
#   If mkdir = True: create new '<filename>.h5' file for image data
#   If mdir = False: add new 'removed' image to existing '<filename>.h5'
def im2file(img, filename, mkdir=False):
    if mkdir:
        if os.path.isfile(filename):
            os.remove(filename)
        f = h5py.File(filename, 'w')
        f.create_group("imageseries")
        f['imageseries']['images'] = img
    else:
        f = h5py.File(filename, 'r+')
        f['imageseries']['removed'] = img

# Given a list of peaks to remove, replace each with background sampled from gaussian kde
def remove_2pks(toRemove, img):
    f = h5py.File('background.h5', 'r')
    bg = np.array(f['synthetic_bg'])
    for pk in toRemove:
        img[pk[0]-4:pk[0]+5, pk[1]-4:pk[1]+5] = bg[pk[0]-4:pk[0]+5, pk[1]-4:pk[1]+5]

    return img

# Removes second peaks from images in directory.
#   If overwrite = True: replace each file with a new .h5 file containing image data with peaks removed
#   If overwrite = False: add peaks removed image to each existing file in directory
def find_to_remove(raw_scans):
    scan_pks = [get_pks(scan, 'both') for scan in raw_scans]
    remove_pks = []
    for i in range(len(scan_pks)):
        scan_remove = []
        for j in range(i-3, i+4):
            if j >= 0 and j < len(scan_pks):
                toRemove = find_2pks(scan_pks[i], scan_pks[j])
                scan_remove.extend(toRemove)
        remove_pks.append(scan_remove)
    remove_unique=[np.unique(scan, axis=0) for scan in remove_pks]
    remove_unique = [[[round(pk[0]), round(pk[1])]for pk in pks] for pks in remove_unique]

    return remove_unique

# Runs peak removal algorithm on n='images_to_process' images in a user-defined folder. 
# If used with lab dataset, set img_dev = True in order to compare this algorithm to the 
# algorithm used to create the preprocessed image dataset
def main():
    images_to_process = 100
    folder = 'nobg_2024-03-13-21-19-01-scan Ti7Al_z25p5_deg360_step0p1_rot45'
    #folder = 'synthetic_images'

    write_to_file = True # set to true for creating new preprocessed files
    img_dev = True # set to true when testing lab data
    verbose = True # output second peaks removed in each scan by algorithm.py and not removed by lab algorithm

    raw_scans = load_images_from_h5_folder(folder, images_to_process)
    to_remove = find_to_remove(raw_scans)

    # Compare results of lab algorithm and algorithm.py. 
    # Requires user to run image_dev.py in order to generate deviation images.
    if img_dev:
        lab_found = 0
        algorithm_found = 0
        diff_found = 0
        dev_dir = 'img_dev'
        lab_removed = []
        for i, file in enumerate(os.listdir(dev_dir)):
            if i > images_to_process: break
            f = os.path.join(dev_dir, file)
            f1 = h5py.File(f, 'r+')
            lab_removed.append(np.array(f1['removed_peaks']))

        i = 0
        for lab_scan, found_scan in zip(lab_removed, to_remove):
            i += 1
            scan_diff_lab = []
            scan_diff_algorithm = []
            lab_found += len(lab_scan)
            algorithm_found += len(found_scan)
            # found in this algorithm but not found by lab's
            for pk in found_scan:
                if pk not in lab_scan:
                    scan_diff_algorithm.append(pk)
            diff_found += len(scan_diff_algorithm)
            if verbose:
                print('image', i)
                print('found in algorithm.py but not img_dev: ', scan_diff_algorithm)
        
        print()
        print(images_to_process, 'images tested')
        print('approximate secondary peaks identified by LabHEDM.py:', lab_found)
        print('estimated secondary peaks not removed: ', diff_found, ' (', (diff_found/(lab_found+diff_found))*100, '%)', sep='')

    # Write new files with secondary peaks removed
    if write_to_file:
        if not os.path.exists('processed_images'):
            os.makedirs('processed_images')
        for i in range(0, len(raw_scans)):
            img = remove_2pks(to_remove[i], raw_scans[i])
            fname = os.path.join('processed_images', 'removed_'+ format(i+1, '04') +'.h5')
            im2file(img, fname, mkdir=True)
        
    return

    

if __name__ == "__main__":
    main()



