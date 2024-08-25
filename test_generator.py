'''
Author: Lennon Seiders
'''

import h5py
import random
import numpy as np
from scipy import ndimage
from scipy.stats import gaussian_kde
import imaging
import os
import cv2
import skimage
import background_generator
import math

'''
Functions for generating and testing synthetic image data. Images are saved in .h5 files 
along with a list of primary and secondary peak coordinates.

Set parameter values in main() and run this script in order to generate synthetic images.

'''

# Load peak value data from lab dataset
pkvals = np.loadtxt('pkvals.csv', dtype=np.int16, converters=float)
ints_kde = gaussian_kde(pkvals)

def sample(kde):
    x = kde.resample(1)[0]
    while x < np.min(pkvals):
        x = kde.resample(1)[0]
    return x[0]

# Generate synthetic based on input parameters
def generate_radial_peaks_image(peaks_per_level, newImageFile, p_second, p_tail, size_mult):
    size = (4096, 4096)    
    image = np.zeros(size, dtype=np.int16)
    center = (2063, 2059)
    radii_first = [1068, 1160, 1220, 1599, 1904]
    radii_second = [1083, 1171, 1232, 1623, 1932]

    # Create a 2d peak array. Higher peak intensity correlated with larger size.
    def create_peak_structure(coord, size, peakIntensity):
        mdl = int(size/2)
        chunk = np.zeros((9,9))
        peak = np.random.randint(0, peakIntensity, (3,3), dtype=np.int16)
        peak[1][1] = peakIntensity
        if size > 4: peak = np.pad(peak, (mdl-1,), mode='linear_ramp', end_values=np.random.randint(0, np.min(peak)+1, dtype=np.int16))
        peak = peak * cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(size,size))
        chunk[4-(mdl):5+(mdl),4-(mdl):5+(mdl)] = peak
        image[coord[0]-4:coord[0]+5, coord[1]-4:coord[1]+5] = chunk
    
    # Place peaks on each radius. Peaks_per_level primary peaks + secondary peaks generated with probability p_second.
    primary_peaks = []
    secondary_peaks = []
    for i, radius in enumerate(radii_first):
        for _ in range(peaks_per_level):
            angle = np.random.uniform(0, 2 * np.pi)
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            intensity = sample(ints_kde)
            size = 3 + int(ints_kde.integrate_box_1d(0, intensity) * 2 * size_mult) * 2 
            create_peak_structure([x,y], size, intensity)
            primary_peaks.append((x, y))
            if random.random() < p_second:
                dx = int(center[0] + radii_second[i] * np.cos(angle))
                dy = int(center[1] + radii_second[i] * np.sin(angle))
                create_peak_structure([dx,dy], size, intensity)
                secondary_peaks.append((dx, dy))
                if (random.random() < p_tail) and math.dist([x,y],[dx,dy]) < 13:
                    rr, cc = skimage.draw.line(x,y,dx,dy)
                    mid_x = int((rr[0] - rr[-1]) / 2)
                    mid_y = int((cc[0] - cc[-1]) / 2)
                    image[mid_x, mid_y] = intensity/(math.dist([x,y],[dx,dy])+(7-size))
                    a = np.geomspace(image[x,y], image[mid_x, mid_y], num=int((len(rr)/2) + 1))
                    b = np.geomspace(image[mid_x, mid_y], image[dx, dy], num=int(len(rr)/2))
                    tailvalues = np.hstack((a, b))
                    for j in range(0, len(rr)):
                        image[rr[j], [cc[j]]] = tailvalues[j]

    # Gaussian smoothing and adding  background noise
    image = np.maximum(ndimage.gaussian_filter(image, sigma=0.7), image)
    try: bg_f = np.array(h5py.File('background.h5','r')['synthetic_bg'])
    except: 
        background_generator.main()
        bg_f = np.array(h5py.File('background.h5','r')['synthetic_bg'])
    finally: image += bg_f
    
    # Set synthetic image file
    newImageFile.create_group("imageseries")
    newImageFile['imageseries']['images'] = image
    newImageFile['imageseries']['primaryPeaks'] = primary_peaks
    newImageFile['imageseries']['secondaryPeaks'] = secondary_peaks 
    return

# Create synthetic image file, generate image and peak lists
def generate_images(num_images, peaks_per_level=4, p_second=0.5, p_tail=0.75, size_mult=1):
    print("generating", num_images, "diffraction images...")
    if not os.path.exists('synthetic_images'):
        os.makedirs('synthetic_images')
    for i in range(num_images):
        filename = 'synthetic_image_' + format(i+1, '04') + '.h5'
        if os.path.isfile('synthetic_images/' + filename):
            os.remove('synthetic_images/' + filename)
        newImageFile = h5py.File('synthetic_images/' + filename,'a')
        generate_radial_peaks_image(peaks_per_level, newImageFile, p_second, p_tail, size_mult)

# Test to run different filtering methods on synthetic data. Keeps track of peaks missed and false positives.
def test_synthetic_images(thres, directory='synthetic_images'):
    missed = 0
    false_positive = 0
    correct = 0
    total_pks = 0
    i = 0
    for filename in os.listdir(directory):
        i += 1
        f = os.path.join(directory, filename)
        x, numpks = imaging.find_peaks_2d_test(f, thres)
        total_pks += numpks
        if x > 0: false_positive += x
        elif x < 0: missed -= x
        else: correct += 1
    print(total_pks, 'total peaks tested in', i, 'images')
    print('correct images:', correct)
    print('missed:', missed)
    print('incorrectly classified:', false_positive)



def main():
    num_images = 10 # number of synthetic diffraction images to be generated
    peaks_per_level = 4 # peaks per radius level
    p_second = 0.5 # probability of secondary peaks being generated for each primary peak
    p_tail = 0.5 # probaility of a pair of peaks having additional "tail" noise
    size_mult = 1 # size multiplier. higher value (ex. 1.3) results in a higher chance of generating large peaks
    generate_images(num_images, peaks_per_level, p_second, p_tail, size_mult)

    




if __name__ == "__main__":
    main()

