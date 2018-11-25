# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 00:17:21 2018

@author: echtpar
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import scipy.misc
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


#print(check_output(["ls", "../input/sample_images/"]).decode("utf8"))


# Some constants 
INPUT_FOLDER = os.getcwd()
print(INPUT_FOLDER)
path = os.path.join(INPUT_FOLDER, "AllLuna16Data\\R_004\\06-30-1997-Diagnostic Pre-Surgery Contrast Enhanced CT-71813\\3- NONE -29295")
patients = os.listdir(path)
print(patients)
patients.sort()


#path = path + patients[0]
#print(path)

#slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]



# Any results you write to the current directory are saved as output.
#lung = dicom.read_file('../input/sample_images/00cba091fa4ad62cc3200a657aeb957e/38c4ff5d36b5a6b6dc025435d62a143d.dcm')


def read_ct_scan(folder_name):
        # Read the slices from the dicom file
        print('folder name:::',folder_name)
        slices = [dicom.read_file(folder_name + '/' + filename) for filename in os.listdir(folder_name)]
        
        # Sort the dicom slices in their respective order
        slices.sort(key=lambda x: int(x.InstanceNumber))
        
        # Get the pixel values for all the slices
        slices = np.stack([s.pixel_array for s in slices])
        slices[slices == -2000] = 0
        return slices
    


slices = read_ct_scan(path)

lung = slices[0]

slice = lung
slice[slice == -2000] = 0
plt.imshow(slice, cmap=plt.cm.gray)


    
def plot_ct_scan(scan, rows, cols, size_x, size_y):
    f, plots = plt.subplots(rows, cols, figsize=(size_x, size_x),sharex=True, sharey=True)
    axes = plots.flat
    for i in range(len(axes)):
        axes[i].axis('off')
        axes[i].imshow(scan[i], cmap=plt.cm.bone)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0, hspace=0)
    plt.tight_layout()
    
    
#    for i in range(0, np.array(scan).shape[0]-4, 4):
#        plots[int((i) / (np.array(scan).shape[0]/4-1)), int(((i) % (np.array(scan).shape[0]/4-1))/4)].axis('off')
#        plots[int((i) / (np.array(scan).shape[0]/4-1)), int(((i) % (np.array(scan).shape[0]/4-1))/4)].imshow(scan[i], cmap=plt.cm.bone) 


plot_ct_scan(slices, 5,5,12,12)        
        


def get_segmented_lungs(im, plot=False):

    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, axes = plt.subplots(2, 4, figsize=(20, 60))
        plots = axes.flat    
        print(len(plots))
    
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < 604
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone) 
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone) 
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = np.array(label_image > 0, dtype=np.int8)
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone) 
        
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone) 

    return im
    
    
    
get_segmented_lungs(slices[42], True)


def segment_lung_from_ct_scan(ct_scan):
    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])


segmented_ct_scan = segment_lung_from_ct_scan(slices)

plot_ct_scan(segmented_ct_scan, 5,5,12,12)


segmented_ct_scan[segmented_ct_scan < 604] = 0
plot_ct_scan(segmented_ct_scan, 5,5,12,12)


selem = ball(2)
binary = binary_closing(segmented_ct_scan, selem)

label_scan = label(binary)

areas = [r.area for r in regionprops(label_scan)]
areas.sort()

for r in regionprops(label_scan):
    max_x, max_y, max_z = 0, 0, 0
    min_x, min_y, min_z = 1000, 1000, 1000
    
    for c in r.coords:
        max_z = max(c[0], max_z)
        max_y = max(c[1], max_y)
        max_x = max(c[2], max_x)
        
        min_z = min(c[0], min_z)
        min_y = min(c[1], min_y)
        min_x = min(c[2], min_x)
    if (min_z == max_z or min_y == max_y or min_x == max_x or r.area > areas[-3]):
        for c in r.coords:
            segmented_ct_scan[c[0], c[1], c[2]] = 0
    else:
        index = (max((max_x - min_x), (max_y - min_y), (max_z - min_z))) / (min((max_x - min_x), (max_y - min_y) , (max_z - min_z)))





def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces, t1, t2 = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()



plot_3d(segmented_ct_scan, 604)        

