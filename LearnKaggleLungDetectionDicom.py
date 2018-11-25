# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 23:50:22 2018

@author: Chandrakant Pattekar
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#newpath = "C:\\Users\\echtpar\\Anaconda3\\KerasProjects\\Keras-CNN-Tutorial\\AllLuna16Data\\R_004\\06-30-1997-Diagnostic Pre-Surgery Contrast Enhanced CT-71813\\3- NONE -29295"


def plot_3d(image, threshold=-300):

    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    #p = first_patient_pixels.transpose(2,1,0)
    
    p = pix_resampled.transpose(2,1,0)
    verts, faces, norm, val  = measure.marching_cubes(p, threshold)
    
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    
    plt.show()


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None



# Some constants 
INPUT_FOLDER = os.getcwd()
print(INPUT_FOLDER)
path = os.path.join(INPUT_FOLDER, "AllLuna16Data\\R_004\\06-30-1997-Diagnostic Pre-Surgery Contrast Enhanced CT-71813\\3- NONE -29295")
patients = os.listdir(path)
print(patients)
patients.sort()


#path = path + patients[0]
#print(path)

slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
y = [x.ImagePositionPatient for x in slices]
print(y)

print(len(slices))

print("look at attributes of slices")
print("============================")
q = [x for x in dir(slices)]
print(dir(slices))

print("\n look at attributes of individual slice")
print("============================")
print([y for y in dir(slices[0])])

print('AcquisitionNumber',slices[0].AcquisitionNumber)
print('BitsAllocated',slices[0].BitsAllocated)
print('BitsStored',slices[0].BitsStored)
print('Columns',slices[0].Columns)
print('FrameOfReferenceUID', slices[0].FrameOfReferenceUID)
print('HighBit',slices[0].HighBit)
print('ImageOrientationPatient',slices[0].ImageOrientationPatient)
print('ImagePositionPatient',slices[0].ImagePositionPatient)
print('InstanceNumber',slices[0].InstanceNumber)
print('KVP',slices[0].KVP)
print('Modality',slices[0].Modality)
print('PatientBirthDate',slices[0].PatientBirthDate)
print('PatientID',slices[0].PatientID)
print('PatientName',slices[0].PatientName)
print('PatientOrientation',slices[0].PatientOrientation)
print('PhotometricInterpretation',slices[0].PhotometricInterpretation)
print('PixelData length',len(slices[0].PixelData))
print('PixelPaddingValue',slices[0].PixelPaddingValue)
print('PixelRepresentation',slices[0].PixelRepresentation)
print('PixelSpacing',slices[0].PixelSpacing)
print('PositionReferenceIndicator',slices[0].PositionReferenceIndicator)
print('RescaleIntercept',slices[0].RescaleIntercept)
print('RescaleSlope',slices[0].RescaleSlope)
print('Rows',slices[0].Rows)
print('SOPClassUID',slices[0].SOPClassUID)
print('SOPInstanceUID',slices[0].SOPInstanceUID)
print('SamplesPerPixel',slices[0].SamplesPerPixel)
print('SeriesDescription',slices[0].SeriesDescription)
print('SeriesInstanceUID',slices[0].SeriesInstanceUID)
print('SeriesNumber',slices[0].SeriesNumber)
print('SliceLocation',slices[0].SliceLocation)
print('SpecificCharacterSet',slices[0].SpecificCharacterSet)
print('StudyInstanceUID',slices[0].StudyInstanceUID)
print('WindowCenter',slices[0].WindowCenter)
print('WindowWidth',slices[0].WindowWidth)



slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
try:
    slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
except:
    slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
for s in slices:
        s.SliceThickness = slice_thickness
        
first_patient = slices

print(slices[0])

print(slices[0].pixel_array.shape)

image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
for slice_number in range(len(slices)):
        
    intercept = slices[slice_number].RescaleIntercept
    slope = slices[slice_number].RescaleSlope
        
    if slope != 1:
        image[slice_number] = slope * image[slice_number].astype(np.float64)
        image[slice_number] = image[slice_number].astype(np.int16)
            
    image[slice_number] += np.int16(intercept)
    
first_patient_pixels = np.array(image, dtype=np.int16)


f, ax = plt.subplots(10,5, figsize=(25,25))

axes = ax.flat

for i, x in enumerate(axes):
    x.imshow(first_patient_pixels[i-1], cmap=plt.cm.gray)
    x.axis("off")

plt.show()


fig = plt.figure()
plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

# Show some slice in the middle
fig = plt.figure()
plt.imshow(first_patient_pixels[67], cmap=plt.cm.gray)
plt.show()



print(first_patient[0].SliceThickness)

print(first_patient[0].PixelSpacing)

print(first_patient_pixels.shape)



print(len(first_patient))
print(first_patient[0].pixel_array.shape)
print(type(first_patient))

print(len(first_patient_pixels))
print(first_patient_pixels.shape)
print(type(first_patient_pixels))



image = first_patient_pixels
scan = first_patient
new_spacing=[1,1,1]


spacing = np.hstack([[first_patient[0].SliceThickness], first_patient[0].PixelSpacing])
spacing = np.array(spacing, dtype=np.float32)
print(spacing)
print(type(spacing))

resize_factor = spacing / new_spacing
print(resize_factor, spacing, new_spacing)


new_real_shape = image.shape * resize_factor
print(new_real_shape)


new_shape = np.round(new_real_shape)
print(new_shape)

real_resize_factor = new_shape / image.shape
print(real_resize_factor)


new_spacing = spacing / real_resize_factor
print(new_spacing)


image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
pix_resampled, spacing = image, new_spacing


print(pix_resampled.shape)
print(spacing)


print("Shape before resampling\t", first_patient_pixels.shape)
print("Shape after resampling\t", pix_resampled.shape)

plot_3d(pix_resampled, 400)    


image = pix_resampled
print(np.unique(image))
fill_lung_structures=True
# not actually binary, but 1 and 2. 
# 0 is treated as background, which we do not want
binary_image = np.array(image > -320, dtype=np.int8)+1
print(binary_image.shape)
print(image.shape)


print(np.unique(binary_image))
print(np.unique(image))


labels = measure.label(binary_image)
print(len([x.shape for x in labels]))

print(len(np.unique(labels)))
#print(labels[1,1,1])
print(binary_image[labels == 100])
    
# Pick the pixel in the very corner to determine which label is air.
#   Improvement: Pick multiple background labels from around the patient
#   More resistant to "trays" on which the patient lays cutting the air 
#   around the person in half
background_label = labels[0,0,0]
print(background_label)

    
#Fill the air around the person
binary_image[background_label == labels] = 2
z = measure.label(binary_image)
print(len(np.unique(z)))
print(binary_image[z == 100])
    

for i, x in enumerate(binary_image):
    print(i,x.shape)
    print(np.unique(measure.label(x-1), return_counts=True))
    #print(measure.label(x-1)[0].shape)
    #print(measure.label(x-1)[1].shape)
    
    
    
# Method of filling the lung structures (that is superior to something like 
# morphological closing)
if fill_lung_structures:
    # For every slice we determine the largest solid structure
    for i, axial_slice in enumerate(binary_image):
        axial_slice = axial_slice - 1
        labeling = measure.label(axial_slice)

        #####
        im = labeling
        bg = 0
        vals, counts = np.unique(im, return_counts=True)
        print(vals,counts)
        counts = counts[vals != bg]
        vals = vals[vals != bg]

        if len(counts) > 0:
            l_max = vals[np.argmax(counts)]
        else:
            l_max = None        
        
        #####
        
        if l_max is not None: #This slice contains some lung
            binary_image[i][labeling != l_max] = 1

    
binary_image -= 1 #Make the image actual binary
binary_image = 1-binary_image # Invert it, lungs are now 1
    
# Remove other air pockets insided body
labels = measure.label(binary_image, background=0)
l_max = largest_label_volume(labels, bg=0)
if l_max is not None: # There are air pockets
    binary_image[labels != l_max] = 0
 
segmented_lungs = binary_image


plot_3d(segmented_lungs, 0)






def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices



def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)





def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    #spacing = np.array(np.array(scan[0].SliceThickness) + np.array(scan[0].PixelSpacing[0]), dtype=np.float32)

    spacing = np.hstack([[scan[0].SliceThickness], scan[0].PixelSpacing])
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing



pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
print("Shape before resampling\t", first_patient_pixels.shape)
print("Shape after resampling\t", pix_resampled.shape)




    
    




def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image

segmented_lungs = segment_lung_mask(pix_resampled, False)
segmented_lungs_fill = segment_lung_mask(pix_resampled, True)


plot_3d(segmented_lungs_fill, 0)
plot_3d(segmented_lungs_fill - segmented_lungs, 0)




