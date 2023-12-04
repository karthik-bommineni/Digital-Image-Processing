# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 19:49:55 2023

@author: karth
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#####################################################################

# PRE PROCESSING

image = cv2.imread('C:\\Users\\karth\\OneDrive\\Desktop\\Digital Image Processing\\Images\\breaking_bad.jpg')
image_copy = np.copy(image)
# print(image_copy.shape)
# print(image_copy[1079][1919][2])

# Converting the training image from BGR to RGB (cv2.imread stores the image as BGR by deafault)
training_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
# Convert the training image from RGB to gray scale
training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)

# Creating a test images by adding Scale invaraince and Rotational Invariance
# cv2.pyrDown is used for pyramid downscaling of the image, by taking the average of the 2x2 block of pixels
# cv2.pyrDown is used for introducing a form of blurrness as a part of downsampling process
test_image = cv2.pyrDown(training_image)
test_image = cv2.pyrDown(test_image)

print(test_image.shape)
rows, cols = test_image.shape[:2]
center = (cols/2, rows/2)


rotation_matrix = cv2.getRotationMatrix2D(center, 30, 1) # (center, angle, scaling factor) if scaling factor = 1, then no scaling is applied
test_image = cv2.warpAffine(test_image, rotation_matrix, (cols, rows))
test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

# Displaying training image and testing image
fx, plots = plt.subplots(1, 2, figsize=(20, 10))
plots[0].set_title('Training Image')
plots[0].imshow(training_image)
plots[1].set_title('Tesing Image')
plots[1].imshow(test_image)

plt.savefig('Images/Train_test_images.jpg')


#####################################################################

# DETECTING KEYPOINTS AND CREATE DESCRIPTOR

sift = cv2.xfeatures2d.SIFT_create()
train_keypoints, train_descriptor = sift.detectAndCompute(training_gray, None)
test_keypoints, test_descriptor = sift.detectAndCompute(test_gray, None)

keypoints_without_size = np.copy(training_image)
keypoints_with_size = np.copy(training_image)

cv2.drawKeypoints(training_image, train_keypoints, keypoints_without_size, color = (0, 255, 0))
cv2.drawKeypoints(training_image, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display image with and without keypoints size
fx, plots = plt.subplots(1, 2, figsize=(20, 10))
plots[0].set_title('Training keypoints with size')
plots[0].imshow(keypoints_with_size, cmap='gray')
plots[1].set_title('Training keypoints without size')
plots[1].imshow(keypoints_without_size, cmap='gray')

# Plotting them separately

# plt.title('Training keypoints with size')
# plt.imshow(keypoints_with_size)
# plt.axis('off')
# plt.savefig('Images/train_keypoints_with_size.jpg')

# plt.title('Training keypoints without size')
# plt.imshow(keypoints_without_size)
# plt.axis('off')
# plt.savefig('Images/train_keypoints_without_size.jpg')

# Print the number of keypoints detected in the training image
print("Number of Keypoints Detected in the Training Image: ", len(train_keypoints))
print("Number of Keypoints Detected in the Query Image: ", len(test_keypoints))


#####################################################################

# MATCHING KEYPOINTS

# Create a Brute Force Matches object
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)

# Perform the matching between the SIFT descriptors of the training image and the test image
matches = bf.match(train_descriptor, test_descriptor)

# The matches with shorter distance are the ones we want
matches = sorted(matches, key = lambda x : x.distance)
result = cv2.drawMatches(training_image, train_keypoints, test_gray, test_keypoints, matches[300:600], test_gray, flags = 2)

# Display the best matching points
plt.rcParams['figure.figsize'] = [14.0, 7.0]
plt.title('Best Matching Points')
plt.imshow(result)
plt.savefig('Images/matching_keypoints_few.jpg')
plt.show()


# Print the total number of matching points between the training and query images
print("\nNumber of Matching Keypoints between the training and query images: ", len(matches))