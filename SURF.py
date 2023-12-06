import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#####################################################################

# PRE PROCESSING

root = os.getcwd()
imgPath = os.path.join(root, 'Images\\office.jpg')

# Load the image
image = cv2.imread(imgPath)

# Conver the training image to RGB
training_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the training image to gray scale
training_gray = cv2.cvtColor(training_img, cv2.COLOR_RGB2GRAY)

# Creating a test images by adding Scale invaraince and Rotational Invariance
# cv2.pyrDown is used for pyramid downscaling of the image, by taking the average of the 2x2 block of pixels
# cv2.pyrDown is used for introducing a form of blurrness as a part of downsampling process
test_img = cv2.pyrDown(training_img)
test_img = cv2.pyrDown(test_img)

print(test_img.shape)
rows, cols = test_img.shape[:2]
center = (cols/2, rows/2)

rotation_matrix = cv2.getRotationMatrix2D(center, 30, 1) # (center, angle, scaling factor) if scaling factor = 1, then no scaling is applied
test_img = cv2.warpAffine(test_img, rotation_matrix, (cols, rows))
test_gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)

# Displaying training image and testing image
fx, plots = plt.subplots(1, 2, figsize=(20, 10))
plots[0].set_title('Training Image')
plots[0].imshow(training_img)
plots[1].set_title('Tesing Image')
plots[1].imshow(test_img)

plt.show()

plt.savefig('Images/Train_test_images_office.jpg')

#####################################################################

# DETECTING KEYPOINTS AND CREATE DESCRIPTOR

HessianThreshold = 800
surf = cv2.xfeatures2d.SURF_create(HessianThreshold) # 800 here is the Hessian Threshold. The Hessian matrix is used in the computation of the key points. 
# The higher the threshold, the fewer key points will be detected. Adjusting this threshold allows you to control the sensitivity of the detector.

train_keypoints, train_descriptor = surf.detectAndCompute(training_gray, None)
test_keypoints, test_descriptor = surf.detectAndCompute(test_gray, None) # None here is the argument for the mask

keypoints_without_size = np.copy(training_img)
keypoints_with_size = np.copy(training_img)

cv2.drawKeypoints(training_img, train_keypoints, keypoints_without_size, color = (0, 255, 0))
cv2.drawKeypoints(training_img, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display image with and without keypoints size
fx, plots = plt.subplots(1, 2, figsize=(20, 10))
plots[0].set_title('Training keypoints with size')
plots[0].imshow(keypoints_with_size, cmap='gray')
plots[1].set_title('Training keypoints without size')
plots[1].imshow(keypoints_without_size, cmap='gray')

plt.show()

plt.savefig('Images/training_keypoints_office.jpg')

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
result = cv2.drawMatches(training_img, train_keypoints, test_gray, test_keypoints, matches, test_gray, flags = 2)

# Display the best matching points
plt.rcParams['figure.figsize'] = [14.0, 7.0]
plt.title('Best Matching Points')
plt.imshow(result)
plt.savefig('Images/matching_keypoints_office.jpg')
plt.show()


# Print the total number of matching points between the training and query images
print("\nNumber of Matching Keypoints between the training and query images: ", len(matches))
