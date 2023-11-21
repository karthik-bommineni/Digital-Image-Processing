# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:37:50 2023

@author: karth
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('C:\\Users\\karth\\OneDrive\\Desktop\\Digital Image Processing\\Images\\toji.jpg')

image_copy = np.copy(image)
# print(image_copy.shape)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
# plt.imshow(image_copy)
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

gray = np.float32(gray)
# print(gray)
# plt.imshow(gray)

dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
# plt.imshow(dst, cmap='gray')
thresh = 0.1 * dst.max()
corner_image = np.copy(image_copy)

print(dst.shape)

for i in range (0, dst.shape[0]):
    for j in range(0, dst.shape[1]):
        if (dst[i, j] > thresh):
            cv2.circle(corner_image, (j, i), 1, (0, 255, 0), 1)

plt.axis('off')
plt.imshow(corner_image)
plt.savefig('Images/corner_harris_det.jpg')