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
plt.imshow(image_copy)
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
# plt.imshow(gray)








plt.axis('off')
# plt.imshow(image)