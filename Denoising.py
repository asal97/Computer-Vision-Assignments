import cv2 as cv
import numpy as np
from skimage.util import random_noise

from skimage import color, data, restoration
from scipy.signal import convolve2d



img = cv.imread('cameraman.png', 0)
converted_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

cv.imshow("Normal", img)
cv.imshow("converted", converted_img)

#############################################

# creating salt and pepper noise
noise_img = random_noise(img, mode='s&p', amount=0.05)
# The above function returns a floating-point image
# on the range [0, 1], thus we changed it to 'uint8'
# and from [0,255]
noise_img = np.array(255 * noise_img, dtype='uint8')
# removing salt and pepper noise
desalt = cv.medianBlur(noise_img, 3)

cv.imshow("salt", noise_img)
cv.imshow("desalt", desalt)

############################################

# applying gaussian filter
blur = cv.GaussianBlur(converted_img, (5, 5), 0)
# cleaning noise created by gaussian filter with sharpening kernel
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpen = cv.filter2D(blur, -1, sharpen_kernel)

cv.imshow("GAUSSIAN", blur)
cv.imshow("deGauss", sharpen)

#############################################
# creating poisson noise
poisson = random_noise(img, mode="poisson")
# dePoiss = cv.medianBlur(poisson, 8)
cv.imshow("Poisson", poisson)
# cv.imshow("dePoiss", dePoiss)
#############################################
# creating speckle noise
speckleD = random_noise(img, mode="speckle", mean=0.04)
cv.imshow("speckleD", speckleD)

# cv.imshow("desp", desp)
speckle = random_noise(img, mode="speckle")
cv.imshow("speckle", speckle)


##############################################
psf = np.ones((5, 5)) / 25
poisson = convolve2d(poisson, psf, 'same')
poisson += 0.1 * poisson.std() * np.random.standard_normal(poisson.shape)

filtered_img, _ = restoration.unsupervised_wiener(poisson, psf)

cv.imshow("filtered_img", filtered_img)
cv.waitKey(0)
