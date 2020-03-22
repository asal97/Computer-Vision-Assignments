import cv2
import numpy as np

#read input image
image = cv2.imread("Capture.png")

# define sharpening kernel
# In image processing, a kernel, convolution matrix, or mask is a small matrix
# It is used for blurring, sharpening, embossing, edge detection, and more.
# This is accomplished by doing a convolution between a kernel and an image.
sharpeningKernel = np.array(([0, -1, 0],[-1, 5, -1],[0, -1, 0]), dtype="int")

# filter2D is used to perform the convolution.
# The third parameter (depth) is set to -1 which means the bit-depth of the output image is the
# same as the input image. So if the input image is of type CV_8UC3, the output image will also be of the same type
output = cv2.filter2D(image, -1, sharpeningKernel)

# after sharpening the image  we want to increase the contrast so we use
# our output as the input for our next process

#convert to YCrCb color space
imageYcb = cv2.cvtColor(output, cv2.COLOR_BGR2YCrCb)

# split into channels
Y, C, B = cv2.split(imageYcb)

# histogram equalization which enhances the image
# Y is the luma component and CB and CR are the blue-difference and red-difference
# luma represents the brightness in an image (the "black-and-white" or achromatic portion of the image)
# Luma Controls adjust the brightness of the image and can increase or decrease the contrast of an image
Y = cv2.equalizeHist(Y)

# merge the channels to change the picture back to normal
imageYcb = cv2.merge([Y, C, B])

#convert back to BGR color space
result = cv2.cvtColor(imageYcb, cv2.COLOR_YCrCb2BGR)


cv2.imshow("image", image)
cv2.imshow("result", result)
cv2.imwrite('result.png', result)
cv2.waitKey(0)
