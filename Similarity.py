import cv2 as cv
from matplotlib import pyplot as plt


# Oriented FAST and Rotated BRIEF Algorithm
def orbfunction(First, Second):
    # Initiate ORB detector
    orb = cv.ORB_create()
    # compute the descriptors with ORB and finding the keypoints
    kp1, des1 = orb.detectAndCompute(First, None)
    kp2, des2 = orb.detectAndCompute(Second, None)
    # In the code below from Line 35 to Line 46 we detect how similar two images are.
    # Considering that high quality images (high quality in this case it means high number of pixels)
    # might have thousands of features so thousands of keypoints while low quality images might have only a few hundreds
    # we need to find a proportion between the matches found and the keypoints.
    # We check the number of keypoints of both images using len(kp_1) and len(kp_2)
    # and we take the number of the images that has less keypoints.
    if len(kp1) <= len(kp2):
        number_keypoints = len(kp1)
    else:
        number_keypoints = len(kp2)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    # In this part we apply the ratio test to select only the good matches.
    # The quality of a match is define by the distance. The distance is a number, and the lower this number is
    # the more similar the features are.
    # By applying the ratio test we can decide to take only the matches with lower distance, so higher quality.
    # If you decrease the ratio value, for example to 0.1 you will get really high quality matches,
    # but the downside is that you will get only few matches.
    # If you increase it you will get more matches but sometimes many false ones. our ratio is 0.95
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good.append([m])
    # draw only matching keypoints location,not size and orientation
    img3 = cv.drawMatchesKnn(First, kp1, Second, kp2, good, None, flags=2)
    plt.imshow(img3), plt.show()

    # Finally we divide the good matches by the number of keypoints.
    # We will get a number between 0 (if there were no matches at all) and 1 (if all keypoints were a match)
    # and then we multiply them by 100 to have a percentage score.
    print("Matches:", len(good))
    print("How good it's the match: ", len(good) / number_keypoints * 100, "%")


first = cv.imread('Img1.jpg', 0)
second = cv.imread('Img2.jpg', 0)
resized = cv.imread("ResizedImg2.jpg", 0)
print(first.shape)
print(second.shape)
print(resized.shape)

im_gray = cv.imread('Img1.jpg', cv.IMREAD_GRAYSCALE)
# if the pixel is smaller than 100 it is set to zero and if its greater than 255 it will be set to 1
_, th1 = cv.threshold(im_gray, 100, 255, cv.THRESH_BINARY)
cv.imwrite('bw_image.png', th1)
orbfunction(first, second)
orbfunction(first, resized)
