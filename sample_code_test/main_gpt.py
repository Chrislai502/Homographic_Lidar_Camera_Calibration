import cv2
import numpy as np

# Load the two images
img1 = cv2.imread("c.png")
img2 = cv2.imread("d.png")

# Define the keypoints and descriptors for both images
# sift = cv2.xfeatures2d.SIFT_create()
sift= cv2.SIFT_create()  # depends on OpenCV version
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Use FLANN matcher to find the matches between the two images
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Filter the matches using the Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Extract the keypoints and coordinates of the good matches
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

# Find the homography matrix using RANSAC
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Use the homography matrix to warp the second image and combine it with the first image
warped_img = cv2.warpPerspective(img2, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
result = cv2.addWeighted(img1, 0.5, warped_img, 0.5, 0)

# Display the stitched image
cv2.imshow("Stitched Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()