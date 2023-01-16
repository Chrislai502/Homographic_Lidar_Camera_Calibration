import cv2
import numpy as np
from math import sqrt
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------- #
#                                 Instructions                                 #
# ---------------------------------------------------------------------------- #
'''
How to use:
1. Update PARAMETERS section with the correct Image paths
2. Run the script `python3 1.py`
3. Click on the keypoints in the correct order in both images
4. Idea: use SIFT features to suggest keypoints
5. BRIEF is a fast descriptor, but it is not robust to rotation

Rules:
1. pic1 is Lidar / Radar 
2. pic2 is image
3. For Lidar-Lidar/Radar-Lidar Calibration, replace pic1 correct projection images
4. P1 = H @ P2 
5. Frame transformation from pic2 to pic1, original_frame to target_frame
'''

# ---------------------------------------------------------------------------- #
#                                  PARAMETERS                                  #
# ---------------------------------------------------------------------------- #
# pic1_path = "image_undistorted.png"
# pic2_path = "lidar_projection.png"
pic1_path = "lidar_projection.png" 
pic2_path = "image_undistorted.png"


# ---------------------------------------------------------------------------- #
# ---- Create an empty list to store the coordinates of the clicked points --- #
# ---------------------------------------------------------------------------- #
points_1 = []
points_2 = []
counter1 = 0
counter2 = 0
pic1 = cv2.imread(pic1_path)
pic2 = cv2.imread(pic2_path)
img1 = pic1.copy()
img2 = pic2.copy()

# ---------------------------------------------------------------------------- #
#                              Clickback functions                             #
# ---------------------------------------------------------------------------- #
def pic1_clickback(event, x, y, flags, param):
    global img1
    global counter1

    # ---------------------------------------------------------------------------- #
    #                                 LeftClick = 1                                #
    # ---------------------------------------------------------------------------- #
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # ---------------------------------------------------------------------------- #
        #                  Record the coordinate of the clicked point                  #
        # ---------------------------------------------------------------------------- #
        points_1.append([x, y])
        print("Target Frame Points: ", points_1)
        
        
        # ---------------------------------------------------------------------------- #
        #                      Draw the circle and label the point                     #
        # ---------------------------------------------------------------------------- #
        cv2.circle(img1, (x, y), 1, (0, 0, 255), 2)
        cv2.putText(img1, str(counter1), (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        counter1 += 1

    # ---------------------------------------------------------------------------- #
    #                               Middle_click = 1                               #
    # ---------------------------------------------------------------------------- #
    if event == cv2.EVENT_MBUTTONDOWN:
        
        # ---------------------------------------------------------------------------- #
        #                              Undo the last point                             #
        # ---------------------------------------------------------------------------- #
        points_1.pop()
        print("Original Frame Points: ", points_1)
        img1 = pic1.copy()

        # ---------------------------------------------------------------------------- #
        #                Redrawing the remaining points in a fresh image               #
        # ---------------------------------------------------------------------------- #
        for i, point in enumerate(points_1):
            cv2.circle(img1, point, 1, (0, 0, 255), 2)
            cv2.putText(img1, str(i), (point[0]-5, point[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        counter1 -= 1


def pic2_clickback(event, x, y, flags, param):
    global img2
    global counter2

    # ---------------------------------------------------------------------------- #
    #                                 LeftClick = 1                                #
    # ---------------------------------------------------------------------------- #
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # ---------------------------------------------------------------------------- #
        #                  Record the coordinate of the clicked point                  #
        # ---------------------------------------------------------------------------- #
        points_2.append([x, y])
        print("Original Frame Points: ", points_2)
        
        
        # ---------------------------------------------------------------------------- #
        #                      Draw the circle and label the point                     #
        # ---------------------------------------------------------------------------- #
        cv2.circle(img2, (x, y), 1, (255, 0, 0), 2)
        cv2.putText(img2, str(counter2), (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        counter2 += 1

    # ---------------------------------------------------------------------------- #
    #                               Middle_click = 1                               #
    # ---------------------------------------------------------------------------- #
    if event == cv2.EVENT_MBUTTONDOWN:
        
        # ---------------------------------------------------------------------------- #
        #                              Undo the last point                             #
        # ---------------------------------------------------------------------------- #
        points_2.pop()
        print("Original Frame Points: ", points_2)
        img2 = pic2.copy()

        # ---------------------------------------------------------------------------- #
        #                Redrawing the remaining points in a fresh image               #
        # ---------------------------------------------------------------------------- #
        for i, point in enumerate(points_2):
            cv2.circle(img2, point, 1, (255, 0, 0), 2)
            cv2.putText(img2, str(i), (point[0]-5, point[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        counter2 -= 1


# create 2 windows
cv2.namedWindow("pic1_target_frame")
cv2.namedWindow("pic2_original_frame")

# different doubleClick action for each window
# you can send center and color to draw_red_circle via param
# param = [(200,200),(0,0,255)]
cv2.setMouseCallback("pic1_target_frame", pic1_clickback) #, param)
cv2.setMouseCallback("pic2_original_frame", pic2_clickback) # param = None

while True:
    while True:
        # both windows are displaying the same img
        cv2.imshow("pic1_target_frame", img1)
        cv2.imshow("pic2_original_frame", img2)

        # press q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    try:
        assert len(points_1) == len(points_2)
        break
    except:
        print("Error! Number of points in each frame must be the same!")
        continue


# ---------------------------------------------------------------------------- #
#                            Homeography Starts here                           #
# ---------------------------------------------------------------------------- #
#Now let us use these key points to register two images. 
#Can be used for distortion correction or alignment
#For this task we will use homography. 
# https://docs.opencv.org/3.4.1/d9/dab/tutorial_homography.html

# Extract location of good matches.
# For this we will use RANSAC.
#RANSAC is abbreviation of RANdom SAmple Consensus, 
#in summary it can be considered as outlier rejection method for keypoints.
#http://eric-yuan.me/ransac/
#RANSAC needs all key points indexed, first set indexed to queryIdx
#Second set to #trainIdx. 
points_1 = np.array(points_1).reshape((-1, 1, 2))
points_2 = np.array(points_2).reshape((-1, 1, 2))


#Now we have all good keypoints so we are ready for homography.   
# Find homography
#https://en.wikipedia.org/wiki/Homography_(computer_vision)
# RANSAC is used to remove outliers
h, mask = cv2.findHomography(points_1, points_2, cv2.RANSAC)
 
# Use homography
height, width, channels = pic2.shape
im1Reg = cv2.warpPerspective(pic1, h, (width, height))  #Applies a perspective transformation to an image.
cv2.imshow("warped", im1Reg)

print("Estimated homography : \n",  h)

# plt.imshow(stitch_img(pic1, pic2, h))
# cv2.imshow("pic1_target_frame", img1)
im1Reg[0:pic2.shape[0], 0:pic2.shape[1]] = pic2*0.5 + im1Reg*0.5 #stitched image
cv2.imshow("stiched", im1Reg)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
