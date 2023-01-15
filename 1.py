import cv2
import numpy as np
from math import sqrt

# ---------------------------------------------------------------------------- #
#                                 Instructions                                 #
# ---------------------------------------------------------------------------- #
'''
How to use:
1. Update PARAMETERS section with the correct Image paths
2. Run the script `python3 1.py`
3. Click on the keypoints in the correct order in both images

Rules:
1. pic1 is image
2. pic2 is Lidar / Radar
3. For Lidar-Lidar/Radar-Lidar Calibration, replace pic1 correct projection images
4. P1 = H @ P2 
5. Frame transformation from pic2 to pic1, original_frame to target_frame
'''

# ---------------------------------------------------------------------------- #
#                                  PARAMETERS                                  #
# ---------------------------------------------------------------------------- #
pic1_path = "image_undistorted.png"
pic2_path = "lidar_projection.png"


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
    if event == cv2.EVENT_LBUTTONDOWN:
        # ---------------------------------------------------------------------------- #
        #                  Record the coordinate of the clicked point                  #
        # ---------------------------------------------------------------------------- #
        points_1.append((x, y))
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
    if event == cv2.EVENT_LBUTTONDOWN:
        # ---------------------------------------------------------------------------- #
        #                  Record the coordinate of the clicked point                  #
        # ---------------------------------------------------------------------------- #
        points_2.append((x, y))
        print("Target Frame Points: ", points_2)
        
        
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
    # both windows are displaying the same img
    cv2.imshow("pic1_target_frame", img1)
    cv2.imshow("pic2_original_frame", img2)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

# ---------------------------------------------------------------------------- #
#                            Homeography Starts here                           #
# ---------------------------------------------------------------------------- #
orb = cv2.ORB_create(25)  #Registration works with at least 50 points
kp1, des1 = orb.detectAndCompute(pic1, None)  #kp1 --> list of keypoints
print(kp1)
print(des1)

# ---------------------------------------------------------------------------- #
# ----- Brute-Force matcher takes the descriptor of one feature in first ----- #
# ---- set and is matched with all other features in second set using some --- #
# --------------------------- distance calculation. -------------------------- #
# ---------------------------------------------------------------------------- #

# create Matcher object
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

# Match descriptors.
matches = matcher.match(des1, des2, None)  #Creates a list of all matches, just like keypoints

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

#Like we used cv2.drawKeypoints() to draw keypoints, 
#cv2.drawMatches() helps us to draw the matches. 
#https://docs.opencv.org/3.0-beta/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html
# Draw first 10 matches.
img3 = cv2.drawMatches(im1,kp1, im2, kp2, matches[:10], None)

cv2.imshow("Matches image", img3)
cv2.waitKey(0)
