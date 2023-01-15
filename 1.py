import cv2
import numpy as np
from math import sqrt


'''
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
pic1 = cv2.imread(pic1_path)
pic2 = cv2.imread(pic2_path)
pic1_ = cv2.imread(pic1_path)
pic2_ = cv2.imread(pic2_path)

# param contains the center and the color of the circle 
def pic1_clickback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Record the coordinate of the clicked point
        points_1.append((x, y))
        print("Target Frame Points: ", points_1)
        cv2.circle(pic1_, (x, y), 1, param[1], 2)
    if event == cv2.EVENT_RBUTTONDOWN:
        # undo the last point
        points_1.pop()
        print("Original Frame Points: ", points_1)
        pic1_ = pic1
        for point in points_1:
            cv2.circle(pic1_, point, 3, (0, 0, 255), -1)


def pic2_clickback(event, x, y, flags, param):
    # if event == cv2.EVENT_LBUTTONDBLCLK:
    if event == cv2.EVENT_LBUTTONDOWN:
        center = (100,100)
 
        cv2.circle(pic2_, center, 1, (255, 0, 0), 2)

# img = np.zeros((512,512,3), np.uint8)
# img1 = cv2.imread("image_undistorted.png")
# img2 = cv2.imread("image_undistorted.png")

# create 2 windows
cv2.namedWindow("pic1_target_frame")
cv2.namedWindow("pic2_original_frame")

# different doubleClick action for each window
# you can send center and color to draw_red_circle via param
param = [(200,200),(0,0,255)]
cv2.setMouseCallback("pic1_target_frame", pic1_clickback, param)
cv2.setMouseCallback("pic2_original_frame", pic2_clickback) # param = None


while True:
    # both windows are displaying the same img
    cv2.imshow("pic1_target_frame", pic1_)
    cv2.imshow("pic2_original_frame", pic2_)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()