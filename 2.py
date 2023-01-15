import cv2

# Define a function to handle mouse events
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Record the coordinate of the clicked point
        points.append((x, y))
        # Draw a dot on the image at the clicked point
        cv2.circle(img_copy, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("Image", img_copy)
    if event == cv2.EVENT_RBUTTONDOWN:
        # undo the last point
        points.pop()
        img_copy = img.copy()
        for point in points:
            cv2.circle(img_copy, point, 3, (0, 0, 255), -1)
        cv2.imshow("Image", img_copy)


# Load the image
img = cv2.imread("image_undistorted.png")

# Create a copy of the original image to display the dots
img_copy = img.copy()

# Create an empty list to store the coordinates of the clicked points
points = []



try:
    # Create a window to display the image
    cv2.namedWindow("Image")

    # Set the mouse callback function for the window
    cv2.setMouseCallback("Image", click_event)

    # Display the image
    cv2.imshow("Image", img)

    while True:
        # Wait for user input
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
except KeyboardInterrupt:
    # Handle the ctrl+C key combination
    print("Exiting...")
    cv2.destroyAllWindows()



