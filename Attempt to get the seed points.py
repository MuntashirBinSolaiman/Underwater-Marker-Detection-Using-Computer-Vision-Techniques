import cv2
import numpy as np

# Global variable to store selected points
selected_points = []

# Mouse callback function to capture clicks
def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Append clicked point to the selected_points list
        selected_points.append((x, y))
        print(f"Point selected: ({x}, {y})")

# Load the image
image = cv2.imread('/data/shared/CSIT_Placement_2025_3D_Reef/CBHE_BA2D_P1/images/frame_00001.JPG')

# Set up the window for interaction
cv2.imshow("Select Points", image)
cv2.setMouseCallback("Select Points", select_points)

# Wait for the user to click on the image
cv2.waitKey(0)
cv2.destroyAllWindows()

# Now 'selected_points' will contain all the points that the user clicked on
print("Selected seed points:", selected_points)
