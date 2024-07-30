import cv2
import numpy as np

# ==================================================================================================================

def color_select(cls):

    if cls == 1 or cls == 'VEHICLE':
      color = (0, 255, 0)     # Green

    elif cls == 2 or cls == 'PEDESTRIAN':
      color = (0, 0, 255)     # Red

    elif cls == 3 or cls == 'SIGN':
      color = (255, 255, 0)   # Cyan

    elif cls == 4 or cls == 'CYCLIST':
      color = (0, 255, 255)   # Yellow

    elif cls == 'Van':
      color = (255, 0, 255)   # Purple

    else:                     # Unknown
      color = (255, 255, 255) # White

    return color

# ==================================================================================================================

def draw_box_2d(image, corners, cls):

  c = color_select(cls)
  cv2.line(image, (corners[0][0], corners[1][0]), (corners[0][1], corners[1][1]), c, 1, lineType=cv2.LINE_AA)
  cv2.line(image, (corners[0][1], corners[1][1]), (corners[0][2], corners[1][2]), c, 1, lineType=cv2.LINE_AA)
  cv2.line(image, (corners[0][2], corners[1][2]), (corners[0][3], corners[1][3]), c, 1, lineType=cv2.LINE_AA)
  cv2.line(image, (corners[0][3], corners[1][3]), (corners[0][0], corners[1][0]), c, 1, lineType=cv2.LINE_AA)

  # Draw the heading of the BEV bbox (only on BEV map)
  center_x = np.mean(corners[0])
  center_y = np.mean(corners[1])
  heading_center_x = (corners[0][0] + corners[0][3]) / 2
  heading_center_y = (corners[1][0] + corners[1][3]) / 2
  cv2.line(image, (int(center_x), int(center_y)), (int(heading_center_x), int(heading_center_y)), c, 1, lineType=cv2.LINE_AA)
  return image
