import cv2
import numpy as np
import os, sys

fullimage = cv2.imread(os.path.join(sys.path[0], "shion.jpg"), cv2.IMREAD_UNCHANGED)
specificimg = cv2.imread(os.path.join(sys.path[0], "face.jpg"), cv2.IMREAD_UNCHANGED)

result = cv2.matchTemplate(fullimage, specificimg, cv2.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

print(f"confidance: {max_val}, location: {str(max_loc)}")
print(specificimg.shape) # (h,w,c) c = channels

top_left = max_loc
bottom_right = (top_left[0] + specificimg.shape[1], top_left[1] + specificimg.shape[0])

threshhold = 0.9
if max_val >= threshhold:
    print("match")
    cv2.rectangle(fullimage, top_left, bottom_right, color=(0,0,255), thickness=2, lineType=cv2.LINE_4)
    cv2.imwrite(os.path.join(sys.path[0],'result.jpg'), fullimage)
else:
    print("no match :c")

#cv2.imshow('result', result)
#cv2.waitKey()