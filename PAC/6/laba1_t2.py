import cv2
import numpy as np

img = cv2.imread(r"lab7.png",1)
ghost = cv2.imread(r"img/candy_ghost.png",1)

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ghost_gray = cv2.cvtColor(ghost,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints_img, descriptors_img = sift.detectAndCompute(img_gray, None)
keypoints_ghost, descriptors_ghost = sift.detectAndCompute(ghost_gray, None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_ghost, descriptors_img)
matches = sorted(matches, key=lambda x: x.distance)

good_matches = matches[:int(len(matches)*0.7)]

src_pts = np.float32([keypoints_ghost[i.queryIdx].pt for i in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints_img[i.trainIdx].pt for i in good_matches]).reshape(-1, 1, 2)

M, inline = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

h,w = ghost.shape[:2]

pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

dst = cv2.perspectiveTransform(pts, M)

cv2.polylines(img, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

img = cv2.resize(img, (1000,600))

cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()