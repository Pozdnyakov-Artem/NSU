import os
import cv2
import numpy as np


def is_valid_bounding_box(dst, img_shape, max_size_ratio=0.8):

    h_img, w_img = img_shape[:2]

    points = dst.reshape(-1, 2)

    x_coords = points[:, 0]
    y_coords = points[:, 1]

    # print(x_coords)
    # print(y_coords)
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    width = x_max - x_min
    height = y_max - y_min

    width1 = abs(x_coords[1]-x_coords[2])
    width2 = abs(x_coords[0]-x_coords[3])

    if not(x_min >= 0 and y_min >= 0):
        return False
    elif not(x_max <= w_img and y_max <= h_img):
        return False
    elif not(width1 > 80 and  width2 > 80):
        return False

    return True

def find_ghost(mas_of_ghosts):
    img = cv2.imread(r"lab7.png", 1)

    sift = cv2.SIFT_create()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h_img, w_img = img_gray.shape
    mask_img = np.ones((h_img, w_img), dtype=np.uint8) * 255

    img2 = img.copy()

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    for path_to_ghost in mas_of_ghosts:

        ghost = cv2.imread(r"img/" + path_to_ghost, 1)

        for _ in range(2):

            ghost_gray = cv2.cvtColor(ghost, cv2.COLOR_BGR2GRAY)

            while True:
                keypoints_img, descriptors_img = sift.detectAndCompute(img_gray, mask_img)
                keypoints_ghost, descriptors_ghost = sift.detectAndCompute(ghost_gray, None)

            # print(keypoints_ghost)
            # print(descriptors_ghost)

                matches = bf.match(descriptors_ghost, descriptors_img)

                matches = sorted(matches, key = lambda x:x.distance)

                # img2 = cv2.drawMatches(img, keypoints_img, ghost, keypoints_ghost,matches, img, )

                good_mathces = matches[:int(len(matches)*0.7)]

                src_pts = np.float32([keypoints_ghost[m.queryIdx].pt for m in good_mathces]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints_img[m.trainIdx].pt for m in good_mathces]).reshape(-1, 1, 2)

                M, mask2 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                h,w = ghost.shape[:2]

                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                dst = cv2.perspectiveTransform(pts, M)
                if(is_valid_bounding_box(dst, img_gray.shape, max_size_ratio=0.8)):
                    # print()
                    cv2.polylines(img2, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.fillPoly(mask_img, [np.int32(dst)], 0)
                    print("нарисовал")
                else:
                    break

            ghost = cv2.flip(ghost, 1)
        print("-----------------------------------")
        # break
    mask_img = cv2.resize(mask_img, (1000,600))
    cv2.imshow("ghost", mask_img)
    return img2


arr_of_ghosts = os.listdir("img")
# arr_of_ghosts = [r"candy_ghost.png"]
img2 = find_ghost(arr_of_ghosts)

img2 = cv2.resize(img2,(1000,600))
cv2.imshow("img2", img2)
cv2.waitKey(0)