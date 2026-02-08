import cv2
import numpy as np
import time

def create_map(dif, height, width):

    red_mask = (dif[:,:,0]>200) | (dif[:,:,1]>200) | (dif[:,:,2]>200)

    color_map = np.zeros((height, width, 3), dtype=np.uint8)

    color_map[:,:,1] = 255

    color_map[red_mask,2] = 255
    color_map[red_mask,1] = 0

    return color_map


def otical_flow(prev_frame, curr_frame, wind_size=15):
    height, width = prev_frame.shape
    flow = np.zeros((height, width, 2), dtype=np.float32)
    half_window = wind_size // 2

    prev_frame_float = prev_frame.astype(np.float64)
    curr_frame_float = curr_frame.astype(np.float64)

    for y in range(half_window, height - half_window):
        for x in range(half_window, width - half_window):
            prev_window = prev_frame_float[y - half_window:y + half_window + 1,
                          x - half_window:x + half_window + 1]

            min_error = float('inf')
            best_dx, best_dy = 0, 0
            search_range = 5

            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    xx, yy = x + dx, y + dy

                    if (yy - half_window >= 0 and yy + half_window < height and
                            xx - half_window >= 0 and xx + half_window < width):

                        curr_window = curr_frame_float[yy - half_window:yy + half_window + 1,
                                      xx - half_window:xx + half_window + 1]

                        if prev_window.shape == curr_window.shape:
                            try:
                                error = np.sum((prev_window - curr_window) ** 2) / prev_window.size
                                if error < min_error:
                                    min_error = error
                                    best_dx, best_dy = dx, dy
                            except:
                                continue

            flow[y, x] = [best_dx, best_dy]

    return flow


def create_flow_map(flow, threshold=1.25):

    magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

    color_map = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)

    color_map[:, :, 1] = 255


    motion_mask = magnitude > threshold

    color_map[motion_mask, 1] = 0
    color_map[motion_mask, 2] = 255

    return color_map

cap = cv2.VideoCapture(0)

prev_frame = None
start_time = 0
mode = 0
last_int = 0
while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame,1)

    if prev_frame is None:
        start_time = time.time()

    cur_time = time.time()-start_time

    if cur_time-last_int >= 10:
        mode+=1
        last_int = cur_time

    # if mode%2==0:
    #     cv2.rectangle(frame, (0, 0), (100, 100), (0, 255, 0), -1)
    #     color_map = np.full((480, 640, 3), (0, 255, 0), dtype=np.uint8)
    # else:
    #     cv2.rectangle(frame, (0, 0), (100, 100), (0, 0, 255), -1)
    #
    #     diff = cv2.absdiff(frame,prev_frame)
    #     _, diff = cv2.threshold(diff, 25,255, cv2.THRESH_BINARY)
    #     color_map = create_map(diff,diff.shape[0],diff.shape[1])

    if mode % 2 == 0:
        cv2.rectangle(frame, (0, 0), (100, 100), (0, 255, 0), -1)
        color_map = np.full((480, 640, 3), (0, 255, 0), dtype=np.uint8)
    else:
        cv2.rectangle(frame, (0, 0), (100, 100), (0, 0, 255), -1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            # flow = otical_flow(prev_frame_gray, frame_gray)
            flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray,
            None,pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=2,
                poly_n=5,
                poly_sigma=1.5,
                flags=0
            )
            color_map = create_flow_map(flow)

    cv2.putText(frame, str(round(cur_time,2)), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)

    key = cv2.waitKey(20)
    combined = np.hstack((color_map, frame))

    cv2.imshow('Camera + Motion Map', combined)
    prev_frame = frame
    if key==27:
        break

cap.release()
cv2.destroyWindow("Camera + Motion Map")
