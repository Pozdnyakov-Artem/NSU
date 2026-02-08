import cv2

cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    key = cv2.waitKey(10)
    cv2.imshow('Frame', frame)
    if key == 27:  # Esc
        break

cv2.destroyWindow('Frame')
cap.release()