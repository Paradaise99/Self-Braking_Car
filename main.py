import cv2

mac_cam = cv2.VideoCapture(0)

while True:
    retrieve, frames = mac_cam.read()
    cv2.imshow('Try2Catch', frames)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
