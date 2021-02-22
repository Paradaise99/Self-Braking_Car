import cv2

# Create the variable called mac and video capture using the main camera of the laptop
mac_cam = cv2.VideoCapture(0)

#While loop that will run the actual camera capture 
while True:
    # Capture all the frames
    retrieve, frames = mac_cam.read()
    cv2.imshow('Self-Braking System', frames)

    # To be bale to break the code
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
