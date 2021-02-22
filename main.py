import cv2

# Create the variable called mac and video capture using the main camera of the laptop
mac_cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mouth.xml)

#While loop that will run the actual camera capture 
while True:
    # Capture all the frames
    retrieve, frames = mac_cam.read()

    gray = cv2.cvtColor(frames, cv2.COLOR_BGRA2GRAY)

    #detect_eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    detect_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y,w,h) in detect_faces:
        cv2.rectangle(frames, (x,y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('Self-Braking System', frames)

    # To be bale to break the code
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
