import cv2

# Create the variable called mac and video capture using the main camera of the laptop, also create the face and eye cascade cariable that will locate the xml file in the opencv library
mac_cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mouth.xml)

#While loop that will run the actual camera capture 
while True:
    # Capture all the frames
    retrieve, frames = mac_cam.read()

    # Transform the webcam is to a grayscale
    gray = cv2.cvtColor(frames, cv2.COLOR_BGRA2GRAY)

    # Created the variable that will detect the faces in the grayscale 
    detect_faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # For loop created to run the varaibles until the user dosent want it to work
    for (x, y,w,h) in detect_faces:

        # Create an rectangle around the face of the user
        cv2.rectangle(frames, (x,y), (x+w, y+h), (255, 0, 0), 2)

        # Creation of variable that locate the rectangles in the face 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frames[y:y+h, x:x+w]
        
        # Run the location of the eyes on the face to create a rectangle around it
        detect_eyes = eye_cascade.detectMultiScale(roi_gray)

        # The for loop for the eyes
        for (ex, ey, ew, eh) in detect_eyes:

            #Create the rectangle around the eyes
            cv2.rectangle(roi_color,(ex, ey), (ex+ew,ey+eh), (0, 255, 0), 2)

    # Open the webcam and give it the name Self-Braking System
    cv2.imshow('Self-Braking System', frames)

    # To be bale to break the code
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
