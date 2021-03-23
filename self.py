#Import all the libraries used in the code
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

#Create a class to define the path of the sond alert
def sound_alert(path):
	#Play the alert sound located on the project
	playsound.playsound(path)

#Create the secound class to define the compute the distance between the 4 points located in the eyes
def eye_aspect_ratio(eye):
	#The coordinates of the vertical eye landmarks X and Y
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	#The coordinates of the horizontal eye landmarks X and Y
	C = dist.euclidean(eye[0], eye[3])

	#Make the calculation of the aspect of the eye, this result will detect if the eye is closed or open 
	ear = (A + B) / (2.0 * C)

	#To complet the class return the claculation of the ear
	return ear

    #The following arguments are nessesary for the code to run, and when running the code one line more need to be done, that is the path to the dlib facial landmark 
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())

# Create 2 arguments that will declare the eye ratio 
#The first one is the blink of the eye 
EYE_AR_THRESH = 0.3

#The second is the number of frames that will need to be with the eyes closed so the alarm can go of 
EYE_AR_CONSEC_FRAMES = 48

#Initialize the counter of the frames 
COUNTER = 0

#Set the alarm to start with false value
ALARM_ON = False

#This part of the code detects the facial landmarks by giving it numbers, like eyes mouth ears and locate the eyes perfectly
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#This is the part of the detection of the eyes regions to complete the last part 
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#This code starts the stream of the video capture in this case by the PC webcam, but externels could be used 
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)


#The while loop is where everything starts, now with all the arguments and classes created the loop of the frames can be created, and the stages could also be created 
while True:
	
	#The creation of the frame argument that will be incrised in size and turned into a grayscale, with that being able to locate the region of the face landmarks 
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#Detect the face area using a invisible rectangle 
	rects = detector(gray, 0)

	#For loop to interect with the graycascade  
	for rect in rects:
		
		#Using the grayscale, predict the face landmarks and putying the landmarks in an array using numpy  
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		#Locate the eye location and compute the EAR for detection of drowsiness 
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		#The avarege of the EAR for both eyes open at the same time 
		ear = (leftEAR + rightEAR) / 2.0

		
		#Compute the convexHull of the both eyes and visualize each of them 
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		#The if statement compares the EAR to the treshhold created before, on the specific frame  
		if ear < EYE_AR_THRESH:
			COUNTER += 1

			#This if statement sees if the eyes are closed for more them the treshhold amount in the amout of frames that where specified
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				#Because the default alarm is of turn it on when needed 
				if not ALARM_ON:
					ALARM_ON = True

					#If the alarm sound is real turn the alarm sound on the background
					if args["alarm"] != "":
						t = Thread(target=sound_alert,
							args=(args["alarm"],))
						t.deamon = True
						t.start()

				#Write on the system that the driver is getting drowsy 
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		#Because it is a while loop is the ear is bigger then the treshhold continue running the frames 
		else:
			COUNTER = 0
			ALARM_ON = False

		#Write on the system the EAR so for testing is easier to understand 
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	#Code to show the frame 
	cv2.imshow("Self-Braking Car by Pedro Incaio", frame)
	key = cv2.waitKey(1) & 0xFF
 
	#Creating the exit key in this case de 'q'
	if key == ord("q"):
		break
#Turn of the system after the 'q' was pressed 
cv2.destroyAllWindows()
vs.stop()