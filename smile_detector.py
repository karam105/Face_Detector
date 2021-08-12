import cv2

# Face classifier
face_detector = cv2.CascadeClassifier('D:/haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('D:/haarcascade_smile.xml')

# Grab webcam feed
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
	# Read the current frame from the webcam video stream
	successful_frame_read, frame = webcam.read()

	# If there's an error, abort
	if not successful_frame_read:
		break

	# Change to grayscale to improve performance
	frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces first
	faces = face_detector.detectMultiScale(frame_grayscale)
	

	# Run face detection within each of those faces
	for (x, y, w, h) in faces:
		# Draw rectangle around the face
		cv2.rectangle(frame, (x,y), (x+w, y+h), (100,200,50), 4)

		# Get the sub frame (using numpy N-dimensional array slicing)
		the_face = frame[y:y+h, x:x+w]

		# Change to grayscale to improve performance
		face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
		
		smiles = smile_detector.detectMultiScale(face_grayscale, 1.7, 20)

		# # Run smile detection within each of those faces
		# for (x_, y_, w_, h_) in smiles:
		# 	# Draw rectangle around the smile
		# 	cv2.rectangle(the_face, (x_,y_), (x_+w_, y_+h_), (50,50,200), 4)

		# Label this face as smiling
		if len(smiles) > 0:
			cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))
	
	# Show the current frame
	cv2.imshow('Smile Detector', frame)

	# Display
	# waitkey is waiting for a key press but with the '1', it'll wait that much time for a key before moving on to the next frame
	# the input is in milliseconds so it's waiting 1 ms before going to the next frame
	cv2.waitKey(1)

# Cleanup
webcam.release()
cv2.destroyAllWindows()

print("Complete")