# Supress warnings
from typing import Tuple
import warnings; warnings.filterwarnings('ignore')

# Image processing
import cv2
import mediapipe as mp

# Image visualizations
import matplotlib.pyplot as plt

# OS functions
import os
import time

# Custom functions
from functions import draw_hand_landmarks, print_record_flag, print_predicted_sign, process_results, make_prediction, FONT_ARGS, FONT_ARGS_2

# Define recording lenght
FRAME_SIZE = 30

# Create Mediapipe Hands model
hands = mp.solutions.hands.Hands(model_complexity=0, 
								min_detection_confidence=0.5, 
								min_tracking_confidence=0.5)
# Settings
#frame_list = []			# store recorded frames
results_list = []		# store hands processing results
recording = False		# flag to capture hand motion
show_results = False	# flag to draw detected sign
show_results_time = 0	# number of frames which results will be displayed

# Start webcam
cam = cv2.VideoCapture(0)
while cam.isOpened():

	# Read frame
	_, image = cam.read()

	# Improve performance = False
	image.flags.writeable = False
    
    # Convert image to RGB
	rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process results with mediapipe Hands
	results = hands.process(rgb_image)

	# Draw hand landmarks (nodes/connections)
	image = draw_hand_landmarks(image, results)

	# Start "recording"
	if recording:
        
        # Append results
		if len(results_list) < FRAME_SIZE:
            # Print Recording flag
			image = print_record_flag(image, rate=len(results_list)/FRAME_SIZE)
			results_list.append(results)
            
		# Stop record and predict sign
		else:
			# Process results with mediapipe Hands
			left_hand_gesture, left_hand_distance ,_ ,_ = process_results(results_list, True)
            # Make predictions with reference db
			predicted_sign = make_prediction(left_hand_gesture, left_hand_distance, True)
            # Empty list and set flags
			results_list.clear()
			show_results = True
			recording = False
			print('Recording stopped.')
            
	# Draw predicted sign in image
	if show_results and not recording:
		image = print_predicted_sign(image, label='Letter predicted: '+ predicted_sign)
		show_results_time += 1
	else:
        # reset counter if record button was pressed
		show_results_time = 0
    # show results time limit
	if show_results_time > 120:
		show_results = False
		show_results_time = 0
    
	# Show image
	cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
	# Key control
	pressed_key = cv2.waitKey(1) & 0xFF
	# Start recording
	if pressed_key == ord('r'):
		recording = True
	# Close webcam
	elif pressed_key == ord('q'):
		recording = False
		break

# Release video capture
cam.release()
# Memory dump
cv2.destroyAllWindows()
# fix window not closing bug on macOS 10.15
cv2.waitKey(1)