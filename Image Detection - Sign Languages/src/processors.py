
from src.functions    import frame_scan, load_classifiers
from streamlit_webrtc import VideoProcessorBase

import av
import cv2
import streamlit as st
import mediapipe as mp

from src.sign_language_functions import draw_hand_landmarks, print_record_flag, print_predicted_sign,\
	process_results, make_prediction, FONT_ARGS, FONT_ARGS_2

WHITE = '\033[39m'
CYAN = '\033[36m'

# Used with OpenCV haarcascade
class VideoProcessor(VideoProcessorBase):

	def __init__(self):
		self.classifier_list = []
		self.active_classifiers = {}
	
	def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

		# update classifiers
		if 'classifiers' not in st.session_state:
			self.default_classifiers = load_classifiers()
		else:
			self.default_classifiers = st.session_state['classifiers']
		
		self.active_classifiers = {i:self.default_classifiers[i] for i in self.classifier_list }

		img = frame.to_ndarray(format='bgr24')

		frame_scan(img, self.active_classifiers)

		img = cv2.flip(img, 1)
		cv2.putText(img=img, 
                text='Potato', 
                org=(50, 50), 
                **FONT_ARGS)

		return av.VideoFrame.from_ndarray(img, format="bgr24")

# Used with Mediapipe Hands
class VideoProcessor2(VideoProcessorBase):

	def __init__(self):
		self.FRAME_SIZE = 20
		self.hands = mp.solutions.hands.Hands(
						model_complexity=0, 
						min_detection_confidence=0.5, 
						min_tracking_confidence=0.5)
		self.results_list = []
		self.recording = False
		self.show_results = False
		self.show_results_time = 0
		self.predicted_sign = '?'

	def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

		image = frame.to_ndarray(format='bgr24')

		# Convert image to RGB
		rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		
		# Process results with mediapipe Hands
		results = self.hands.process(rgb_image)

		# Draw hand landmarks (nodes/connections)
		image = draw_hand_landmarks(image, results)

		# Start "recording"
		if self.recording:
			
			# Append results
			if len(self.results_list) < self.FRAME_SIZE:
				# Print Recording flag
				image = print_record_flag(image, rate=len(self.results_list)/self.FRAME_SIZE)
				self.results_list.append(results)
				
			# Stop record and predict sign
			else:
				# Process results with mediapipe Hands
				left_hand_gesture, left_hand_distance ,_ ,_ = process_results(self.results_list, True)
				# Make predictions with reference db
				self.predicted_sign = make_prediction(left_hand_gesture, left_hand_distance, True)
				# Empty list and set flags
				self.results_list.clear()
				self.show_results = True
				self.recording = False
				print('Recording stopped.')
				
		# Draw predicted sign in image
		if self.show_results and not self.recording:
			image = print_predicted_sign(image, label='Letter predicted: '+ self.predicted_sign)
			self.show_results_time += 1
		else:
			# reset counter if record button was pressed
			self.show_results_time = 0
		# show results time limit
		if self.show_results_time > 120:
			self.show_results = False
			self.show_results_time = 0

		# Flip image for mirroed display
		image = cv2.flip(image, 1)

		# Display frame
		return av.VideoFrame.from_ndarray(image, format="bgr24")

