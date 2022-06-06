
######################################################
#             Streamlit Rendering Functions
######################################################

import streamlit as st
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

from src.functions   			 import *
from src.defines     			 import IMAGES_URL
from src.processors  			 import VideoProcessor, VideoProcessor2
from src.sign_language_functions import REFERENCE_SIGNS
from src.css_styles 			 import *

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Turn Server: https://ourcodeworld.com/articles/read/1175/how-to-create-and-configure-your-own-stun-turn-server-with-coturn-in-ubuntu-18-04
# RTC_CONFIGURATION = RTCConfiguration(
#     {
#       "RTCIceServer": [{
#         "urls": ["turn:turn.xxx.dev:5349"],
#         "username": "user",
#         "credential": "password",
#       }]
#     }
# )

######################################################
#                 Instructions Page
######################################################
# Instructions
def show_main_page():

	st.markdown(f'''
	*Codes from {link_style("Let's Code", "https://letscode.com.br/")} Pi-DS course.   
	Select the app mode on the sidebar.*
	''', unsafe_allow_html=True)

	st.markdown(
	'''
	This app aims to explore cases of Unstructured Data analysis.
	Unlike a conventional table or dataframe, these datasets consists on data that doesn't have a predifined structured, 
	like texts, images, videos or audios.  
  
	There is a continuous increase in unstructured data, due to the quickly growing uses of digital applications and services,
	and if analyzed correctly, it can provide quality insights that not even statistical methods alone could explain.  

	Some examples of methods that can be used to exploit unstructured data:  
	- **Audio processing**: process audio data and turns into signals to identify patterns;  
	- **Speech-to-text**: convert audio signals to readable text;  
	- **Pattern recognition algorithms**: identify people, animals, sings or any object of interest of images or videos;  
	- **Natural language Processing**: can be used to understand the meaning in texts, very usefull to analyze bussiness documents or social media posts.  

	'''
	)
	st.markdown(h2_style('OpenCV'), unsafe_allow_html=True)
	st.markdown(
	f'''
	{link_style("Open Source Computer Vision Library", "https://opencv.org/")} offers powerfull image and video processing tools to 
	analyze and create pattern recognition algorithms.
	''', unsafe_allow_html=True)

	st.markdown(h2_style('MediaPipe'), unsafe_allow_html=True)
	st.markdown(
	f'''
	Developed by Google, {link_style("MediaPipe", "https://google.github.io/mediapipe/")} library offers a cross-platform and customizable 
	Machine learning solutions for live and streaming media.  
	Some examples:  
	- Face Detection;  
	- Holistic: full pose recognition (face, body, hands etc);
	- Instant Motion Tracking
	''', unsafe_allow_html=True)

	st.markdown(h2_style('Additional Information'), unsafe_allow_html=True)
	st.markdown(
	f'''
	Source code: {link_style("GitHub", "https://github.com/baiochi/OpenCV-Pattern-Recognition")}  
	Contact: {link_style("e-mail", "mailto:joao.baiochi@outlook.com.br")}  
	''', unsafe_allow_html=True)

######################################################
#                 Image Detection Page
######################################################

# Modifty classifiers params
def adjust_classifier_params():

	# Create sidebar
	with st.sidebar.expander('Classifier Parameters'):

		# Select param
		_c_selected = st.selectbox(label='Select classifier to modify', options=st.session_state['classifiers'].keys())

		if _c_selected:
			# change minNeighbors
			c_minNeighbors = st.number_input(label='minNeighbors', min_value=1, max_value=100, 
											value=int(st.session_state.classifiers[_c_selected]['minNeighbors']))
			# change frame color - value is obtained in rgb -> hex and -> rgb again
			c_color        = st.color_picker('Select color', 
											value=rgb_to_hex(st.session_state.classifiers[_c_selected]['color'])); 
			c_color = hex_to_rgb(c_color)

			# Build classifier structure
			updated_classifier = {
				'minNeighbors': c_minNeighbors,
				'color'       : c_color
			}
			# Update classifer
			st.session_state['classifiers'][_c_selected].update(updated_classifier)
		
		# Reset to default values
		with st.form('reset_classifiers'):
			st.write('Reset params to default values')
			submitted = st.form_submit_button('Submit')
			if submitted:
				st.session_state['classifiers'] = load_classifiers()
				st.success('Values set to default')

def image_classifier_selection(frame):
	
	classifier_list = st.multiselect('Active classifiers', options=st.session_state['classifiers'])
	active_classifiers = {i:st.session_state['classifiers'][i] for i in classifier_list }
	frame_scan(frame, active_classifiers)

def show_image_detection_page():

	# Sidebar Select image
	with st.sidebar.expander(label='Image Options', expanded=False):
		image_opt_choice = st.radio('Select an option to load a image:', options=('Upload file', 'Type image url', 'Select from database'))

		if image_opt_choice == 'Upload file':
			upload_image = st.file_uploader(label='Upload an image.', type='xml')
			if upload_image:
				st.session_state['image'] = load_image_file(upload_image)
		if image_opt_choice == 'Type image url':
			image_url = st.text_input('Insert the url of an image')
			if image_url:
				st.session_state['image'] = load_image_url(image_url)
		if image_opt_choice == 'Select from database':
			sample_image = st.selectbox('Select an image:', options=IMAGES_URL.keys())
			if sample_image:
				st.session_state['image'] = load_image_url(IMAGES_URL[sample_image])
		
	# Sidebar Params configuration
	adjust_classifier_params()

	# MAIN
	if isinstance(st.session_state['image'], np.ndarray):
		# Create a copy of session_state image
		_original_image = st.session_state['image'].copy()
		# Select patterns to detect
		image_classifier_selection(st.session_state['image'])
		# Print image
		st.image(st.session_state['image'], width=600)
		# Reset image without drawings
		st.session_state['image'] = _original_image
	
######################################################
#                 Image Detection Page
######################################################
			
def show_video_detection_page():
	pass

def show_live_video_detection_page():

	# Sidebar Params configuration
	adjust_classifier_params()

	# Video window
	ctx = webrtc_streamer(key='main_video',
						rtc_configuration=RTC_CONFIGURATION,
						video_processor_factory=VideoProcessor)
	# Select classifiers
	if ctx.video_processor:
		ctx.video_processor.classifier_list = st.multiselect('Active classifiers', options=st.session_state['classifiers'])

######################################################
#                 Language Sign Page
######################################################

def show_language_sign_page():

	st.markdown(f'''
	Detailed instructions describe in the following {link_style("notebook", "https://github.com/baiochi/OpenCV-Pattern-Recognition/blob/205f910acefe879efddc69b72b25d18847d5531e/notebook/Sign%20Language%20Detection.ipynb")}
	''', unsafe_allow_html=True)

	# Video window
	ctx = webrtc_streamer(key='main_video',
						rtc_configuration=RTC_CONFIGURATION,
						video_processor_factory=VideoProcessor2)

	if ctx.video_processor:
		with st.sidebar.expander(label='Tracking Options', expanded=True):
			# Recording checkbox
			ctx.video_processor.recording = st.checkbox('Start tracking sign', 
														value=False,
														help='Double click after the first time',
														key='recording_checkbox')
			
			# Show references
			if st.checkbox('Show reference images'):
				st.text('Temporarily unavailable')

			# Show database
			if st.checkbox('Show known letters', help='Current mapped signs in database.'):
				st.text(list(REFERENCE_SIGNS.keys()))
	
	with st.expander("Known bugs"):
		st.markdown(
		"""
		> Current processing for predict sign can take up to 0.05 seconds per frame, wich can raise WebRTC Exception 'is taking too long to execute'. Depending on your current bandwidth this can occur since the first run.  
		
		> Since the database is very small, current predictions can't generalize very well, some signs are hard to reproduce, and letters like 'C' and 'I' have a high false positive rate.
		""")

######################################################
#                 Deprecated
######################################################

# Render form to add new classifiers via file upload
def render_add_new_classifier(classifiers):
	with st.sidebar.expander(label='Add a new classifier', expanded=False):

		# User form to upload file
		with st.form(key='upload_form'):

			# Section to upload file and define parameters
			uploaded_classifier = st.file_uploader(label='Upload a xml file.', type='xml')
			#classifier_url = st.text_input(label='Classifier name')
			classifier_name = st.text_input(label='Classifier name')
			up_minNeighbors = st.number_input(label='minNeighbors', min_value=1, max_value=100, value=5)
			up_color        = st.color_picker('Select color', value='#FF0000'); up_color = hex_to_rgb(up_color)
			up_is_sub       = st.checkbox('Classifier is a sub feature')
			up_sub_class    = st.multiselect(label='Select sub classifiers (if any)', default=None, options=[key for key in classifiers.keys() if classifiers[key]['is_sub']])
			if up_sub_class:
				up_sub_search = True
			else:
				up_sub_search = False

			# Build classifier structure
			new_classifier = {
				#'classifier'  : cv2.CascadeClassifier(uploaded_classifier),
				'minNeighbors': up_minNeighbors,
				'color'       : up_color,
				'sub_search'  : up_sub_search,
				'is_sub'      : up_is_sub,
				'sub_class'   : up_sub_class
			}

			submit_upload = st.form_submit_button('Submit')

			# Store new classifier when submit button is hit.
			if submit_upload:
				if not uploaded_classifier:
					st.error('Please upload a file.')
				elif not classifier_name:
					st.error('Please insert a name for the classifier.')
				else:
					st.session_state.classifiers[classifier_name] = new_classifier
					st.success('File added to classifiers list.')

######################################################
#                  Image Settings
######################################################


# Select active classifiers
# with st.sidebar.expander(label='Select classifiers', expanded=False):

# 	# User form to select classifiers
# 	with st.form(key='select_classifiers_form', clear_on_submit=True):
# 		_active_classifiers = {}
# 		for key_index, classifier_name in enumerate(st.session_state.classifiers):
# 			_current_classifier = st.session_state.classifiers[classifier_name]
# 			_current_checkbox   = st.checkbox(label=f'{classifier_name}', key=key_index)
# 			if _current_checkbox and ( classifier_name not in st.session_state.active_classifiers.keys() ):
# 				_active_classifiers[classifier_name] = _current_classifier

# 		submit_active_classifiers = st.form_submit_button('Apply')

# 		# Store new classifier when submit button is hit.
# 		if submit_active_classifiers:
# 			st.session_state.active_classifiers = _active_classifiers

# st.sidebar.markdown(f"Active classifiers: {list(st.session_state['active_classifiers'].keys())}")

# # Change classifier param
# with st.sidebar.expander(label='Modify params', expanded=False):

# 	_c_selected = st.selectbox(label='Select classifier', options=st.session_state.classifiers.keys())
# 	if _c_selected:
# 		c_minNeighbors = st.number_input(label='minNeighbors', min_value=1, max_value=100, 
# 										value=int(st.session_state.classifiers[_c_selected]['minNeighbors']))
# 		c_color        = st.color_picker('Select color', 
# 										value=rgb_to_hex(st.session_state.classifiers[_c_selected]['color'])); 
# 		c_color = hex_to_rgb(c_color)

# 		# Build classifier structure
# 		updated_classifier = {
# 			'minNeighbors': c_minNeighbors,
# 			'color'       : c_color
# 		}

# 		st.session_state.classifiers[_c_selected].update(updated_classifier)


######################################################
#                  Video Settings
######################################################


# webcam_capture = st.checkbox('Open Webcam')
# cam = cv2.VideoCapture(0)
# window_frame = st.image([], width=200)

# while webcam_capture:
# 	# keep webcam connected
# 	_, frame = cam.read()
# 	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 	# feature scan
# 	#video_scan(frame, st.session_state.classifiers)
# 	# show video
# 	window_frame.image(frame)
# else:
# 	cam.release()

# webrtc_streamer(key="example")
