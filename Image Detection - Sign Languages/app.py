######################################################
#                Libraries and APIs
######################################################

# Supress warnings
import warnings; warnings.filterwarnings('ignore')
# Image processing and visualization
from PIL.Image import open as open_image
# HTTP requests
from urllib.request import urlopen
# Web rendering API
import streamlit as st
# Custom functions
from src.functions   import *
from src.defines     import *
from src.processors  import *
from src.image_db    import *
from src.st_render   import *
from src.css_styles  import *

######################################################
#                 Page Configuration
######################################################

st.set_page_config(
	page_title='Unstructured Data Analysis', 
	page_icon=open_image('img/favicon.png'), 
	layout='wide')

if 'image' not in st.session_state:
    st.session_state['image'] = False
if 'classifiers' not in st.session_state:
	st.session_state['classifiers'] = load_classifiers()
if 'recording_checkbox' not in st.session_state:
	st.session_state['recording_checkbox'] =  False
# st.session_state

######################################################
#                 Sidebar Settings
######################################################

main_page					= 'About'
image_detection_page		= 'Image Detection'
video_detection_page		= 'Video Detection'
live_video_detection_page	= 'Live Video Detection'
language_sign_page			= 'Language Sign prediction'

app_mode = st.sidebar.selectbox(label='Choose the app mode',
			options=[
				main_page,
				image_detection_page,
				video_detection_page,
				live_video_detection_page,
				language_sign_page
			])

######################################################
#                 Main Section
######################################################

# Main title
st.markdown(h1_style('Unstructured Data Analysis') + hr_style(), unsafe_allow_html=True)

# Mode title
st.markdown(h2_style(f'{app_mode}'), unsafe_allow_html=True)

if   app_mode == main_page:
	show_main_page()
elif app_mode == image_detection_page:
	show_image_detection_page()
elif app_mode == video_detection_page:
	show_video_detection_page()
elif app_mode == live_video_detection_page:
	show_live_video_detection_page()
elif app_mode == language_sign_page:
	show_language_sign_page()

