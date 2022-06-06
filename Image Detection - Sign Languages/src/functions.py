
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL.Image import open as open_image
from urllib.request import urlopen
from src.defines import IMAGES_URL, DEFAULT_CLASSIFIERS

# Load default classifiers
load_classifiers = lambda : DEFAULT_CLASSIFIERS.copy()
# Load image from url
load_image_url = lambda url : np.array(open_image(urlopen(url)))
# Load image from file
load_image_file = lambda file : np.array(open_image(file))
# Convert HEX to RBG
hex_to_rgb = lambda hex : tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
# Convert RGB to HEX
rgb_to_hex = lambda rgb_tuple : '#%02x%02x%02x' % rgb_tuple

# Dected feature for a given classifier
def feature_detection(rgb_image, gray_image, classifier, 
                      scaleFactor=1.3, minNeighbors=5, sub_search=False,
                      color=(255,0,0), c_name='', font=cv2.FONT_HERSHEY_SIMPLEX, **kwargs):
    
    # Create empty tuples for roi
    roi_rgb, roi_gray = (), ()
    
    # Select region of interest
    classifier_results = classifier.detectMultiScale(gray_image, scaleFactor, int(minNeighbors))
        
    # Draw rectangle for current feature
    for (cx, cy, cw, ch) in classifier_results:
        
        # Apply drawing in image
        rgb_image = cv2.rectangle(img     =rgb_image,        # image to draw
                                  pt1       =(cx, cy),       # top-left corner
                                  pt2       =(cx+cw, cy+ch), # bottom-right corner
                                  color     =color,          # color of the rectangle (red)
                                  thickness =2)              # thickness of the rectangle
        # Annotate
        text = str(c_name) + ' [' + str(int(minNeighbors)) + str(']')
        cv2.putText(img       = rgb_image, 
                    text      = text, 
                    org       = (cx+cw-(len(text)*5), cy-5),     # place in top-right corner
                    fontFace  = font, 
                    fontScale = 0.3, 
                    color     = (255,255,255), 
                    thickness = 1,
                    lineType  = cv2.LINE_AA
                )
    
        # Select the region of interest, both for gray and rgb scale
        if sub_search:
            roi_rgb  += ( rgb_image[cy:cy+ch, cx:cx+cw],)
            roi_gray += (gray_image[cy:cy+ch, cx:cx+cw],) 

    # Return roi values
    return roi_rgb, roi_gray

# Search for features given a dict of classifiers
def frame_scan(frame, classifiers):

    # Create gray scale image
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    for classifier in classifiers.values():

        # Detect features (global)
        if not classifier['is_sub']:
            c_roi_rgb, c_roi_gray = feature_detection(frame, gray_frame, **classifier)

        # Detect if current classifier has sub_features to analyze
        if classifier['sub_search'] and any(i in classifier['sub_class'] for i in classifiers.keys()):
            # Get only sub features that have been selected in menu
            sub_class = {key:value for key, value in classifiers.items() if value['is_sub']}
            # Get region of interest
            for roi_rgb, roi_gray in zip(c_roi_rgb, c_roi_gray):
                # Make detection for each sub feature available
                for sub_classifier in sub_class:
                    feature_detection(roi_rgb, roi_gray, **classifiers[sub_classifier])


# Print FPS on frame
def print_fps(video, frame, font=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255)):
    
    # Get FPS
    fps = int(video.get(5))
    
    height, width, channel = frame.shape
    
    offset = int(height / len(str(fps))) - 10

    cv2.putText(img=frame, 
                text=str(fps), 
                org=(50, offset), 
                fontFace=font, 
                fontScale=1, 
                color=color, 
                thickness=5,
                lineType=8)


