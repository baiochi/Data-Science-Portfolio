# Supress warnings
import warnings; warnings.filterwarnings('ignore')

# Docstrings
from typing import List, Tuple

# Data and math operations
import re
import numpy as np
import pandas as pd

# Compute DTW distance
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# Image processing
import cv2
import mediapipe as mp

# Image visualizations
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import plotly.express as px
import plotly.graph_objects as go

# Read mediapipe labels
from google.protobuf.json_format import MessageToDict

# OS functions
import os
import time

# Load binary
import pickle

# Progress bar
from tqdm.auto import tqdm

# Mediapipe instances
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# Image drawing colors
BLUE_HEX   = '#00fafd'    
YELLOW_HEX = '#f5b324'   
BLUE_BGR   = (253,250,0)
YELLOW_BGR = (36,179,245)

# Landmark styles
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS
# DEFAULT_LANDMARK_POINTS = mp_drawing_styles.get_default_hand_landmarks_style()
# DEFAULT_LANDMARK_CONNECTIONS = mp_drawing_styles.get_default_hand_connections_style()
LANDMARK_DRAWING_SPECS = mp_drawing.DrawingSpec(
        color=YELLOW_BGR, 
        thickness=1, 
        circle_radius=4
)
CONNECTIONS_DRAWING_SPECS=mp_drawing.DrawingSpec(
        color=BLUE_BGR, 
        thickness=2, 
        circle_radius=5
)
FONT_ARGS = {
    'fontFace'  : cv2.FONT_HERSHEY_SIMPLEX,
    'fontScale' : 2,
    'color'     : BLUE_BGR,
    'thickness' : 2,
    'lineType'  : cv2.LINE_AA
}
FONT_ARGS_2 = {
    'fontFace'  : cv2.FONT_HERSHEY_SIMPLEX,
    'fontScale' : 2,
    'color'     : YELLOW_BGR,
    'thickness' : 2,
    'lineType'  : cv2.LINE_AA
}

# Terminal ASCII colors
WHITE = '\033[39m'
CYAN  = '\033[36m'


# Load binary containing data from reference signs
with open('../database/reference_signs_2.pickle', 'rb') as file:
    REFERENCE_SIGNS = pickle.load(file)

# Draw annotations for detected hand
def draw_hand_landmarks(frame:np.ndarray, results:any) -> np.ndarray:
    annot_image = frame.copy()

    if results.multi_hand_landmarks:
        for hand_number, hand_landmark in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                        annot_image,
                        hand_landmark,
                        HAND_CONNECTIONS,
                        LANDMARK_DRAWING_SPECS,
                        CONNECTIONS_DRAWING_SPECS)
    return annot_image

def print_predicted_sign(image:np.ndarray, label:str) -> np.ndarray:
    # Get image shape
    width, height, _ = image.shape
    # Flip image
    image = cv2.flip(image, 1)
    # Get space to draw background rectangle in text
    (tw, th), _ = cv2.getTextSize(label, FONT_ARGS['fontFace'], FONT_ARGS['fontScale'], FONT_ARGS['thickness'])
    # Configure coordinates
    x1, y1 = 25, width-80
    offset = 20
    # Draw background rectangle
    image = cv2.rectangle(image, 
                          (x1-offset,    y1-th-offset), 
                          (x1+tw+offset, y1+offset), 
                          (0,0,0), -1)
    # Draw text
    image = cv2.putText(image, label, (x1, y1), **FONT_ARGS)
    # Return to original position
    image = cv2.flip(image, 1)

    return image

def print_record_flag(image:np.ndarray, rate:float) -> np.ndarray:
    '''
    rate (float): current frame length proportion, e.g. len(results_list)/FRAME_SIZE
    '''
    # Get image shape
    width, height, _ = image.shape
    # Flip image
    image = cv2.flip(image, 1)
    # Coordinates configuration
    width_level  = int(width*0.05)         # offset 5% of image width
    heigth_level = int(height*0.05)        # offset 5% of image heigth
    bar_heigth_level = int(height*0.07)    # offset 7% of image heigth
    # Draw recording text
    image = cv2.putText(image, 'Recording...', (width_level, heigth_level), **FONT_ARGS_2)
    # Draw progress bar
    full_bar_width = int(width*0.55)
    current_progress_bar = int(full_bar_width * rate)
    # Draw empty bar
    image = cv2.line(image,            
                     (width_level,    bar_heigth_level), 
                     (full_bar_width, bar_heigth_level), 
                     (0,0,0), 7)
    # Draw filling bar
    image = cv2.line(image,            
                     (width_level,     bar_heigth_level), 
                     (current_progress_bar, bar_heigth_level), 
                     YELLOW_BGR, 6)
    image = cv2.flip(image, 1)
    
    return image

'''
process_results
|
|--get_hand_coordinates
|
|--create_hand_gesture
|   |
|   |--calculate_finger_distances
|   |--get_landmark_angles
|      |
|      |--calculate_3D_angle
|--> hand_gesture_arrays
'''

# Create coordinates dataframes for each landmarks results
def get_hand_coordinates(results_list:list, verbose=False) -> Tuple[list,list]:
    # Store mapping for each hand
    left_hand_list = []
    right_hand_list = []

    # Iterate over results
    for frame_number, results in enumerate(results_list):
        # Check if any hand was detected
        if results.multi_hand_landmarks:

            # Iterate over hands (right or left)
            for hand_number, hand_landmark in enumerate(results.multi_hand_landmarks):

                # Extract hand orientation str
                hand_label = MessageToDict(results.multi_handedness[hand_number])
                hand_label = hand_label['classification'][0]['label']

                # Create dataframe to store node coordinates
                hand_map_df = pd.DataFrame()

                # Iterate over landmarks of current hand
                for node_id, landmark in enumerate(hand_landmark.landmark):
                    # Get coordinates and labels
                    _row = pd.DataFrame(
                            data={
                                'x'          : landmark.x,
                                'y'          : landmark.y,
                                'z'          : landmark.z,
                            }, index=[node_id])
                    # Append row to dataframe
                    hand_map_df = pd.concat([hand_map_df, _row])

                # Add mapping for into the corresponding hand
                if hand_label == 'Left':
                    left_hand_list.append(hand_map_df)
                elif hand_label == 'Right':
                    right_hand_list.append(hand_map_df)
        #print(f'Frame {frame_number}, Left={len(left_hand_list)}, Right={len(right_hand_list)}')
    if verbose:
        print(f'Frames detected:\nLeft hand:{len(left_hand_list)}, Right hand:{len(right_hand_list)}')
    
    return left_hand_list, right_hand_list

def calculate_finger_distances(landmark:pd.DataFrame) -> List[float]:
    '''
    Calculate the distance between Wrist node and all fingertips.  
    frame_list: pd.Dataframe containing landmark coordinates  
    Return: list of floats
    '''
    finger_distances = []
    
    for index in [4,8,12,16,20]:
        # Wrist node
        p1 = landmark.iloc[0,:].values
        # Fingertip node
        p2 = landmark.iloc[index,:].values
        # Calculate distance
        squared_dist = np.sum((p1-p2)**2, axis=0)
        dist = np.sqrt(squared_dist)
        finger_distances.append(dist)
    
    return finger_distances

def calculate_3D_angle(u:np.ndarray, v:np.ndarray) -> float:
    '''
    Calculate the angle between 2 points with (x,y,z) coordinates
    ang = acos( (x1*x2 + y1*y2 + z1*z2) / sqrt( (x1*x1 + y1*y1 + z1*z1)*(x2*x2+y2*y2+z2*z2) ) )
    Return: angle in radians
    '''
    # Calculate cross product
    dot_product = np.dot(u, v)
    # Calculate vector norm
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    # Calculate angle in radians
    angle = np.arccos(dot_product / norm)
    if np.isnan(angle) or np.isinf(angle):
        angle=0
    return angle

# TODO: verify alternative way to compute only more relevant angles
def get_landmark_angles(landmark_df:pd.DataFrame) -> np.ndarray:
    '''
    Obtain all angles ffrom a hand landmarks.
    Dataframe format:
    x	y	z
    0.183309	0.889030	9.577590e-09
    Return: list of shape 441
    '''
    landmark_angles = []
    # Multiply every node with each other, 21 connections * 21 = 441 angles
    for i in range(landmark_df.shape[0]):
        for j in range(landmark_df.shape[0]):
            # Calculate angle between X and Y coordinates
            _node_1 = landmark_df.iloc[i,[1,2]]
            _node_2 = landmark_df.iloc[j,[1,2]]
            landmark_angles.append(calculate_3D_angle(_node_1, _node_2))
    return landmark_angles

def create_hand_gesture(frame_list:list, connections=441, hand_label='') -> np.ndarray:
    '''
    frame_list: list of pd.Dataframe containing each frame landmark coordinates
    connections(int): number of connected nodes
    Return: array of shape(frame_number, connections) to compute distance
    '''
    frame_size = len(frame_list)
    
    # Create empty dict/arrays
    distance_array = np.zeros([frame_size, 5])
    gesture_array = np.zeros([frame_size, connections])
    
    for frame_index, landmark in tqdm(enumerate(frame_list), 
                                      desc=f'Calculating {hand_label} landmark angles/distances',
                                      total=frame_size,
                                      colour='#00fafd'):
        # Compute finger distances
        distance_array[frame_index] = calculate_finger_distances(landmark)
        # Compute angles for each landmark
        gesture_array[frame_index] = get_landmark_angles(landmark)
        
    return gesture_array, distance_array

# Process results
def process_results(landmarks_results, verbose=False):
    left_hand_gesture   = None
    left_hand_distance  = None
    right_hand_gesture  = None
    right_hand_distance = None
    # Process results
    left_hand_list, right_hand_list = get_hand_coordinates(landmarks_results, verbose)
    # Create gesture arrays
    if len(left_hand_list)>0:
        left_hand_gesture, left_hand_distance  = create_hand_gesture(left_hand_list, hand_label='left')
    if len(right_hand_list)>0:
        right_hand_gesture, right_hand_distance  = create_hand_gesture(right_hand_list, hand_label='right')
    
    return left_hand_gesture, left_hand_distance, right_hand_gesture, right_hand_distance


def find_best_candidate(angle_predictions:dict, finger_predictions:dict, threshold=3) -> str:
    
    try:
        angle_rank = sorted(angle_predictions, key=angle_predictions.get)[:threshold]
        finger_rank = sorted(finger_predictions, key=finger_predictions.get)[:threshold]
    except AttributeError:
        print('Hand landmarks not detected. Try switch hands.')
        return '?'
    
    print(sorted(angle_predictions, key=angle_predictions.get))
    print(sorted(finger_predictions, key=finger_predictions.get))
        
    for i in range(threshold):
        for j in range(threshold):
            if angle_rank[i] in finger_rank:
                return angle_rank[i]
    return '?'

def calculate_dtw(landmark_array:np.ndarray, measure:str) -> dict:
    '''
    measure (str): 'angles' or 'distance'
    '''
    if not isinstance(landmark_array, np.ndarray):
        return None
    
    predictions = {}
    for letter, sign in REFERENCE_SIGNS.items():
        dist, _ = fastdtw(landmark_array, 
                          sign[measure], 
                          dist=euclidean)
        #print(f'Distance for sign {letter}: {dist:.2f}')
        predictions[letter] = dist
        
    return predictions

# Compare recorded sign with references
# TODO: modify function to load all sign
def make_prediction(angle_array:np.ndarray, finger_array:np.ndarray, verbose=False) -> str:
    
    angle_predictions = calculate_dtw(angle_array, 'angles')
    finger_predictions = calculate_dtw(finger_array, 'distance')
    predicted_sign = find_best_candidate(angle_predictions, finger_predictions)
    if verbose:
        print(f'{CYAN}Predicted sign {predicted_sign}{WHITE}')

    return predicted_sign

def get_video_frames(file_name):
    
    frame_list = []
    cap = cv2.VideoCapture(file_name)
    pbar = tqdm(desc='Reading frames', total=cap.get(cv2.CAP_PROP_FRAME_COUNT), colour='#f5b324')
    while cap.isOpened():
        # Read frame
        ret, frame = cap.read()
        if not ret:
            pbar.close()
            break
        # Store frame
        frame_list.append(frame)
        pbar.update(1)
    cap.release()
    
    return frame_list

def generate_reference_database(files:list, path:str, model) -> dict:
    
    # Store gesture arrays
    reference_signs = {}
        
    # Loop through the video list
    for file_name in files:
        
        # Extra sign name
        sign = file_name[12:-4]

        # Read frames
        frame_list = get_video_frames(path + '/' + file_name)

        # Process frames with mediapipe Hands
        results_list = []
        for frame in tqdm(frame_list, 
                            desc=f'Learning sign {sign}', 
                            total=len(frame_list),
                            colour='#00fafd'):
            results_list.append(model.process(frame))

        # Create and store gesture array
        reference_signs[sign] = {}
        reference_signs[sign]['angles'], reference_signs[sign]['distance'], _, _ = process_results(results_list)

        # Dump memory
        frame_list.clear(); results_list.clear();

    return reference_signs