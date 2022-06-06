# Importar os módulos
from typing import Mapping, Tuple
from math import degrees, atan2
from collections import Counter
import re
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp

# Drawing Specs
mp_drawing = mp.solutions.drawing_utils
LANDMARK_DRAWING_SPECS = mp_drawing.DrawingSpec(
        color=(36, 179, 245), 
        thickness=1, 
        circle_radius=4
)
CONNECTIONS_DRAWING_SPECS=mp_drawing.DrawingSpec(
        color=(253, 250, 0), 
        thickness=2, 
        circle_radius=5
)
FONT_ARGS = {
	'fontFace' 		: cv2.FONT_HERSHEY_SIMPLEX,
	'fontScale' : 1,
	'color'		: (253, 250, 0),
	'thickness' : 2,
	'lineType'	: cv2.LINE_AA
}

# Mapping of handpoints
_hand_point_names = [
    ['THUMB_' + i for i in ['CMC', 'MCP', 'IP', 'TIP']],
    ['INDEX_FINGER_' + i for i in ['MCP', 'PIP', 'DIP', 'TIP']],
    ['MIDDLE_FINGER_' + i for i in ['MCP', 'PIP', 'DIP', 'TIP']],
    ['RING_FINGER_' + i for i in ['MCP', 'PIP', 'DIP', 'TIP']],
    ['PINKY_' + i for i in ['MCP', 'PIP', 'DIP', 'TIP']]
]
_hand_point_names = ['WRIST'] + [value for sublist in _hand_point_names for value in sublist]

HAND_POINT_MAPPING = {index: _hand_point_names[index] for index in range(21)}


class Hand_Detector:
    '''
	A class to represent a detect hand in image frame with `mediapipe.Hands`.

	Attributes
	----------
	hand_map : Mapping[str, tuple]
			Coordinates for each node in hand.

	Methods
	-------
	
    '''

    def __init__(self,
                mode: bool = False,
                max_num_hands: int = 2,
                min_detection_confidence: float = 0.5,
                min_tracking_confidence: float = 0.5,) -> None:

        self.mode = mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.hand_map = {}

    def find_hands(self, image: np.ndarray, annot=True) -> np.ndarray:
        '''
        Detec hand and store values in self.results.

        Parameters
        ----------
        image : numpy.ndarray
                image in BGR to be processed
        draw_hand : bool 
                draw annotations on the image

        Returns
        -------
        numpy.ndarray
                processed image with annotations
        '''
        # Convert image from BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Detect hands
        self.results = self.hands.process(rgb_image)
        # Make actions for each hand detected
        if self.results.multi_hand_landmarks:
            for hand_number, hand_landmark in enumerate(self.results.multi_hand_landmarks):
                
                # Map hand coordinates
                self.map_hand_position(image, hand_number)

                # Draw annotations
                if annot:
                    self.mp_draw.draw_landmarks(
                        image,
                        hand_landmark,
                        self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=LANDMARK_DRAWING_SPECS,
                         connection_drawing_spec=CONNECTIONS_DRAWING_SPECS
                        )
        

        return image


    def map_hand_position(self, image: np.ndarray, hand_number: int) -> None:
        '''
        Map hand points coordinates in image from a landmark object and stores 
        in self.hand_map[hand_number].  

        Parameters
        ----------
        image : numpy.ndarray 
                image matrix to extrac shape
        hand_number : int
                hand number to map coordinates

        Returns
        -------
        None
        '''
        # Get image dimensions to calculate coordinates
        height, width, channel = image.shape

        # Create empty list to store results
        landmark_results = []

        # Get coordinates for every node in hand_number
        if self.results.multi_hand_landmarks:
            current_hand = self.results.multi_hand_landmarks[hand_number]
            for _id, landmark in enumerate(current_hand.landmark):
                x_coord, y_coord = int(
                    landmark.x * width), int(landmark.y * height)
                landmark_results.append([_id, x_coord, y_coord])

        # Map nodes
        hand_map = self.create_mapping(landmark_results)
        # Get current hand orientation, e.g. 'Right'
        hand_orientation = self.get_current_hand(self.results.multi_handedness[hand_number])
        # Store mapping results
        self.hand_map[hand_orientation] = hand_map

    def get_connection_angle(self, node_1: str, node_2: str, key: str, verbose=False) -> float:
        '''
        Calculate the angle between two nodes.  

        Parameters
        ----------
        node_1 , node_2 : str  
            keys for accessing node coordinates stored in self.hand_map
        key : str
            name of the hand orientation, 'Right' or 'Left'
        Returns
        -------
        float : Angle between the two vectors
        '''
        # Get coordinates for both nodes
        x1, y1 = self.hand_map[key][node_1]
        x2, y2 = self.hand_map[key][node_2]
        # Calculate delta
        delta_x = x2 - x1
        delta_y = y2 - y1
        # Angle in radians
        rad = atan2(delta_y, delta_x)
        # Convert to degress
        angle = degrees(rad)
        # Print results
        if verbose:
            print(f'Angle of {node_1} to {node_2}: {angle}')

        return angle
    
    def get_hand_map(self):
        try:
            return self.hand_map
        except KeyError:
            print('Hand number {hand_number} not mapped.')
    
    @staticmethod
    def create_mapping(landmark_results:list) -> dict:
        '''
        Map every position with the respective name
        Example: 0 -> 'WRIST'; 1 -> 'THUMB_CMC' etc
        '''
        _hand_coord = [([i[1], i[2]]) for i in landmark_results]
        hand_map = {
            HAND_POINT_MAPPING[index]: coord for index, coord in enumerate(_hand_coord)}
        return hand_map

    @staticmethod
    def get_current_hand(multi_handedness) -> str:
        # Return the orientation of the current hand
        return re.findall('"([^"]*)"', str(multi_handedness))[0]


class Frame_Record(object):
    def __init__(self, reference_signs: pd.DataFrame, frame_length=10):
        # Recording settings
        self.is_recording = False
        self.frame_length = frame_length

        # List of results stored in each frame
        self.recorded_results = []
        self.recorded_frames = []

        # DataFrame storing the distances between the recorded sign & all the reference signs from the dataset
        self.reference_signs = reference_signs

    def record(self, flag:bool) -> None:
        """
        Initialize sign_distances & start recording
        """
        #self.reference_signs["distance"].values[:] = 0
        self.is_recording = flag

    def process_results(self, results, image) -> Tuple[str, bool]:
        """
        If the SignRecorder is in the recording state:
            it stores the landmarks during seq_len frames and then computes the sign distances
        :param results: mediapipe output
        :return: Return the word predicted (blank text if there is no distances)
                & the recording state
        """
        if self.is_recording:
            if len(self.recorded_results) < self.frame_length:
                self.recorded_results.append(results)
                self.recorded_frames.append(image)
            else:
                print('list full')

        # if np.sum(self.reference_signs["distance"].values) == 0:
        #     return "", self.is_recording
        # return self._get_sign_predicted(), self.is_recording

    # def compute_distances(self):
    #     """
    #     Updates the distance column of the reference_signs
    #     and resets recording variables
    #     """
    #     left_hand_list, right_hand_list = [], []
    #     for results in self.recorded_results:
    #         _, left_hand, right_hand = extract_landmarks(results)
    #         left_hand_list.append(left_hand)
    #         right_hand_list.append(right_hand)

    #     # Create a SignModel object with the landmarks gathered during recording
    #     recorded_sign = SignModel(left_hand_list, right_hand_list)

    #     # Compute sign similarity with DTW (ascending order)
    #     self.reference_signs = dtw_distances(recorded_sign, self.reference_signs)

    #     # Reset variables
    #     self.recorded_results = []
    #     self.is_recording = False

    # def _get_sign_predicted(self, batch_size=5, threshold=0.5):
    #     """
    #     Method that outputs the sign that appears the most in the list of closest
    #     reference signs, only if its proportion within the batch is greater than the threshold
    #     :param batch_size: Size of the batch of reference signs that will be compared to the recorded sign
    #     :param threshold: If the proportion of the most represented sign in the batch is greater than threshold,
    #                     we output the sign_name
    #                       If not,
    #                     we output "Sign not found"
    #     :return: The name of the predicted sign
    #     """
    #     # Get the list (of size batch_size) of the most similar reference signs
    #     sign_names = self.reference_signs.iloc[:batch_size]["name"].values

    #     # Count the occurrences of each sign and sort them by descending order
    #     sign_counter = Counter(sign_names).most_common()

    #     predicted_sign, count = sign_counter[0]
    #     if count / batch_size < threshold:
    #         return "Signe inconnu"
    #     return predicted_sign