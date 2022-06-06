import cv2
import urllib

IMAGES_URL = {
    'lena.png':       'https://i.imgur.com/ncE3dty.png',
    'ednaldo.jpg':    'https://i.imgur.com/JnJD9FB.jpg',
    'batman.jpg':     'https://i.imgur.com/8jKnbg5.jpg',
    'calculista.jpg': 'https://i.imgur.com/MQNty8H.jpg',
    'vacilao.jpg':    'https://i.imgur.com/cgp0aY9.jpg',
    'harold.jpg':     'https://i.imgur.com/C8YrIjB.jpg',
    'tool.jpg':       'https://i.imgur.com/bgyZxWt.jpg'
}

# Load Face classifier
frontal_face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load Eye classifier
eye_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load Smile classifier
smile_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml')

# Load Upper Body classifier
upperbody_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Load Profile Face classifier
profile_face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Load Hand classifier
urllib.request.urlretrieve('https://raw.githubusercontent.com/Aravindlivewire/Opencv/master/haarcascade/aGest.xml', filename='xml/aGest.xml')
hand_classifier = cv2.CascadeClassifier('xml/aGest.xml')

# Create classifiers dictionary
DEFAULT_CLASSIFIERS = {
    'frontal_face':
    {   
        'c_name'      : 'Frontal face',
        'classifier'  : frontal_face_classifier,
        'minNeighbors': 5,
        'color'       : (255,0,0),
        'sub_search'  : True,
        'is_sub'      : False,
        'sub_class'   : ['eye','smile']
    },
    'profile_face':
    {
        'c_name'      : 'Profile face',
        'classifier'  : profile_face_classifier,
        'minNeighbors': 5,
        'color'       : (255,0,255),
        'sub_search'  : True,
        'is_sub'      : False,
        'sub_class'   : ['eye','smile']
    },
    'hand':
    {
        'c_name'      : 'Hand',
        'classifier'  : hand_classifier,
        'minNeighbors': 3,
        'color'       : (255,255,0),
        'sub_search'  : False,
        'is_sub'      : False,
        'sub_class'   : None
    },
        'eye':
    {
        'c_name'      : 'Eye',
        'classifier'  : eye_classifier,
        'minNeighbors': 8,
        'color'       : (0,0,255),
        'sub_search'  : False,
        'is_sub'      : True,
        'sub_class'   : None
    },
    'smile':
    {
        'c_name'      : 'Smile',
        'classifier'  : smile_classifier,
        'minNeighbors': 10,
        'color'       : (0,255,0),
        'sub_search'  : False,
        'is_sub'      : True,
        'sub_class'   : None
    },
}