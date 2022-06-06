import numpy as np
from PIL.Image import open as open_image
from urllib.request import urlopen

# Dictionary with file name and it respective url
file_urls = {
    'lena.png':       'https://i.imgur.com/ncE3dty.png',
    'ednaldo.jpg':    'https://i.imgur.com/JnJD9FB.jpg',
#   'batman.jpg':     'https://i.imgur.com/8jKnbg5.jpg',
#   'calculista.jpg': 'https://i.imgur.com/MQNty8H.jpg',
    'vacilao.jpg':    'https://i.imgur.com/cgp0aY9.jpg',
    'harold.jpg':     'https://i.imgur.com/C8YrIjB.jpg',
    'tool.jpg':       'https://i.imgur.com/bgyZxWt.jpg'
}

# Function to load pictures from a dict of urls
load_image = lambda x : np.array(open_image(urlopen(file_urls[x])))

# Load images
image_db = {file:load_image(file) for file in file_urls.keys()}