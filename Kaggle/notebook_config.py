# Aesthetics and utils
import os
from IPython.core.display import display, HTML

# Data maniputalion
from datetime import datetime
import re
import numpy as np
import pandas as pd

# Pandas config
pd.set_option('display.float_format', lambda x: '%.2f' % x) # supress scientific notation
pd.set_option('display.max_columns', 100)                   # increase max columns displayed

# Saving binary files
import pickle

# Data visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import plotly.express as px
import plotly.graph_objects as go

# Machine Learning
import sklearn

# Disable warnings - useless most of the time
import warnings
warnings.filterwarnings('ignore')

# Terminal colors
WHITE = '\033[39m'
CYAN = '\033[36m'
GREEN = '\033[32m'
RED = '\033[31m'

# Markdown styling
# header 1 - yellow 
# <span style='font-family:Lato,sans-serif;color:#fdbf11;font-size: 1em;'>
# header 2 - blue #46abdb
# <span style='font-family:Lato,sans-serif;color:#1696d2;font-size: 1em;'>

# Color pallete for plotting
colors = {
    'cyan': '#1696d2',
    'gray': '#5c5859',
    'black': '#000000',
    'yellow': '#fdbf11',
    'orange': '#ca5800',
    'magenta': '#af1f6b',
    'green': '#408941',
    'red': '#a4201d'
}

# Check Libraries version
print(f'Numpy: {np.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'Sklearn: {sklearn.__version__}')
print(f'Matplotlib: {matplotlib.__version__}')
print(f'Seaborn: {sns.__version__}')
datetime.now().strftime('Last run on %d/%m/%Y at %H:%M:%S')