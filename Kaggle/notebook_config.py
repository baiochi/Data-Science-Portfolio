from IPython.core.display import display
import pickle
import re
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from biokit.viz import corrplot

# terminal colors
WHITE = '\033[39m'
CYAN = '\033[36m'
GREEN = '\033[32m'
RED = '\033[31m'

# color pallete
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

# disable warnings
import warnings
warnings.filterwarnings('ignore')

# pandas config
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# libraries version
print(f'Numpy: {np.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'Sklearn: {sklearn.__version__}')
print(f'Matplotlib: {matplotlib.__version__}')
print(f'Seaborn: {sns.__version__}')


def display_na(df):
    plt.figure(figsize=(10,8))
    sns.displot(
        data=df.loc[:, df.isna().any()].isna().melt(value_name="missing"),
        y="variable",
        hue="missing",
        multiple="fill",
        aspect=3,
        palette='crest'
    )
    plt.title('Bar plot showing Non-Missing Values', weight='bold', fontsize = 18, pad=20, loc='left')
    plt.xlabel(" ")
    plt.ylabel(" ")
    plt.xticks(size=12, weight = 'bold')
    plt.yticks(size=12, weight = 'bold');

    plt.figure(figsize=(18,8))
    sns.heatmap(df.loc[:, df.isna().any()].isna().transpose(),
                cmap="GnBu",
                cbar_kws={'label': 'Missing Data'})
    plt.title('Heatmap showing Missing Values', weight='bold', fontsize=18, pad=20, loc='left')
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.show();



from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
def display_metrics(X_train, X_test, y_train, y_test, estimator):

    print('Train metrics:')
    for metric in [r2_score, mean_absolute_error, mean_squared_error]:
        print(f'{CYAN}{metric.__name__}{WHITE}: {metric(y_train, estimator.predict(X_train)):.2f}')
    print('Test metrics:')
    for metric in [r2_score, mean_absolute_error, mean_squared_error]:
        print(f'{CYAN}{metric.__name__}{WHITE}: {metric(y_test, estimator.predict(X_test)):.2f}')
