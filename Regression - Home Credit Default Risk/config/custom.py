
# Aesthetics and utils
import os
from IPython.core.display import display, HTML

# Data maniputalion
import re
import numpy as np
import pandas as pd

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

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,\
	ConfusionMatrixDisplay, classification_report, roc_auc_score, f1_score, roc_curve, plot_roc_curve, precision_recall_curve


from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import MultinomialNB, CategoricalNB

from hyperopt import Trials, hp, tpe, fmin, space_eval
from xgboost import XGBClassifier

# Model interpretation
from lime import lime_tabular

################### CUSTOM CONFIGURATIONS ###################

# Pandas options
# supress scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# show all columns
pd.set_option("display.max_columns", 100)
# show more rows
#pd.set_option("display.max_rows", 100)

# Colors for Cell output
WHITE = '\033[39m'
CYAN = '\033[36m'
GREEN = '\033[32m'
RED = '\033[31m'

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

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

# Check Libraries version
print(f'Numpy: {np.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'Sklearn: {sklearn.__version__}')
print(f'Matplotlib: {matplotlib.__version__}')
print(f'Seaborn: {sns.__version__}')


################### CUSTOM FUNCTIONS ###################

# Create hyper-parameter space
def create_hps_space(hps_params:dict, verbose=False):
	hps_space = {}
	for keys, value in hps_params.items():
		hps_space[keys] = hp.choice(keys, value)
	if verbose:
		print(f'hps_space: {hps_space}')
	return hps_space

# Create fmin function used in hyperopt
def create_fmin_function(estimator:any, pp_pipeline:Pipeline, 
                        X_train:np.array, y_train:np.array, 
                        metric:str, hps_space:dict, random_state=42, verbose=False) -> dict: 

	# create objective function
	def objective_function(hps_space) -> float:

		pipeline = Pipeline([('pp', pp_pipeline),
							('est', estimator(**hps_space, random_state=random_state))])

		splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

		results = cross_validate(pipeline, 
								X_train, y_train, 
								cv=splitter, 
								scoring=metric, 
								n_jobs=-1) 

		return -results['test_score'].mean()

	return objective_function


def display_metrics(X_train, X_test, y_train, y_test, estimator, metric_type):

	if metric_type=='regression':
		print(f'{CYAN}Train metrics:{WHITE}')
		for metric in [r2_score, mean_absolute_error, mean_squared_error]:
			print(f'{metric.__name__}: {metric(y_train, estimator.predict(X_train)):.3f}')
		print(f'{CYAN}Test metrics:{WHITE}')
		for metric in [r2_score, mean_absolute_error, mean_squared_error]:
			print(f'{metric.__name__}: {metric(y_test, estimator.predict(X_test)):.3f}')

	elif metric_type=='classification':
		print(f'{CYAN}Train metrics:{WHITE}')
		print(f'{roc_auc_score.__name__} {roc_auc_score(y_train, estimator.predict_proba(X_train)[:,1]):.3f}')
		print(f'{f1_score.__name__} {f1_score(y_train, estimator.predict(X_train)):.3f}')
		print(f'{CYAN}Test metrics:{WHITE}')
		print(f'{roc_auc_score.__name__} {roc_auc_score(y_test, estimator.predict_proba(X_test)[:,1]):.3f}')
		print(f'{f1_score.__name__} {f1_score(y_test, estimator.predict(X_test)):.3f}')
	
	elif metric_type=='clustering':
		print('To be implemented.')
	
	else:
		raise ValueError('Invalid metric type')
	
	
def select_df_split_size(split_size):

	split_size /= 100
	print()

	if split_size > 1:
		raise ValueError('Wrong input.')
	elif split_size < 0.05:
		raise ValueError('Percentage is too small.')
	else:
		train_size = 0.8 * split_size
		test_size = 0.2 * split_size
		return train_size, test_size


def display_classification_metrics(estimator, X_train, X_test, y_train, y_test, 
									plot_roc_auc=True, plot_confusion_matrix=True, plot_precision_recall=True):

	# predict for train/test
	y_pred_train = estimator.predict(X_train)
	y_pred_test = estimator.predict(X_test)
	# predict proba for train/test
	y_proba_train = estimator.predict_proba(X_train)
	y_proba_test = estimator.predict_proba(X_test)

	# Calculate ROC AUC Score
	print(f'{CYAN}ROC AUC Score for Train dataset:{WHITE} {roc_auc_score(y_train, y_proba_train[:,1])}')
	print(f'{CYAN}ROC AUC Score for Test dataset:{WHITE} {roc_auc_score(y_test, y_proba_test[:,1])}')
	print()
	# Classification report: precision, recall, f1-score
	print(f'{CYAN}Train data Classification Report:{WHITE}\n',classification_report(y_train, y_pred_train))
	print(f'{CYAN}Test data Classification Report:{WHITE}\n',classification_report(y_test, y_pred_test))

	if plot_roc_auc:
		# get false positive rates and true positive rates
		fpr, tpr, _ = roc_curve(y_test, y_proba_test[:,1])
		# score for test dataset
		score = round(roc_auc_score(y_test, y_proba_test[:,1]),3)
		# plot ROC AUC curve
		plt.figure(figsize=(8,6))
		plt.plot(fpr, tpr, label=f'Classifier AUC = {score}')
		plt.plot(np.linspace(0,1,100),np.linspace(0,1,100), ls=':', label='Random = 0.5')
		plt.title('ROC AUC curve', fontsize=18, pad=20, loc='left')
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.legend()
		plt.tight_layout()
		plt.show()

	if plot_precision_recall:
		# get precisions, recalls and thres
		precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_test[:,1])
		# plot Precision-Recall curve
		plt.figure(figsize=(8,6))
		plt.plot(thresholds, precisions[:-1], label='Precision')
		plt.plot(thresholds, recalls[:-1], label='Recall', color='black')
		plt.title('Precision-Recall curve', fontsize=18, pad=20, loc='left')
		plt.xlabel('Tresholds cutoffs')
		plt.legend()
		plt.tight_layout()
		plt.show()
		
	if plot_confusion_matrix:
		fig, ax = plt.subplots(1,2,figsize=(10,6))
		# plot Confusion Matrix from Train dataset
		fig.suptitle('Confusion Matrix', size=22)
		ConfusionMatrixDisplay.from_predictions(y_train, y_pred_train, ax=ax[0], colorbar=False)
		ax[0].grid(False)
		ax[0].set_title('Train dataset', size=18, pad=20)

		# plot Confusion Matrix from Test dataset
		cf = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, ax=ax[1], colorbar=False)
		ax[1].grid(False)
		ax[1].set_title('Test dataset', size=18, pad=20)
		
		# coordinates to plot colorbar on the right plot
		cax = plt.axes([0.95, 0.21, 0.03, 0.60])
		plt.colorbar(cf.im_, cax=cax)
		plt.show()

"""
ct = make_column_transformer(
                (SimpleImputer(strategy='mean'),        numeric_features),
                (SimpleImputer(strategy='constant', 
                                fill_value='unknow'),   categorical_features),
                (MinMaxScaler(),                        numeric_features),
                (OneHotEncoder(handle_unknown='ignore'),categorical_features),
                remainder='passthrough')
pipeline = make_pipeline(ct, RandomForestClassifier(random_state=42)).fit(X_train, y_test)
"""