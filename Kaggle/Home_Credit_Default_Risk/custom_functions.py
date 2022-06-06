import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Remove outliers from pd Series or Dataframe
def remove_outliers(data):
	Q1 = data.quantile(0.25)
	Q3 = data.quantile(0.75)
	IQR = Q3 - Q1
	if isinstance(data, pd.Series):
		return data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR)))]
	elif isinstance(data, pd.DataFrame):
		return data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Add points on top of a barplot
def add_bar_top_values(plot_data, text_size=8):
	xlocs, xlabs = plt.xticks()
	for i, v in enumerate(plot_data.round(1)):
		plt.text(xlocs[i] - 0.25, v + 0.01, str(v), size=10)
	plt.xticks(xlocs, xlabs, size=text_size)


def upper_triangle(df):
    '''
Returns the upper triangle of a correlation matrix.
You can use scipy.spatial.distance.squareform to recreate matrix from upper triangle.  
Args:
    df: pandas or numpy correlation matrix
Returns:
    list of values from upper triangle
'''
    try:
        assert(type(df)==np.ndarray)
    except:
        if type(df)==pd.DataFrame:
            df = df.values
        else:
            raise TypeError('Must be np.ndarray or pd.DataFrame')
    mask = np.triu_indices(df.shape[0], k=1)
    return df[mask]


def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}






# plot_data = df.select_dtypes(include='object').apply(lambda x : x.astype('category').cat.codes) #.apply(lambda x: x.value_counts())
# # rainbow colors
# c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, plot_data.shape[1])]
# fig = go.Figure(data=[go.Box(
# 					name=column, 
# 					y=plot_data[column],
# 					marker_color=c[i])
# 					for i, column in enumerate(plot_data)])
# fig.update_layout(
# 	height=500,
# 	title='Distribuition of Categorical features', 
# 	title_font_size=18)
# fig.show()

