
def hr_style():
	style = '''
	margin-top: 0px;
	background: linear-gradient(to right, #00fafd, #f5b324);
	height: 2px;
	'''
	return f'<hr style="{style}">'

def h1_style(text):
	style = '''
	color:#f5b324;
	-webkit-font-smoothing: antialiased;
	'''
	return f'<h1 style="{style}">' + text + '</h1>'

def h2_style(text):
	style = '''
	color:#f5b324;
	-webkit-font-smoothing: antialiased;
	'''
	return f'<h2 style="{style}">' + text + '</h2>'

def link_style(text, url):
	style = '''
	color:#00fafd;
	-webkit-font-smoothing: antialiased;
	'''
	return f'<a href="{url}" style="{style}">' + text + '</a>'