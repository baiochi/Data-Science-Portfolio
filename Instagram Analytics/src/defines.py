
# Imports
from datetime import datetime
import requests
import json


# DONE
def make_api_call(api_url: str, endpoint_params: dict, debug: bool=False) -> dict:
	'''
	Make GET request with given url/parameters and store content in json format.
	  
	Parameters
	---
	api_url : str
		base URL of the API  
	endpoint_params : dict 
		names and values of the parameters
	debug : bool, optional
		print debug information 
	
	Returns
	---
	dict: 
		API content in json format.
	'''

	# GET request
	data = requests.get(api_url, endpoint_params)

	# Extract response
	response = json.loads(data.content)

	# Print debug
	if debug:
		print('URL: ' + api_url)
		print('Endpoint Params: ' + json.dumps(endpoint_params, indent=4))
		print(f'Status code: ' + str(data.status_code))

	return response

# TODO: docs
def debug_token(api_url: str, token: str, debug: bool=False) -> dict:
	'''
	Show token metadata parsed in json format and print expiration date.
	'''

	# Create base URL
	api_url += 'debug_token'

	# Create parameters
	endpoint_params = {'input_token': token, 'access_token': token}

	# GET request
	response = make_api_call(api_url, endpoint_params, debug)

	# Print expiration date
	expire_at = datetime.fromtimestamp(response['data']['data_access_expires_at'])
	print(f"Token expires at: {expire_at}")
	
	return response

# TODO: docs
def get_page_ids(api_url: str, token: str, debug: bool=False) -> list:
	'''
	Returns list containing Pages IDs.
	'''
	# Create base URL
	api_url += 'me/accounts'
	
	# Create parameters
	endpoint_params= {'access_token' : token}

	# GET request
	response = make_api_call(api_url, endpoint_params, debug)
	return [page['id'] for page in response['data']]

# TODO: docs
def get_ig_business_id(page_id: str, api_url: str, token: str,  debug: bool=False) -> str:

	# Create base URL
	api_url += page_id

	# Create parameters
	endpoint_params= {
		'fields' : 'instagram_business_account',
		'access_token' : token
	}
	
	# GET request
	response = make_api_call(api_url, endpoint_params, debug)
	return response['instagram_business_account']['id']

# TODO: docs
def get_account_info(ig_id: str, api_url: str, token: str, debug: bool=False) -> dict:

	# Create base URL
	api_url += ig_id

	# Create parameters
	endpoint_params= {
		'fields' : 'username,name,biography,followers_count,follows_count,media_count,profile_picture_url',
		'access_token' : token
	}

	# GET request
	return make_api_call(api_url, endpoint_params, debug)

# TODO: docs
def get_long_live_token(
	api_url: str, app_id: str, app_secret: str, token: str, 
	debug: bool=False) -> dict:
    """ Get long lived access token
    API Endpoint:
        https://graph.facebook.com/
        {graph-api-version}/oauth/access_token?
        grant_type=fb_exchange_token&
        client_id={app-id}&
        client_secret={app-secret}&
        fb_exchange_token={your-access-token}
    """

    api_url += 'oauth/access_token'

    endpoint_params = {
        'grant_type': 'fb_exchange_token',
        'client_id' : app_id,
        'client_secret' : app_secret,
        'fb_exchange_token' : token
    }

    # GET request
    return make_api_call(api_url, endpoint_params, debug)



