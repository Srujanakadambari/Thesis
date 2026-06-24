# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv, find_dotenv
                     
def load_env():
    _ = load_dotenv(find_dotenv(), override=True)

def get_openai_api_key():
    load_env()
    openai_api = {}
    openai_api["apikey"] = os.getenv("OPENAI_API_KEY")
    openai_api["baseurl"] = os.getenv("OPENAI_BASE_URL")
    openai_api["model"] = os.getenv("MODEL_NAME")
    return openai_api

def get_azure_env_var():
    load_env()
    azure_dict = {}
    azure_dict["api"] = os.getenv("AZURE_API_KEY") #<--------------Das habe leer gelassen
    azure_dict["endpoint"] = os.getenv("OPENAI_BASE_URL") #<--------------Das habe ich geändert
    azure_dict["model"] = os.getenv("MODEL_NAME") #<--------------Das habe ich geändert
    azure_dict["version"] = os.getenv("AZURE_API_VERSION")
    return azure_dict

def get_phoenix_endpoint():
    load_env()
    phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    return phoenix_endpoint


