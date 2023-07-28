import os
import json
import csv
from ancillary import list_recursive
import numpy as np
import torch
import requests
import nibabel as nib
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def remove_file(file_path):
    try:
        # Use os.remove() to delete the file
        os.remove(file_path)
        print(f"File '{file_path}' has been removed successfully.")
    except Exception as e:
        print(f"Error occurred while removing the file: {e}")

def unzip_file(zip_file_path):
    try:
        # Use the 'unzip' command to extract the contents of the zip file to the same path
        os.system(f'unzip {zip_file_path} -d {os.path.dirname(zip_file_path)}')
        print(f"Successfully extracted files from: {zip_file_path}")
    except Exception as e:
        print(f"Error occurred: {e}")

def download_url_contents(url, folder_path, file_name):
    try:
        # Send a request to the URL to get the contents
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Create the folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)

            # Combine the folder path and file name to get the full file path
            file_path = os.path.join(folder_path, file_name)

            # Save the contents to the file
            with open(file_path, 'wb') as file:
                file.write(response.content)

            print(f"Download successful. File saved at: {file_path}")
        else:
            print(f"Failed to download. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error occurred: {e}")


def local_1(args):

    input = args["input"]
    with open(os.path.join(args["state"]["outputDirectory"]+ os.sep +'input.json'),'w')as fp:
        json.dump(args, fp)
    computation_output = {
        "output": {
            'iteration':input['iteration'],
            'epochs':input['epochs'],
            'value':input['value'],
            "computation_phase": 'local_1'
        }
    }

    return computation_output

def local_2(args):

    input = args["input"]

    computation_output = {
        "output": {
            'iteration':input['iteration'],
            'epochs':input['epochs'],
            'value':input['value'],
            "computation_phase": 'local_1'
        }
    }
    return computation_output


def start(PARAM_DICT):
    PHASE_KEY = list(list_recursive(PARAM_DICT, "computation_phase"))
    if not PHASE_KEY:
        download_url_contents('https://meshnet-pr-dataset.s3.amazonaws.com/data.zip',PARAM_DICT['state']['outputDirectory'],'data.zip')
        unzip_file(os.path.join(PARAM_DICT['state']['outputDirectory'],'data.zip'))
        remove_file(os.path.join(PARAM_DICT['state']['outputDirectory'],'data.zip'))
        PARAM_DICT['input'].update({'epochs':10})
        return local_1(PARAM_DICT)
    elif 'remote_1' in PHASE_KEY:
        return local_2(PARAM_DICT)
    else:
        raise ValueError("Error occurred at Local")
