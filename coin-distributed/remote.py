import os
import json
import numpy as np
from ancillary import list_recursive
import torch

def calculate_average(arrays_list):
    n = len(arrays_list)
    
    if n == 0:
        return None  # Handle the case of an empty list
    
    sum_arrays = [np.zeros_like(arr) for arr in arrays_list[0]]  # Initialize sum list
    
    for arrays in arrays_list:
        for i, arr in enumerate(arrays):
            sum_arrays[i] += arr
    
    average_arrays = [sum_arr / n for sum_arr in sum_arrays]
    
    return average_arrays


def remote_1(args):

    input = args["input"]
    with open(os.path.join(args["state"]["outputDirectory"]+ os.sep +'input.json'),'w')as fp:
        json.dump(args, fp)
    temp = []
    URL = []
    datasets = []
    epochs = []
    for site in input:
        temp.append(input[site]["value"])
        datasets.append(input[site]["datasetsize"])
        epochs.append(input[site]["epochs"])
        URL.append(input[site]["project_url"])
    try:
      gradients = []
      for site in input:
        local_grad = [np.array(array) for array in input[site]["gradients"]]
        gradients.append(local_grad)

      aggregated_grad =  calculate_average(gradients)
      aggregated_grad = [array.tolist() for array in aggregated_grad]
    except Exception as e:
        print(e)
    URL = list(set(URL))
    if input[site]['epochs']>input[site]['iteration']:
      if input[site]['datasetsize']>input[site]['datsetiteration']:
        computation_output = {
        "output": {
            'PROJECT_URL':','.join(URL),
            'iteration':input[site]['iteration'],
            'epochs':min(epochs),
            "datasetsize":min(datasets),
            "datsetiteration":input[site]['datsetiteration']+1,
            'value':(sum(temp) / len(temp))+1,
            "computation_phase": 'remote_1',
            "validate":0,
            "gradients":aggregated_grad,

            }
            }
      else:
        computation_output = {
        "output": {
            'PROJECT_URL':','.join(URL),
            'iteration':input[site]['iteration']+1,
            'epochs':min(epochs),
            "datasetsize":min(datasets),
            "datsetiteration":0,
            'value':(sum(temp) / len(temp))+1,
            "computation_phase": 'remote_1',
            "validate":1,
            "gradients":aggregated_grad
            }
            }
    else:
      computation_output = {
        "output": {
            'PROJECT_URL':','.join(URL),
            'iteration':input[site]['iteration'],
            'epochs':min(epochs),
            "datasetsize":min(datasets),
            "datsetiteration":input[site]['datsetiteration'],
            'value':(sum(temp) / len(temp)),
            "computation_phase": 'remote_2',
            "validate":0,
            "gradients":aggregated_grad
            }, "success": True
            }

    return computation_output


def start(PARAM_DICT):
    for site in PARAM_DICT['input']:
        continue
    #while PARAM_DICT['input'][site]['epochs']>PARAM_DICT['input'][site]['iteration']:
    PHASE_KEY = list(list_recursive(PARAM_DICT, "computation_phase"))
    if "local_1" in PHASE_KEY:
        return remote_1(PARAM_DICT)
    else:
        raise ValueError("Error occurred at Remote")