import os
import json
import numpy as np
from ancillary import list_recursive
import torch


def remote_1(args):

    input = args["input"]
    with open(os.path.join(args["state"]["outputDirectory"]+ os.sep +'input.json'),'w')as fp:
        json.dump(args, fp)
    temp = []
    for site in input:
        temp.append(input[site]["value"])

    # aggregated_gradients = []
    # for grad_list in zip(*temp):
    #         aggregated_grad = torch.stack(grad_list).mean(dim=0)
    #         aggregated_gradients.append(aggregated_grad)
    
    if input[site]['epochs']>input[site]['iteration']:
      computation_output = {
        "output": {
            'iteration':input[site]['iteration']+1,
            'epochs':input[site]['epochs'],
            'value':10,
            "computation_phase": 'remote_1'
            }
            }
    else:
        computation_output = {
        "output": {
            'iteration':input[site]['iteration'],
            'epochs':input[site]['epochs'],
            'value':10,
            "computation_phase": 'remote_2'
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
