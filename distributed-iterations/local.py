import os
import json
from ancillary import list_recursive


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
        #PARAM_DICT = {'input':{'iteration':0,'epochs':20,'value':20}}
        return local_1(PARAM_DICT)
    elif 'remote_1' in PHASE_KEY:
        return local_2(PARAM_DICT)
    else:
        raise ValueError("Error occurred at Local")
