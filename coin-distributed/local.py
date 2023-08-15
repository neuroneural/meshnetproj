import os
import json
import logging
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ancillary import list_recursive
from wandbreq import wandb_class
from distributed import GenericLogger,Dataloader,trainer
from cifar10 import get_loaders,CNNModel,validate_model


log_path = '/output/local.log'
logger = GenericLogger(log_path)
loader =  Dataloader([],[])
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.007)
trainer =  trainer(model,criterion, optimizer)
wandb = wandb_class('','','','','')

def local_1(args):

    input = args["input"]
    with open(os.path.join(args["state"]["outputDirectory"]+ os.sep +'input.json'),'w')as fp:
        json.dump(args, fp)
    images, labels = loader.get(int(input['datsetiteration']))    
    loss, gradients =  trainer.train(images, labels)
    wandb.log('Dataset Batch '+str(input['datsetiteration'])+'  Loss',loss)
    logger.log_message(str(args['state']['clientId'])+" Epoch : "+str(input['iteration'])+" Dataset : "+str(input['datsetiteration'])+", Loss : "+str(loss), level=logging.INFO)
    computation_output = {
        "output": {
            'iteration':input['iteration'],
            'epochs':input['epochs'],
            'value':1,#input['value'],
            'gradients': gradients,
            "computation_phase": 'local_1',
            "datasetsize":input['datasetsize'],
            "datsetiteration":input['datsetiteration']
        }
    }

    return computation_output

def local_2(args):

    input = args["input"]
    images, labels = loader.get(int(input['datsetiteration']))
    agg_grad = [np.array(array) for array in input["gradients"]]   
    trainer.optimize(agg_grad) 
    if input['validate']==1:
        logger.log_message("######### Validation Phase #########", level=logging.INFO)
        Accuracy = validate_model(trainer.model, loader.val) 
        logger.log_message("Validation Accuracy " +str(Accuracy), level=logging.INFO)
        wandb.log('Validation Accuracy',Accuracy)
        logger.log_message("######### Validation Phase Completed #########", level=logging.INFO)
    loss, gradients =  trainer.train(images, labels)
    wandb.log('Dataset Batch '+str(input['datsetiteration'])+'  Loss',loss)
    logger.log_message(str(args['state']['clientId'])+" Epoch : "+str(input['iteration'])+", Dataset : "+str(input['datsetiteration'])+", Loss : "+str(loss), level=logging.INFO)
    computation_output = {
        "output": {
            'iteration':input['iteration'],
            'epochs':input['epochs'],
            'value':input['value'],
            'gradients': gradients,
            "computation_phase": 'local_1',
            "datasetsize":input['datasetsize'],
            "datsetiteration":input['datsetiteration']
        }
    }
    return computation_output


def start(PARAM_DICT):
    PHASE_KEY = list(list_recursive(PARAM_DICT, "computation_phase"))
    if not PHASE_KEY:
        logger.log_file_path=PARAM_DICT['state']['outputDirectory']+'/local.log'
        logger.logger = logger._configure_logger()
        train, val, test = get_loaders(PARAM_DICT['state']['outputDirectory'])
        loader.data =  train
        loader.val = val
        PARAM_DICT['input'].update({'epochs':5,'datasetsize':len(train)})#len(train)})
        wandb.key = PARAM_DICT['input']['key']
        wandb.dataset =   PARAM_DICT['input']['dataset']
        wandb.epochs = PARAM_DICT['input']['epochs']
        wandb.architecture  =   PARAM_DICT['input']['architecture']
        wandb.project = PARAM_DICT['state']['clientId']
        wandb.env()
        wandb.conf()

        return local_1(PARAM_DICT)
    elif 'remote_1' in PHASE_KEY:
        return local_2(PARAM_DICT)
    else:
        raise ValueError("Error occurred at Local")
