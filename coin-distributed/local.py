import os
import json
import logging
import importlib.util
import numpy as np
from ancillary import list_recursive
from wandbreq import wandb_class
from distributed import GenericLogger,Dataloader,trainer, download_model

log_path = '/output/local.log'
logger = GenericLogger(log_path)
loader =  Dataloader([],[])
get_loader =''
model = ''
criterion = ''
optimizer = ''
wandb = wandb_class('','','','','','')



def local_1(args):

    input = args["input"]
    with open(os.path.join(args["state"]["outputDirectory"]+ os.sep +'input.json'),'w')as fp:
        json.dump(args, fp)
    images, labels = loader.get(int(input['datsetiteration']))    
    loss, gradients =  trainer.train(images, labels)
    wandb.log('Dataset Batch '+str(input['datsetiteration'])+'  Loss',loss)
    logger.log_message(str(args['state']['clientId'])+" Epoch : "+str(input['iteration'])+" Dataset : "+str(input['datsetiteration'])+", Loss : "+str(loss), level=logging.INFO)
    logger.log_message('Project_Urls :'+str(wandb.url.url), level=logging.INFO)
    
    computation_output = {
        "output": {
            'project_url':str(wandb.url.url),
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
        Accuracy = get_loader.validate_model(trainer.model, loader.val) 
        logger.log_message("Validation Accuracy " +str(Accuracy), level=logging.INFO)
        wandb.log('Validation Accuracy',Accuracy)
        logger.log_message("######### Validation Phase Completed #########", level=logging.INFO)
        get_loader.save(trainer.model,args['state']['outputDirectory'],input['iteration'])
    loss, gradients =  trainer.train(images, labels)
    wandb.log('Dataset Batch '+str(input['datsetiteration'])+'  Loss',loss)
    logger.log_message(str(args['state']['clientId'])+" Epoch : "+str(input['iteration'])+", Dataset : "+str(input['datsetiteration'])+", Loss : "+str(loss), level=logging.INFO)
    logger.log_message('Project_Urls :'+str(input['PROJECT_URL']), level=logging.INFO)
    computation_output = {
        "output": {
            'project_url':str(wandb.url.url),
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
    global model, criterion,optimizer, get_loader, trainer
    if not PHASE_KEY:
        logger.log_file_path=PARAM_DICT['state']['outputDirectory']+'/local.log'
        logger.logger = logger._configure_logger()
        download_model(PARAM_DICT['input']['models'],PARAM_DICT['state']['outputDirectory'])
        spec = importlib.util.spec_from_file_location('models', PARAM_DICT['state']['outputDirectory']+'/models.py')
        models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models)
        model = models.response['model']
        criterion = models.response['criterion']
        optimizer = models.response['optimizer']
        get_loader = models.response['dataloader']
        trainer =  trainer(model,criterion, optimizer)
        train, val, test = get_loader.get_loaders(PARAM_DICT['state']['outputDirectory'])
        loader.data =  train
        loader.val = val
        wandb.key = PARAM_DICT['input']['key']
        wandb.dataset =   models.response['dataset']
        wandb.epochs = PARAM_DICT['input']['epochs']
        wandb.lr = models.response['lr']
        wandb.architecture  =   models.response['architecture']
        wandb.project = PARAM_DICT['state']['clientId']
        wandb.env()
        wandb.conf()

        return local_1(PARAM_DICT)
    elif 'remote_1' in PHASE_KEY:
        return local_2(PARAM_DICT)
    else:
        raise ValueError("Error occurred at Local")
