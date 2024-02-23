import sys
from ancillary import list_recursive
from dist import GenericLogger, training
from wandbreq import wandb_class

class distributed():
    def initial_send_gradients(args):
        gradients = trainer.get_gradients()
        computation_output = {
            "output": {
                "gradients": gradients,     
                "computation_phase": 'local',
                "iterations": iterations,
                "current_iteration":current_iteration,
                "URL":str(trainer.wandb.url.url)
            }
        }
        return computation_output
    
    def next_send_gradients(args):
        if args["input"]["last_iteration"]==False:
            trainer.agg_gradients(args["input"]["gradients"])
            gradients = trainer.get_gradients()
            computation_output = {
                "output": {
                    "gradients":gradients,       
                    "computation_phase": 'local',
                    "iterations": iterations,
                    "current_iteration":current_iteration,
                    "URL":str(trainer.wandb.url.url)
                }
            }
        else:
            wandb.end()
            computation_output = {"output":{"computation_phase":"end","URL":str(trainer.wandb.url.url)}}
        return computation_output


def start(PARAM_DICT):
    PHASE_KEY = list(list_recursive(PARAM_DICT, "computation_phase"))
    if not PHASE_KEY:
        global iterations, current_iteration, Dice, model, trainer, train_size, wandb
        sys.path.append(PARAM_DICT['state']['baseDirectory'])
        import meshnet, dice, loader
        current_iteration = 0
        wandb = wandb_class(key=PARAM_DICT['input']['wan'],lr=PARAM_DICT['input']['lr'])
        wandb.env()
        wandb.conf()
        logging = GenericLogger(PARAM_DICT['state']['outputDirectory']+'/local.log')
        logging.logger = logging._configure_logger()
        trainer = training(databasefile='mindboggle.db',output_path=PARAM_DICT['state']['outputDirectory'],wandb = wandb, logger=logging,path = PARAM_DICT['state']['baseDirectory'],loader=loader, meshnet = meshnet, classes = PARAM_DICT['input']['classes'], learning_rate=PARAM_DICT['input']['lr'], Dice = dice)
        train_size = len(trainer.trainloader)
        iterations = PARAM_DICT['input']['epochs']*train_size
        trainer.save_test_nii()
        return distributed.initial_send_gradients(PARAM_DICT)
        
    else:
        if PARAM_DICT['input']['computation_phase']=='remote':
            current_iteration+=1
            trainer.current_iteration+=1
            return distributed.next_send_gradients(PARAM_DICT)
        else:
            raise ValueError("Error occurred at Local")

