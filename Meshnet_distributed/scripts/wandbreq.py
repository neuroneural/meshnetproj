import os
import wandb


class wandb_class():
    def __init__(self, key,lr):
        self.project = 'MeshNet-distributed'
        self.architecture = 'Meshnet'
        self.key = key
        self.dataset = 'Mindboggle'
        self.lr = lr
        self.url = ''
    
    def env(self):
        os.environ["WANDB_API_KEY"] = self.key

    def conf(self):
        self.url=wandb.init(
            project=self.architecture,
            config={
            "learning_rate": self.lr,
            "architecture": self.architecture,
            "dataset": self.dataset,
            }
        )


    def log(self,log_name, log_value):
        wandb.log({log_name:log_value})
    
    def end(self):
        wandb.finish()


        