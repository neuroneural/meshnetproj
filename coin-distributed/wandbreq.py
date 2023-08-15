import os
import wandb


class wandb_class():
    def __init__(self,project, architecture, key, epochs, dataset):
        self.project = project
        self.architecture = architecture
        self.key = key
        self.epochs = epochs
        self.dataset = dataset
    
    def env(self):
        os.environ["WANDB_API_KEY"] = self.key

    def conf(self):
        wandb.init(
            project=self.architecture,
            config={
            "learning_rate": 0.02,
            "architecture": self.architecture,
            "dataset": self.dataset,
            "epochs": self.epochs,
            }
        )

    def log(self,log_name, log_value):
        wandb.log({log_name:log_value})
    
    def end(self):
        wandb.finish()


        