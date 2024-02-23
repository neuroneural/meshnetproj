import os
import torch
import logging
import inspect
import nibabel as nib
import numpy as np

class GenericLogger:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.logger = self._configure_logger()

    def _configure_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def log_message(self, message, level=logging.INFO):
        if level == logging.DEBUG:
            frame = inspect.currentframe().f_back
            filename = inspect.getframeinfo(frame).filename
            function_name = frame.f_code.co_name
            line_number = frame.f_lineno
            caller_info = f"{filename} - {function_name} - Line {line_number}"
            message = f"{caller_info} - {message}"
        self.logger.log(level, message)


class training():
    def __init__(self, wandb = '',path ='',output_path = '',logger = '', databasefile = 'mindboggle.db', loader='', meshnet= '', learning_rate=0.0004,modelAE ='modelAE.json', classes =3,  Dice = '',cubes = 1):
        self.dabasefile = databasefile
        self.path = path
        self.wandb = wandb
        self.learning_rate = learning_rate
        self.cubes = cubes
        self.criterion = torch.nn.CrossEntropyLoss()
        self.dice = Dice
        self.logger = logger
        self.output = output_path
        self.classes = classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = meshnet.enMesh_checkpoint(1, self.classes, 1, self.path+'/'+modelAE)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        self.trainloader, self.validloader, self.testloader = loader.Scanloader(self.path+'/'+self.dabasefile, label_type='GWlabels', num_cubes=self.cubes).get_loaders()
        self.shape = 256 // self.cubes
        self.logger.log_message('Logging started')
        self.logger.log_message('Wandb URL'+str(self.wandb.url.url))
        self.current_iteration = 0
    
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for image, label in self.validloader:
                output = self.model(image.reshape(-1,1,self.shape,self.shape,self.shape))
                loss = self.criterion(output, label.reshape(-1, self.shape, self.shape, self.shape).long() * 2)
                if self.cubes == 1:
                    dice_loss = self.dice.faster_dice(torch.argmax(torch.squeeze(output), 0), label.reshape(self.shape, self.shape, self.shape) * 2, labels=[i for i in range(self.classes)])
                else:
                    dice_loss = self.dice.faster_dice(torch.argmax(torch.squeeze(output), 1), label.reshape(-1, self.shape, self.shape, self.shape) * 2, labels=[i for i in range(self.classes)])
                for i in range(self.classes):
                    self.wandb.log('Val Dice '+str(i),dice_loss[i])
                self.wandb.log('Val Dice Score',dice_loss.mean().item())
                self.wandb.log('Val Loss',loss.item())
                
    def get_train(self):
        for batch_id, (image,label) in enumerate(self.trainloader):
            if batch_id == int(self.current_iteration%len(self.trainloader)):
                if self.current_iteration!=0 and self.current_iteration%len(self.trainloader) == 0:
                    self.validate()
                    self.save_test_pred_nii()
                    self.save()
                return image, label

    def save(self):
        torch.save(self.model.state_dict(), self.output+'/meshnet.pth')
    
    def get_gradients(self):
        image,  label = self.get_train()
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(image.reshape(-1, 1, self.shape, self.shape, self.shape))
        loss = self.criterion(output, label.reshape(-1, self.shape, self.shape, self.shape).long() * 2)
        if self.cubes == 1:
            dice_loss = self.dice.faster_dice(torch.argmax(torch.squeeze(output), 0), label.reshape(self.shape, self.shape, self.shape) * 2, labels=[i for i in range(self.classes)])
        else:
            dice_loss = self.dice.faster_dice(torch.argmax(torch.squeeze(output), 1), label.reshape(-1, self.shape, self.shape, self.shape) * 2, labels=[i for i in range(self.classes)])
        loss.backward()

        self.logger.log_message('Epoch :'+str(self.current_iteration//len(self.trainloader))+' Batch : '+str(int(self.current_iteration%len(self.trainloader)))+' loss :'+str(loss.item())+' Dice :'+str(dice_loss.mean().item()), level=logging.INFO)
        self.wandb.log('Dataset Batch '+str(int(self.current_iteration%len(self.trainloader)))+'  Loss',loss.item())
        for i in range(self.classes):
            self.wandb.log('Dice '+str(i),dice_loss[i])

        self.wandb.log('Dice Score',dice_loss.mean().item())

        local_gradients = [param.grad.clone() for param in self.model.parameters()]
        numpy_arrays = [tensor.numpy() for tensor in local_gradients]
        nested_lists = [array.tolist() for array in numpy_arrays]
        return nested_lists
    
    def agg_gradients(self, agg_grad):
        agg_grad = [np.array(array) for array in agg_grad] 
        with torch.no_grad():
            for param, avg_grad in zip(self.model.parameters(), agg_grad):
                if param.requires_grad:
                    avg_grad = torch.tensor(avg_grad)
                    avg_grad = avg_grad.to(param.grad.dtype)
                    param.grad = avg_grad
        self.optimizer.step()


    def save_test_pred_nii(self):
        for batch_id, (image,label) in enumerate(self.testloader):
            output = self.model(image.reshape(-1,1,self.shape,self.shape,self.shape))
            if self.cubes == 1:
                output = torch.argmax(torch.squeeze(output), 0)
            else:
                output = torch.argmax(torch.squeeze(output), 1)
            
            output=output.reshape(self.shape,self.shape,self.shape).to(torch.device('cpu')).numpy()
            output = output.astype(np.uint16)
            output = nib.Nifti1Image(output, affine=np.eye(4))
            nib.save(output
                     , self.output+'/images/'+str(batch_id+1)+'/prediciton.nii.gz')

    def save_test_nii(self):
        if not os.path.exists(self.output+'/images'):
            os.mkdir(self.output+'/images')
        for batch_id, (image,label) in enumerate(self.testloader):
            image=image.reshape(self.shape,self.shape,self.shape).to(torch.device('cpu')).numpy()
            label=label.reshape(self.shape,self.shape,self.shape).to(torch.device('cpu')).numpy()
            image = nib.Nifti1Image(image, affine=np.eye(4))
            label = nib.Nifti1Image(label, affine=np.eye(4))          
            os.mkdir(self.output+'/images/'+str(batch_id+1))
            nib.save(image, self.output+'/images/'+str(batch_id+1)+'/image.nii.gz')
            nib.save(label, self.output+'/images/'+str(batch_id+1)+'/label.nii.gz')



