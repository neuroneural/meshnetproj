import inspect
import logging
import torch


import requests
import os


def download_model(URL,path):
    file_path = os.path.join(path, 'models.py')
    # Make an HTTP GET request to the file URL
    response = requests.get(URL)
    if response.status_code == 200:
        # If the request is successful, save the content to the specified folder
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f'File downloaded to: {file_path}')
    else:
        print(f'Failed to download the file. Status code: {response.status_code}')


class trainer():
    def __init__(self, model, criterion, optimizer):
        self.model =model
        self.criterion =  criterion
        self.optimizer = optimizer
    
    def train(self, images, labels):
        self.model.train()
        running_loss = 0.0
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        running_loss += loss.item()
        local_gradients = [param.grad.clone() for param in self.model.parameters()]
        # Convert TensorFlow tensors to NumPy arrays
        numpy_arrays = [tensor.numpy() for tensor in local_gradients]
        # Convert NumPy arrays to nested lists
        nested_lists = [array.tolist() for array in numpy_arrays]
        return running_loss, nested_lists
    
    def optimize(self, avg_grad1):
        with torch.no_grad():
            for param, avg_grad in zip(self.model.parameters(), avg_grad1):
                if param.requires_grad:
                    avg_grad = torch.tensor(avg_grad)
                    avg_grad = avg_grad.to(param.grad.dtype)
                    param.grad = avg_grad
        self.optimizer.step()


class Dataloader():
    def __init__(self, data, val):
        self.data =  data
        self.val = val
    
    def get(self,set):
        count = 0
        for images, labels in self.data:
            if count == set:
                break
            else:
                count =  count+1
        return images, labels


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