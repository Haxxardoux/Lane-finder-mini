import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F
import mlflow
from torchvision import transforms
import mlflow

def start_mlflow_experiment(experiment_name, model_store):
    '''
    model_store options: pv-finder, lane-finder
    '''
    if model_store == 'lane-finder':
        mlflow.tracking.set_tracking_uri('file:/share/lazy/will/ConstrastiveLoss/Logs')
    if model_store == 'pv-finder':
        mlflow.tracking.set_tracking_uri('file:/share/lazy/pv-finder_model_repo')

    mlflow.set_experiment(experiment_name)

    
def save_to_mlflow(stats_dict:dict, params):
    '''
    Requires that the dictionary be structured as:
    Parameters have the previx "Param: ", metrics have "Metric: ", and artifacts have "Artifact: "
    It will ignore these tags. 
    
    Example: {'Parameter: Parameters':106125, 'Metric: Training Loss':10.523, 'Artifact':'run_stats.pyt'}
    '''
    for key, value in stats_dict.items():
        if 'Parameter: ' in key:
            mlflow.log_param(key[11:], value)
        if 'Metric: ' in key:
            mlflow.log_metric(key[8:], value)
        if 'Artifact' in key:
            mlflow.log_artifact(value)
    for key, value in vars(params).items():
        mlflow.log_param(key, value)

class Params(object):
    '''
    Order: batch_size, epochs, lr, img size
    '''
    def __init__(self, batch_size, epochs, lr, size, device):
        self.batch_size = batch_size
        self.epoch = epochs
        self.lr = lr
        self.size = size
        self.device = device

def count_parameters(model):
    """
    Counts the total number of parameters in a model
    Args:
        model (Module): Pytorch model, the total number of parameters for this model will be counted. 

    Returns: Int, number of parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)