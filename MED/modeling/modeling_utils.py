import mlflow.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, jaccard_score, confusion_matrix
import numpy as np
import os
import mlflow
import pandas as pd
import time
import copy

from .models import CNN, LSTM, Siamese_CNN, Siamese_LSTM, FeatureExtractor
from .models_TCN import MultiStageModel, Transformer
from .models_COG import COG

def define_inputs(images: torch.Tensor,
                    kinematics: torch.Tensor,
                    feature_extractor: torch.nn.Module,
                    exp_kwargs: dict,
                    device: torch.device) -> torch.Tensor:
    """
    Define the inputs for the model based on the experiment configuration.

    Args:
        images (torch.Tensor): The images tensor.
        kinematics (torch.Tensor): The kinematics tensor.
        feature_extractor (torch.nn.Module): The feature extractor module.
        exp_kwargs (dict): Additional experiment parameters.

    Returns:
        torch.Tensor: The concatenated inputs tensor.
    """

    #Instantiate input depending on the data type
    if exp_kwargs['data_type'] == "multimodal":

        images = images.to(device)
        image_features = feature_extractor(images)
        kinematics = kinematics.to(device)
        if exp_kwargs['model_name'] == "COG":
            #Only do not permute dimensions if model is COG
            inputs = torch.cat((image_features, kinematics), dim=2)
        else:
            inputs = torch.cat((image_features, kinematics), dim = 2).permute(0, 2, 1) #change order to (batch_size, features, time_steps)
    
    elif exp_kwargs['data_type'] == "kinematics":

        if exp_kwargs['model_name'] == "COG":
            #Only do not permute dimensions if model is COG
            inputs = kinematics
        else:
            inputs = kinematics.permute(0, 2, 1)
        inputs = inputs.to(device)
    
    elif exp_kwargs['data_type'] == "video":
        
        images = images.to(device)
        if exp_kwargs['video_dims'] == 2048:
            #Only do not permute dimensions if model is COG
            if exp_kwargs['model_name'] == "COG":
                inputs = images
            else:
                inputs = images.permute(0, 2, 1)

        else:
            image_features = feature_extractor(images)
            #Only do not permute dimensions if model is COG
            if exp_kwargs['model_name'] == "COG":
                inputs = image_features
            else:
                inputs = image_features.permute(0, 2, 1)
        

    else:
        raise ValueError(f"Data type {exp_kwargs['data_type']} is not supported.")
    
    #Check if inputs are empty
    if inputs.size(0) == 0:
        raise ValueError("Inputs tensor is empty. Check the data loader and the inputs.")

    return inputs


def define_inputs_siamese(paired_images: torch.Tensor,  
                            paired_kinematics: torch.Tensor,
                            feature_extractor: torch.nn.Module,
                            exp_kwargs: dict,
                            device: torch.device) -> torch.Tensor:
    
    """
    Define the inputs for the Siamese model based on the experiment configuration.

    Args:
        paired_images (torch.Tensor): The paired images tensor.
        paired_kinematics (torch.Tensor): The paired kinematics tensor.
        feature_extractor (torch.nn.Module): The feature extractor module.
        exp_kwargs (dict): Additional experiment parameters.
        device (torch.device): The device to use.
    
    Returns:
        torch.Tensor: The concatenated inputs tensor for the Siamese model.
    """

    #Instantiate input depending on the data type
    if exp_kwargs['data_type'] == "multimodal":
        paired_images = paired_images.to(device)
        paired_image_features = feature_extractor(paired_images)
        paired_kinematics = paired_kinematics.to(device)
        inputs = torch.cat((paired_image_features, paired_kinematics), dim=3).permute(0, 1, 3, 2)
    
    elif exp_kwargs['data_type'] == "kinematics":
        paired_kinematics = paired_kinematics.permute(0, 1, 3, 2) 
        inputs = paired_kinematics.to(device)
        
    elif exp_kwargs['data_type'] == "video":
        paired_images = paired_images.to(device)
        if exp_kwargs['video_dims'] == 2048:
            inputs = paired_images.permute(0, 2, 1)
        
        else:
            paired_image_features = feature_extractor(paired_images)
            inputs = paired_image_features.permute(0, 1, 3, 2)
    
    else:
        raise ValueError(f"Data type {exp_kwargs['data_type']} is not supported.")
    
    #Check if inputs are empty
    if inputs.size(0) == 0:
        raise ValueError("Inputs tensor is empty. Check the data loader and the inputs.")
    
    return inputs


def define_error_labels(e_labels: torch.Tensor, exp_kwargs: dict) -> torch.Tensor:
    """
    Define the error labels based on the experiment configuration.

    Args:
        e_labels (torch.Tensor): The error labels tensor.
        exp_kwargs (dict): Additional experiment parameters.

    #e_labels is a tensor of shape (n_samples, 5) where each column corresponds to a different error type.    
    #Error types: Out_Of_View', 'Needle_Drop', 'Multiple_Attempts', 'Needle_Position', 'Error'

    Returns:
        torch.Tensor: The defined error labels tensor.
    """

    
    #error_dict = {
    #    'Out_Of_View': 0,
    #    'Needle_Drop': 1,
    #    'Multiple_Attempts': 2,
    #    'Needle_Position': 3,
    #    'global': -1
    #}
    
    error_dict = {
        'No Error': 0,
        'Out_Of_View': 1,
        'Multiple_Attempts': 2,
        'Needle_Position': 3,
        'Out_Of_View_Multiple_Attempts': 4,
        'Multiple_Attempts_Needle_Position': 5,
        'global': -1,
        'all_errors': [0, 1, 2, 3, 4, 5], #list of all error types (and no error)
    }

    #Check input error_type 
    if 'error_type' not in exp_kwargs:
        raise ValueError("error_type must be defined in exp_kwargs.")
    
    elif exp_kwargs['error_type'] not in error_dict:
        raise ValueError(f"Error type {exp_kwargs['error_type']} is not supported. Supported error types are: {list(error_dict.keys())}.")
    
    else:
        #Get the position of the error type in the e_labels tensor
        error_position = error_dict[exp_kwargs['error_type']]
        
        if exp_kwargs['dataset_type'] == "window":
            #Extract the specific error labels based on the error type
            e_labels_specific = e_labels[:, error_position]
        
        elif exp_kwargs['dataset_type'] == "frame":
            #Extract the specific error labels based on the error type
            e_labels_specific = e_labels[:, :, error_position] 

    return e_labels_specific


def define_model_objects(exp_kwargs: dict,
                         in_features_dict: dict,
                         device: torch.device,
                         class_counts: tuple,
                         window_size: int=0) -> tuple:

    """
    Define the model objects, loss function, optimizer, and scheduler based on the experiment configuration.

    Args:
        exp_kwargs (dict): Additional experiment parameters.

    Returns:
        tuple: A tuple containing the feature extractor, model, loss function, optimizer, and scheduler.
    """
    
    #a. Define feature extractor (if using video data) and model to be used
    torch.mps.manual_seed(42)
    torch.manual_seed(42)
    model = instantiate_model(exp_kwargs = exp_kwargs,
                            in_features = in_features_dict[exp_kwargs['data_type']],
                            window_size = window_size,
                            device=device).to(device)
    
    if exp_kwargs['data_type'] != "kinematics":
        
        feature_extractor = FeatureExtractor(input_dim=2048, output_dim=exp_kwargs['video_dims'], hidden_dims=[512, 256]).to(device)
        optimizer = torch.optim.Adam(list(feature_extractor.parameters()) + list(model.parameters()), lr=exp_kwargs['lr'],
                                    weight_decay=exp_kwargs['weight_decay'])
        
        print("Number of parameters to optimize:", sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad) + \
            sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    else: 
        feature_extractor = None
        optimizer = torch.optim.Adam(list(model.parameters()), lr=exp_kwargs['lr'], weight_decay=exp_kwargs['weight_decay'])
        print("Number of parameters to optimize:", sum(p.numel() for p in model.parameters() if p.requires_grad))


    #b. Define loss function and optimizer
    if exp_kwargs['pos_weight']:

        if exp_kwargs['error_type'] == 'global':
            pos_weight = torch.tensor(class_counts[0] / class_counts[1], device=device, dtype=torch.float32) #downsample the positive class
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        elif exp_kwargs['error_type'] == 'all_errors':
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_counts, device=device, dtype=torch.float32))
    
    else:
        if exp_kwargs['dataset_type'] == "window":
            if exp_kwargs['error_type'] == 'global':
                criterion = nn.BCEWithLogitsLoss()
            elif exp_kwargs['error_type'] == 'all_errors': #6 classes
                criterion = nn.CrossEntropyLoss()

        elif exp_kwargs['dataset_type'] == "frame":
            if exp_kwargs['error_type'] == 'sequential':
                criterion = nn.CrossEntropyLoss(reduction='none') #Use CrossEntropyLoss for frame classification
            else:
                criterion = nn.CrossEntropyLoss()


    if exp_kwargs['lr_scheduler']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=exp_kwargs['n_epochs'], eta_min=1e-6)
    
    else: scheduler = None

    return feature_extractor, model, criterion, optimizer, scheduler


def compute_loss(outputs: torch.Tensor,
                 e_labels: torch.Tensor,
                 criterion: torch.nn.Module,
                 dataset_type: str) -> torch.Tensor:

    if dataset_type == "window":
        try:
            outputs = outputs.squeeze(1)
        except:
            pass

        loss = criterion(outputs, e_labels)

    elif dataset_type == "frame":
        
        #Loss is computed as the average loss across mstcn stages
        stages = outputs.shape[0]

        #Extract no error vector
        no_e_labels = 1 - e_labels
        no_e_labels = no_e_labels.to(outputs.device) #Move to the same device as outputs
        e_labels_complete = torch.cat((no_e_labels, e_labels), dim=0) #Concatenate no error and error labels; shape (2, n_frames)
        e_labels_complete = e_labels_complete.transpose(1, 0)

        clc_loss = 0.0
        for j in range(stages):
            p_classes = p_classes = outputs[j].squeeze().transpose(1, 0)
            ce_loss = criterion(p_classes, e_labels_complete)
            clc_loss += ce_loss
        
        loss = clc_loss / (stages * 1.0)
    
    return loss, outputs


def train_single_epoch(model: torch.nn.Module, 
                       feature_extractor: torch.nn.Module,
                       train_dataloader: DataLoader,
                       criterion: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler._LRScheduler,    
                       device: torch.device,
                       exp_kwargs: dict) -> tuple:
    
    """
    Train the model for a single epoch.

    Args:
        model (torch.nn.Module): The model to train.
        feature_extractor (torch.nn.Module): The feature extractor to use.
        train_dataloader (DataLoader): The dataloader for the training set.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        device (torch.device): The device to train on.
        exp_kwargs (dict): Additional experiment parameters.

    Returns:
        tuple: A tuple containing the average loss, F1 score, accuracy, AUC ROC score, and Jaccard index for the training set.
    """
    if exp_kwargs['data_type'] != "kinematics":
        feature_extractor.train()
        model.train()
    
    else:
        model.train()
    
    train_loss, train_f1, train_f1_weighted, train_acc, train_jaccard = 0.0, 0.0, 0.0, 0.0, 0.0
    train_all_probs, train_all_preds, train_all_labels, train_all_subjects = [], [], [], []
    train_cm = np.zeros((2, 2), dtype=int)  # Assuming binary classification

    for i, batch in enumerate(tqdm.tqdm(train_dataloader, 
                                        total=len(train_dataloader),
                                        )):
        
        #images, kinematics, g_labels, e_labels, task, trial, subject = batch
        try:
            images, kinematics, g_labels, e_labels, subject = batch
        except:
            images, kinematics, g_labels, e_labels, subject, skill_level = batch

        g_labels = g_labels.to(device).float()
        e_labels_specific = define_error_labels(e_labels = e_labels, exp_kwargs=exp_kwargs)
        e_labels_specific = e_labels_specific.to(device).float()
        
        inputs = define_inputs(images=images,
                               kinematics=kinematics,
                               feature_extractor=feature_extractor,
                               exp_kwargs=exp_kwargs,
                               device=device) #inputs should be of size (batch_size, features, time_steps)
        
        outputs = model(inputs)
        loss, outputs = compute_loss(outputs=outputs,
                            e_labels=e_labels_specific,
                            criterion=criterion,
                            dataset_type=exp_kwargs['dataset_type'])
    

        #Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        #Metrics
        if exp_kwargs['dataset_type'] == "frame":
            _, outputs_binary = torch.max(outputs[-1].squeeze().transpose(1, 0).data, 1)
            e_labels_specific = e_labels_specific.squeeze(0)  #Remove the extra dimension for frame classification

        else:
            outputs_sigmoid = torch.sigmoid(outputs)
            outputs_binary = (outputs_sigmoid > 0.5).float()

        train_f1 += f1_score(e_labels_specific.detach().cpu().numpy(), outputs_binary.detach().cpu().numpy(), average='binary', pos_label=1)
        train_f1_weighted += f1_score(e_labels_specific.detach().cpu().numpy(), outputs_binary.detach().cpu().numpy(), average='weighted')
        train_acc += accuracy_score(e_labels_specific.detach().cpu().numpy(), outputs_binary.detach().cpu().numpy())
        train_jaccard += jaccard_score(e_labels_specific.detach().cpu().numpy(), outputs_binary.detach().cpu().numpy(), average='binary', pos_label=1)
        train_cm += confusion_matrix(e_labels_specific.detach().cpu().numpy(), outputs_binary.detach().cpu().numpy())

        if exp_kwargs['return_train_preds']:
            for j in range(len(outputs)):
                try:
                    train_all_probs.append(outputs_sigmoid[j].item())
                    train_all_subjects.append(subject[j])
                except:
                    train_all_probs = []
                    train_all_subjects.append(subject)
                train_all_preds.append(outputs_binary[j].item())
                train_all_labels.append(e_labels_specific[j].item())
                

    if scheduler is not None:
        scheduler.step()
    
    train_loss /= len(train_dataloader)
    train_f1 /= len(train_dataloader)
    train_f1_weighted /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    train_jaccard /= len(train_dataloader)

    if exp_kwargs['return_train_preds']:
        return train_loss, train_f1, train_f1_weighted, train_acc, train_jaccard, train_cm, train_all_probs, train_all_preds, train_all_labels, train_all_subjects    
    else:
        return train_loss, train_f1, train_f1_weighted, train_acc, train_jaccard, train_cm


def train_single_epoch_ES(model: torch.nn.Module, 
                       feature_extractor: torch.nn.Module,
                       train_dataloader: DataLoader,
                       criterion: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler._LRScheduler,    
                       device: torch.device,
                       exp_kwargs: dict) -> tuple:
    
    """
    Train the model for a single epoch.

    Args:
        model (torch.nn.Module): The model to train.
        feature_extractor (torch.nn.Module): The feature extractor to use.
        train_dataloader (DataLoader): The dataloader for the training set.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        device (torch.device): The device to train on.
        exp_kwargs (dict): Additional experiment parameters.

    Returns:
        tuple: Expanded return signature to match train_single_epoch_COG:
               - For global: average loss, F1, F1 weighted, accuracy, Jaccard, CM, (optional: probs, preds, labels, subjects)
               - For all_errors: binary F1, macro F1, binary accuracy, macro accuracy, binary Jaccard, macro Jaccard,
                                 CM binary, CM macro, (optional: probs, preds, labels, binary labels/preds)
    """
    if exp_kwargs['data_type'] != "kinematics":
        feature_extractor.train()
        model.train()
    else:
        model.train()
    
    train_loss = 0.0

    # === MOD: Added storage lists to record predictions, probs, labels, subjects ===
    train_all_probs = []
    train_all_preds = []
    train_all_preds_binary = []
    train_all_labels = []
    train_all_labels_binary = []
    train_all_subjects = []

    for i, batch in enumerate(tqdm.tqdm(train_dataloader, total=len(train_dataloader))):
        
        if len(batch) == 6:  
            images, kinematics, g_labels, e_labels, subject, skill_level = batch
        else:
            images, kinematics, g_labels, e_labels, subject = batch

        g_labels = g_labels.to(device).float()
        e_labels_specific = define_error_labels(e_labels=e_labels, exp_kwargs=exp_kwargs)
        e_labels_specific = e_labels_specific.to(device).float()
        if exp_kwargs['dataset_type'] == "frame":
            e_labels_specific = torch.argmax(e_labels_specific, dim=2)# 0-5 class labels
        else:
            e_labels_specific = torch.argmax(e_labels_specific, dim=1) #window labels
      
        e_labels_specific = e_labels_specific.view(-1,)

        inputs = define_inputs(images=images,
                               kinematics=kinematics,
                               feature_extractor=feature_extractor,
                               exp_kwargs=exp_kwargs,
                               device=device)
        
        outputs = model(inputs)
        loss, outputs = compute_loss(outputs=outputs,
                                     e_labels=e_labels_specific.float(),
                                     criterion=criterion,
                                     dataset_type=exp_kwargs['dataset_type'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        
        if exp_kwargs['dataset_type'] == "frame":
            _, preds = torch.max(outputs[-1].squeeze().transpose(1, 0).data, 1)
            probs = torch.softmax(outputs[-1].squeeze().transpose(1, 0), dim=1)
            probs_positive = probs[:, 1]
        else:
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            probs_positive = probs[:, 1]
        
        for index in range(len(preds)):
            pred = preds[index]
            label = e_labels_specific[index]
            train_all_preds.append(int(pred))
            train_all_labels.append(int(label))

            # Binary indicators
            if label.sum() > 0:
                train_all_labels_binary.append(1)
            else:
                train_all_labels_binary.append(0)
            if pred.sum() > 0:
                train_all_preds_binary.append(1)
            else:
                train_all_preds_binary.append(0)

    if scheduler is not None:
        scheduler.step()

    train_average_loss = float(train_loss) / len(train_dataloader)
    
    #Binary metrics (error vs no error)
    train_f1_binary = f1_score(train_all_labels_binary, train_all_preds_binary, average='binary', pos_label=1)
    train_jaccard_binary = jaccard_score(train_all_labels_binary, train_all_preds_binary, average='binary', pos_label=1)
    train_accuracy_binary = accuracy_score(train_all_labels_binary, train_all_preds_binary)
    train_cm_binary = confusion_matrix(train_all_labels_binary, train_all_preds_binary)

    #Macro metrics (across all error types)
    train_f1_macro = f1_score(train_all_labels, train_all_preds, average='macro')
    train_jaccard_macro = jaccard_score(train_all_labels, train_all_preds, average='macro')
    train_accuracy_macro = accuracy_score(train_all_labels, train_all_preds)
    train_cm_macro = confusion_matrix(train_all_labels, train_all_preds)

    if exp_kwargs['return_train_preds']:
        print("Unique labels in train_all_labels:", np.unique(train_all_labels, return_counts=True))
        print("Unique preds in train_all_preds:", np.unique(train_all_preds, return_counts=True))
        return (train_average_loss, train_f1_binary, train_f1_macro, train_accuracy_binary, train_accuracy_macro,
                train_jaccard_binary, train_jaccard_macro,
                train_cm_binary, train_cm_macro,
                train_all_probs, train_all_preds, train_all_labels,
                train_all_labels_binary, train_all_preds_binary)
    else:
        return train_average_loss, train_f1_binary, train_f1_macro, train_accuracy_binary, train_accuracy_macro, train_jaccard_binary, train_jaccard_macro, train_cm_binary, train_cm_macro



def train_single_epoch_Sequential(model: torch.nn.Module,
                                  feature_extractor: torch.nn.Module,
                                  train_dataloader: DataLoader,
                                  criterion: torch.nn.Module,
                                  optimizer: torch.optim.Optimizer,
                                  device: torch.device,
                                  scheduler: torch.optim.lr_scheduler._LRScheduler,
                                  exp_kwargs: dict):
    
    """
    Train the model for a single epoch in the sequential architecture

    Args:
        model (torch.nn.Module): The model to train.
        feature_extractor (torch.nn.Module): The feature extractor to use.
        train_dataloader (DataLoader): The dataloader for the training set.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        device (torch.device): The device to train on.
        exp_kwargs (dict): Additional experiment parameters.

    Returns:
        tuple: Expanded return signature to match train_single_epoch_COG:
               - For global: average loss, F1, F1 weighted, accuracy, Jaccard, CM, (optional: probs, preds, labels, subjects)
               - For all_errors: binary F1, macro F1, binary accuracy, macro accuracy, binary Jaccard, macro Jaccard,
                                 CM binary, CM macro, (optional: probs, preds, labels, binary labels/preds)
    """
    
    if exp_kwargs['data_type'] != "kinematics":
        feature_extractor.train()
        model.train()
    else:
        model.train()
    
    train_loss = 0.0

    train_preds_all = []
    train_labels_all = []
    train_preds_error_specific = []
    train_labels_error_specific = []

    batch_size = exp_kwargs['batch_size']

    for i, batch in enumerate(tqdm.tqdm(train_dataloader, total=len(train_dataloader))):
        
        #Extract batch information
        if len(batch) == 6:  
            images, kinematics, g_labels, e_labels, subject, skill_level = batch
        else:
            images, kinematics, g_labels, e_labels, subject = batch

        g_labels = g_labels.to(device).float()
        e_labels_specific = define_error_labels(e_labels=e_labels, exp_kwargs=exp_kwargs)
        e_labels_specific = e_labels_specific.to(device).float()
        
        if exp_kwargs['dataset_type'] == "frame":
            e_labels_specific = torch.argmax(e_labels_specific, dim=2)# 0-5 class labels
        else:
            e_labels_specific = torch.argmax(e_labels_specific, dim=1) #window labels
        e_labels_specific = e_labels_specific.view(-1,)

        #Define inputs
        inputs = define_inputs(images=images,
                               kinematics=kinematics,
                               feature_extractor=feature_extractor,
                               exp_kwargs=exp_kwargs,
                               device=device)
    
        #Create label mask, we don't want to compute loss for frames which are not errors but were predicted as errors, as they will never be correct.
        label_mask = (e_labels_specific != 0).float()  #Assuming 0 is the no error class
        label_mask = label_mask.to(device)  #Move to the same device as the model

        #As the model will only output 5 classes (0-5), we subtract 1 from the labels to match the output shape.
        e_labels_specific = e_labels_specific - 1  #Now the labels are in the range [-1, 4], as the no error class (0) is not included in the output of the model.
        
        #Compute output and loss
        outputs = model(inputs)
        #Use the mask to restrict loss to frames that are errors
        criterion = nn.CrossEntropyLoss(reduction="none")
        loss = criterion(outputs, e_labels_specific)
        loss = loss * label_mask
        if label_mask.sum() > 0:
            loss = loss.sum() / label_mask.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        
        if exp_kwargs['dataset_type'] == "frame":
            _, preds = torch.max(outputs[-1].squeeze().transpose(1, 0).data, 1)
            probs = torch.softmax(outputs[-1].squeeze().transpose(1, 0), dim=1)
            probs_positive = probs[:, 1]
        
        else:
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            probs_positive = probs[:, 1]
        
        for index in range(len(preds)):

            #i. Add 1 to pred and label
            pred = preds[index] + 1
            label = e_labels_specific[index] + 1

            #Change prediction to 0 if true label was 0
            if label == 0:
                pred = 0

            train_preds_all.append(int(pred))
            train_labels_all.append(int(label))

            #iii. Error-specific predictions and labels, i.e., metrics for unmasked frames (true errors that were predicted as errors in the binary model,
            #and are now being evaluated for specific error prediction)
            if label_mask[index] > 0:
                if label == 0:
                    print("AI DIOS MÍO, subject: {}, index: {}, label: {}, pred: {}".format(subject, index, label, pred))  #Debugging line to check the labels and predictions
                train_preds_error_specific.append(int(pred))
                train_labels_error_specific.append(int(label))

    if scheduler is not None:
        scheduler.step()

    train_average_loss = float(train_loss) / len(train_dataloader)
    
    #All error prediction (6 classes) --> use weighted metrics
    train_f1_all = f1_score(train_labels_all, train_preds_all, average='macro')
    train_jaccard_all = jaccard_score(train_labels_all, train_preds_all, average='macro')
    train_accuracy_all = accuracy_score(train_labels_all, train_preds_all)
    train_cm_all = confusion_matrix(train_labels_all, train_preds_all)

    #Only Error-specific metrics (5 classes) --> use weighted metrics
    train_f1_error_specific = f1_score(train_labels_error_specific, train_preds_error_specific, average='macro')
    train_f1_error_specific_weighted = f1_score(train_labels_error_specific, train_preds_error_specific, average='weighted')
    train_jaccard_error_specific = jaccard_score(train_labels_error_specific, train_preds_error_specific, average='macro')
    train_jaccard_error_specific_weighted = jaccard_score(train_labels_error_specific, train_preds_error_specific, average='weighted')
    train_accuracy_error_specific = accuracy_score(train_labels_error_specific, train_preds_error_specific)
    train_cm_error_specific = confusion_matrix(train_labels_error_specific, train_preds_error_specific)
    
    return train_average_loss, train_f1_all, train_f1_error_specific, train_f1_error_specific_weighted, train_accuracy_all, train_accuracy_error_specific, \
        train_jaccard_all, train_jaccard_error_specific, train_jaccard_error_specific_weighted, train_cm_all, train_cm_error_specific



def validate_single_epoch(model: torch.nn.Module,
                          feature_extractor: torch.nn.Module,
                         test_dataloader: DataLoader,
                         criterion: torch.nn.Module,
                         device: torch.device,
                         exp_kwargs: dict) -> tuple:
    
    """
    Validate the model for a single epoch.

    Args:
        model (torch.nn.Module): The model to validate.
        feature_extractor (torch.nn.Module): The feature extractor to use.
        test_dataloader (DataLoader): The dataloader for the test set.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to validate on.
        exp_kwargs (dict): Additional experiment parameters.

    Returns:
        tuple: A tuple containing the average loss, F1 score, accuracy, AUC ROC score, and Jaccard index for the test set.
    """

    #Evaluate on test set
    if exp_kwargs['data_type'] != "kinematics":
        model.eval()
        feature_extractor.eval()
    
    else:
        model.eval()

    test_all_preds, test_all_probs, test_all_labels_specific, test_all_labels, test_all_gest_labels, test_all_subjects = [], [], [], [], [], []
    test_f1, test_f1_weighted, test_acc, test_jaccard, test_loss, total_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    test_cm = np.zeros((2, 2), dtype=int) 

    with torch.no_grad():

        for batch in tqdm.tqdm(test_dataloader, desc="Test"):
            try:
                images, kinematics, g_labels, e_labels, subject = batch
            except:
                images, kinematics, g_labels, e_labels, subject, skill_level = batch
            g_labels = g_labels.to(device).float()
            g_labels = g_labels.squeeze(0)  #Remove the extra dimension for frame classification
            e_labels_specific = define_error_labels(e_labels = e_labels, exp_kwargs=exp_kwargs)
            e_labels_specific = e_labels_specific.to(device).float()

            #Forward pass
            inputs = define_inputs(images=images,
                                   kinematics=kinematics,
                                   feature_extractor=feature_extractor,
                                   exp_kwargs=exp_kwargs,
                                   device=device)

            start_time = time.time()   
            outputs = model(inputs)
            end_time = time.time()
            outputs = outputs.squeeze(1)

            #Loss
            loss, outputs = compute_loss(outputs=outputs,
                            e_labels=e_labels_specific,
                            criterion=criterion,
                            dataset_type=exp_kwargs['dataset_type'])
            
            test_loss += loss.item()

            #Metrics
            if exp_kwargs['dataset_type'] == "frame":
                outputs_sigmoid = torch.sigmoid(outputs[-1])
                _, outputs_binary = torch.max(outputs[-1].squeeze().transpose(1, 0).data, 1)
                e_labels_specific = e_labels_specific.squeeze(0)

            else:
                outputs_sigmoid = torch.sigmoid(outputs)
                outputs_binary = (outputs_sigmoid > 0.5).float()
                
            
            for j in range(outputs_binary.shape[0]):
                test_all_preds.append(outputs_binary[j].item())  
                test_all_labels_specific.append(e_labels_specific[j].item())
                test_all_gest_labels.append(g_labels[j].item())

                if exp_kwargs['dataset_type'] == "frame":
                    test_all_labels.append(e_labels[0, j])
                    test_all_subjects.append(subject)
                    test_all_probs.append(None)
                else:
                    test_all_labels.append(e_labels[j])
                    test_all_probs.append(outputs_sigmoid[j].item())
                    test_all_subjects.append(subject[j])

            total_time = (end_time - start_time)

    test_loss /= len(test_dataloader)
    test_f1 = f1_score(test_all_labels_specific, test_all_preds, average='binary')
    test_f1_weighted = f1_score(test_all_labels_specific, test_all_preds, average='weighted')
    test_acc = accuracy_score(test_all_labels_specific, test_all_preds)
    test_jaccard = jaccard_score(test_all_labels_specific, test_all_preds, average='binary')
    test_cm = confusion_matrix(test_all_labels_specific, test_all_preds)

    inference_rate = total_time * 1000

    return test_loss, test_f1, test_f1_weighted, test_acc, test_jaccard, test_cm, inference_rate, test_all_preds, test_all_probs, test_all_labels, test_all_labels_specific, test_all_gest_labels, test_all_subjects


def validate_single_epoch_ES(model: torch.nn.Module,
                          feature_extractor: torch.nn.Module,
                          test_dataloader: DataLoader,
                          criterion: torch.nn.Module,
                          device: torch.device,
                          exp_kwargs: dict) -> tuple:
    
    """
    Validate the model for a single epoch.

    Handles both global binary classification and multi-class all_errors prediction.

    Returns:
        tuple: 
            For global: loss, f1, f1_weighted, acc, jaccard, cm, inference_rate, (optional preds, probs, labels, subjects)
            For all_errors: loss, f1_binary, f1_macro, acc_binary, acc_macro, jaccard_binary, jaccard_macro, cm_binary, cm_macro, inference_rate, (optional ...)
    """

    if exp_kwargs['data_type'] != "kinematics":
        model.eval()
        feature_extractor.eval()
    else:
        model.eval()

    test_loss, total_time = 0.0, 0.0

    test_all_preds = []
    test_all_preds_binary = []
    test_all_labels = []
    test_all_labels_binary = []
    test_all_probs = []
    test_all_subjects = []
    test_all_gest_labels = []

    with torch.no_grad():
        for batch in tqdm.tqdm(test_dataloader, desc="Test"):
            images, kinematics, g_labels, e_labels, subject = batch
            g_labels = g_labels.to(device).float()

            e_labels_specific = define_error_labels(e_labels=e_labels, exp_kwargs=exp_kwargs)
            e_labels_specific = e_labels_specific.to(device).float()

            if exp_kwargs['error_type'] == "global":
                e_labels_specific = e_labels_specific.view(-1,)
            else:  #all_errors multi-class
                if exp_kwargs['dataset_type'] == "frame":
                    e_labels_specific = torch.argmax(e_labels_specific, dim=2)
                else:
                    e_labels_specific = torch.argmax(e_labels_specific, dim=1)
                e_labels_specific = e_labels_specific.view(-1,)

            # Forward pass
            inputs = define_inputs(images=images,
                                   kinematics=kinematics,
                                   feature_extractor=feature_extractor,
                                   exp_kwargs=exp_kwargs,
                                   device=device)

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()

            # Loss
            loss, outputs = compute_loss(outputs=outputs,
                                         e_labels=e_labels_specific.float(),
                                         criterion=criterion,
                                         dataset_type=exp_kwargs['dataset_type'])
            test_loss += loss.item()

            if exp_kwargs['dataset_type'] == "frame":
                _, preds = torch.max(outputs[-1].squeeze().transpose(1, 0).data, 1)
                probs = torch.softmax(outputs[-1].squeeze().transpose(1, 0), dim=1)
                probs_positive = probs[:, 1]
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                probs_positive = probs[:, 1]

    
            for index in range(len(preds)):
                pred = preds[index]
                label = e_labels_specific[index]
                test_all_preds.append(int(pred))
                test_all_labels.append(int(label))
                test_all_labels_binary.append(1 if label != 0 else 0)
                test_all_preds_binary.append(1 if pred != 0 else 0)
                test_all_probs.append(probs_positive[index].item())
                test_all_subjects.append(subject[index])
                test_all_gest_labels.append(g_labels[index].item())

            total_time = (end_time - start_time)

    test_average_loss = test_loss / len(test_dataloader)

    #Binary metrics (error vs no error)
    test_f1_binary = f1_score(test_all_labels_binary, test_all_preds_binary, average='binary', pos_label=1)
    test_jaccard_binary = jaccard_score(test_all_labels_binary, test_all_preds_binary, average='binary', pos_label=1)
    test_accuracy_binary = accuracy_score(test_all_labels_binary, test_all_preds_binary)
    test_cm_binary = confusion_matrix(test_all_labels_binary, test_all_preds_binary)

    #Macro metrics (all errors)
    test_f1_macro = f1_score(test_all_labels, test_all_preds, average='macro')
    test_jaccard_macro = jaccard_score(test_all_labels, test_all_preds, average='macro')
    test_accuracy_macro = accuracy_score(test_all_labels, test_all_preds)
    test_cm_macro = confusion_matrix(test_all_labels, test_all_preds)

    return (test_average_loss, test_f1_binary, test_f1_macro, test_accuracy_binary, test_accuracy_macro,
                test_jaccard_binary, test_jaccard_macro,
                test_cm_binary, test_cm_macro,
                total_time * 1000,
                test_all_probs, test_all_preds, test_all_labels,
                test_all_labels_binary, test_all_preds_binary, test_all_gest_labels, test_all_subjects)


def validate_single_epoch_Sequential(model:torch.nn.Module,
                                     feature_extractor: torch.nn.Module,
                                     binary_model: torch.nn.Module,
                                     binary_feature_extractor: torch.nn.Module,
                                     test_dataloader: DataLoader,
                                     device: torch.device,
                                     exp_kwargs: dict) -> tuple:    

    """
    Validates a single epoch for sequential models (window and frame excluding TSVN and COG).

    Args:
        model (torch.nn.Module): The model to validate.
        feature_extractor (torch.nn.Module): The feature extractor to use.
        binary_model (torch.nn.Module): The binary model to use.
        binary_feature_extractor (torch.nn.Module): The binary feature extractor to use.
        test_dataloader (DataLoader): The dataloader for the test set.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to validate on.
        exp_kwargs (dict): Additional experiment parameters.

    """


    model.eval(), feature_extractor.eval(), binary_model.eval(), binary_feature_extractor.eval()

    test_preds_all = []
    test_labels_all = []
    test_preds_error_specific = []
    test_labels_error_specific = []
    test_probs_error_specific = []
    test_gest_labels = []
    test_subjects = []
    test_loss = 0.0
    total_time = 0.0

    with torch.no_grad():
        
        for batch in tqdm.tqdm(test_dataloader, desc="Test"):
            
            #1. Extract batch info
            images, kinematics, g_labels, e_labels, subject = batch
            g_labels = g_labels.to(device).float()
            e_labels_specific = define_error_labels(e_labels=e_labels, exp_kwargs=exp_kwargs)
            e_labels_specific = e_labels_specific.to(device).float()

            if exp_kwargs['dataset_type'] == "frame":
                e_labels_specific = torch.argmax(e_labels_specific, dim=2)
            
            else:
                e_labels_specific = torch.argmax(e_labels_specific, dim=1)
            
            e_labels_specific = e_labels_specific.view(-1,)

            #Subtract one to match output classes (5, as "no error" is not being considered)
            e_labels_specific = e_labels_specific - 1

            #1.1. Define inputs
            inputs_binary = define_inputs(images=images,
                                           kinematics=kinematics,
                                           feature_extractor=binary_feature_extractor,
                                           exp_kwargs=exp_kwargs,
                                           device=device)
            
            inputs_sequential = define_inputs(images=images,
                                              kinematics=kinematics,
                                              feature_extractor=feature_extractor,
                                              exp_kwargs=exp_kwargs,
                                              device=device)

            #2. Define mask (directly the predictions) using binary model. Instances predicted as no-error will be considered as such
            with torch.no_grad():
                binary_outputs = binary_model(inputs_binary)
                binary_preds = (binary_outputs > 0.5).float()
                
            #3. Compute output according to sequential model
            start_time = time.time()
            outputs = model(inputs_sequential)
            end_time = time.time()

            #4. Compute loss: use the mask to restrict loss to instances that are errors
            criterion = nn.CrossEntropyLoss(reduction="none")
            loss = criterion(outputs, e_labels_specific)
            loss = loss * binary_preds
            
            if binary_preds.sum() > 0:
                loss = loss.sum() / binary_preds.sum()
            
            else:
                loss = loss.mean()

            test_loss += loss.item()

            #5. Compute final predicitons and probabilities
            if exp_kwargs['dataset_type'] == "frame":
                _, preds = torch.max(outputs[-1].squeeze().transpose(1, 0).data, 1)
                probs = torch.softmax(outputs[-1].squeeze().transpose(1, 0), dim=1)
                probs_positive = probs[:, 1]
            
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

            #5. Store predictions and labels
            #Record error-specific predictions and labels, i.e., metrics for unmasked frames (true errors that were predicted as errors in the binary model,
            #and are now being evaluated for specific error prediction)
            for index in range(len(preds)):
                
                pred = preds.data[index] + 1  #Add 1 to the prediction to match the label range (0-5) as the model outputs 5 classes (0-4) and we want to include no error class (0).
                label = e_labels_specific.data[index] + 1  #Add 1 to the label to match the prediction range (0-5) as the model outputs 5 classes (0-4) and we want to include no error class (0).

                #If the subject mask is 0, it means this frame was not predicted as an error
                if binary_preds[index] == 0:  
                    pred = 0

                test_preds_all.append(int(pred))
                test_labels_all.append(int(label))

                if binary_preds[index] > 0:  #If the frame is an error and was predicted as an error in the binary model
                    
                    if label > 0:
                        test_preds_error_specific.append(int(pred))
                        test_labels_error_specific.append(int(label))
                        test_probs_error_specific.append(probs.data[index].tolist())  #Record the probabilities for the error-specific predictions

            total_time += (end_time - start_time) 

    test_average_loss = float(test_loss) / len(test_dataloader)
    inference_rate = (total_time / images.shape[1]) * 1000  #Convert to ms per frame

    #Specific error prediction (6 classes) --> use weighted metrics
    test_f1_all = f1_score(test_labels_all, test_preds_all, average='macro')
    test_jaccard_all = jaccard_score(test_labels_all, test_preds_all, average='macro')
    test_accuracy_all = accuracy_score(test_labels_all, test_preds_all)
    test_cm_all = confusion_matrix(test_labels_all, test_preds_all) 

    test_f1_error_specific = f1_score(test_labels_error_specific, test_preds_error_specific, average='macro')
    test_f1_error_specific_weighted = f1_score(test_labels_error_specific, test_preds_error_specific, average='weighted')
    test_jaccard_error_specific = jaccard_score(test_labels_error_specific, test_preds_error_specific, average='macro')
    test_jaccard_error_specific_weighted = jaccard_score(test_labels_error_specific, test_preds_error_specific, average='weighted')
    test_accuracy_error_specific = accuracy_score(test_labels_error_specific, test_preds_error_specific)
    test_cm_error_specific = confusion_matrix(test_labels_error_specific, test_preds_error_specific)

    return test_average_loss, test_f1_all, test_f1_error_specific, test_f1_error_specific_weighted, test_accuracy_all, test_accuracy_error_specific, \
        test_jaccard_all, test_jaccard_error_specific, test_jaccard_error_specific_weighted, test_cm_all, test_cm_error_specific, \
        inference_rate, test_preds_all, test_preds_error_specific, test_probs_error_specific, \
        test_labels_all, test_labels_error_specific, test_gest_labels, test_subjects



def train_single_epoch_siamese(model: torch.nn.Module, 
                       feature_extractor: torch.nn.Module,
                       train_dataloader: DataLoader,
                       criterion: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler._LRScheduler,    
                       device: torch.device,
                       exp_kwargs: dict) -> tuple:
    
    """
    Train the Siamese model for a single epoch.

    Args:
        model (torch.nn.Module): The Siamese model to train.
        feature_extractor (torch.nn.Module): The feature extractor to use.
        train_dataloader (DataLoader): The dataloader for the training set.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        device (torch.device): The device to train on.
        exp_kwargs (dict): Additional experiment parameters.
    
    Returns:
        tuple: A tuple containing the average loss, F1 score, accuracy, AUC ROC score, and Jaccard index for the training set.
    """

    if exp_kwargs['data_type'] != "kinematics":
        feature_extractor.train()
        model.train()
    
    else:
        model.train()
    
    train_loss, train_f1, train_f1_weighted, train_acc, train_auc_roc, train_jaccard = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    train_cm = np.zeros((2, 2), dtype=int) 

    for i, batch in enumerate(tqdm.tqdm(train_dataloader, 
                                        total=len(train_dataloader),
                                        )):
        
        paired_images, paired_kinematics, position_1, position_2, e_labels = batch #The shape of data is (B, 2, 30, 2048) for images and (B, 2, 30, 26) for kinematics. label is 1D
        e_labels = e_labels.to(device).float()

        inputs = define_inputs_siamese(paired_images=paired_images,
                                        paired_kinematics=paired_kinematics,
                                        feature_extractor=feature_extractor,
                                        exp_kwargs=exp_kwargs,
                                        device=device)
        if inputs.size(0) < 2: #handle single sample batches
            continue
        
        input1, input2 = inputs[:, 0, :, :], inputs[:, 1, :, :]
        outputs = model(input1, input2)
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, e_labels)

        #Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        #Metrics
        outputs_sigmoid = torch.sigmoid(outputs)
        outputs_binary = (outputs_sigmoid > 0.5).float()
        train_f1 += f1_score(e_labels.detach().cpu().numpy(), outputs_binary.detach().cpu().numpy(), average='binary')
        train_f1_weighted += f1_score(e_labels.detach().cpu().numpy(), outputs_binary.detach().cpu().numpy(), average='weighted')
        train_acc += accuracy_score(e_labels.detach().cpu().numpy(), outputs_binary.detach().cpu().numpy())
        train_auc_roc += roc_auc_score(e_labels.detach().cpu().numpy(), outputs_sigmoid.detach().cpu().numpy(), average='weighted')
        train_jaccard += jaccard_score(e_labels.detach().cpu().numpy(), outputs_binary.detach().cpu().numpy(), average='binary')
        train_cm += confusion_matrix(e_labels.detach().cpu().numpy(), outputs_binary.detach().cpu().numpy())

    if scheduler is not None:
        scheduler.step()
    
    train_loss /= len(train_dataloader)
    train_f1 /= len(train_dataloader)
    train_f1_weighted /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    train_auc_roc /= len(train_dataloader)
    train_jaccard /= len(train_dataloader)

    return train_loss, train_f1, train_f1_weighted, train_acc, train_auc_roc, train_jaccard, train_cm


def validate_single_epoch_siamese(model: torch.nn.Module,
                                    feature_extractor: torch.nn.Module,
                                    test_dataloader: DataLoader,
                                    criterion: torch.nn.Module,
                                    device: torch.device,
                                    exp_kwargs: dict) -> tuple:
        
        """
        Validate the Siamese model for a single epoch.
    
        Args:
            model (torch.nn.Module): The Siamese model to validate.
            feature_extractor (torch.nn.Module): The feature extractor to use.
            test_dataloader (DataLoader): The dataloader for the test set.
            criterion (torch.nn.Module): The loss function.
            device (torch.device): The device to validate on.
            exp_kwargs (dict): Additional experiment parameters.
    
        Returns:
            tuple: A tuple containing the average loss, F1 score, accuracy, AUC ROC score, and Jaccard index for the test set.
        """
    
        if exp_kwargs['data_type'] != "kinematics":
            model.eval()
            feature_extractor.eval()
        
        else:
            model.eval()
    
        test_loss, test_f1, test_acc, test_auc_roc, test_jaccard, total_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        test_cm = np.zeros((2, 2), dtype=int) 

        #Store the ids of test samples to then compute the average prediction across the paired training samples.
        id_test = np.empty((0, 1), dtype=int) 
        all_true = np.empty((0, 1), dtype=int) #Store the true labels for each sample
        all_preds = np.empty((0, 1), dtype=int) #Store the predictions for each sample
    
        with torch.no_grad():
            for batch in tqdm.tqdm(test_dataloader, desc="Test"):
                paired_images, paired_kinematics, position_1, position_2, e_labels = batch
                e_labels = e_labels.to(device).float()
    
                #Forward pass
                inputs = define_inputs_siamese(paired_images=paired_images,
                                                paired_kinematics=paired_kinematics,
                                                feature_extractor=feature_extractor,
                                                exp_kwargs=exp_kwargs,
                                                device=device)
                
                input1, input2 = inputs[:, 0, :, :], inputs[:, 1, :, :]
                start_time = time.time()
                outputs = model(input1, input2)
                end_time = time.time()
                outputs = outputs.squeeze(1)
    
                #Loss
                loss = criterion(outputs, e_labels)
                test_loss += loss.item()
    
                #Metrics
                outputs_sigmoid = torch.sigmoid(outputs)
                outputs_binary = (outputs_sigmoid > 0.5).float()
                all_preds = np.append(all_preds, outputs_binary.detach().cpu().numpy().reshape(-1, 1), axis=0) #Store the predictions for each sample
                all_true = np.append(all_true, e_labels.detach().cpu().numpy().reshape(-1, 1), axis=0) #Store the true labels for each sample
                id_test = np.append(id_test, position_2.detach().cpu().numpy().reshape(-1, 1), axis=0) #Store the ids of the test samples
                total_time += (end_time - start_time)
        
        id_test = np.array(id_test)
        id_test_unique = np.unique(id_test)

        y_true = []
        y_pred = []
        y_pred_vote = []

        for id in id_test_unique:
            #Get the indices of the samples with the same id
            indices = np.where(id_test == id)[0]
            
            if len(indices) > 0:
                #Get the outputs for those samples
                preds_id = all_preds[indices].reshape(-1, 1)
                true_id = all_true[indices].reshape(-1, 1)

                #Find label of the current id
                true_id = true_id[0] #Assuming all labels are the same for the same id

                #Compute the majority vote for the binary output
                vote = np.mean(preds_id)
                majority_vote = 1 if vote > 0.5 else 0
                
                y_true.append(true_id)
                y_pred.append(majority_vote)
                y_pred_vote.append(vote)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_vote = np.array(y_pred_vote)

        #Compute metrics
        test_f1 = f1_score(y_true, y_pred, average='binary')
        test_f1_weighted = f1_score(y_true, y_pred, average='weighted')
        test_acc = accuracy_score(y_true, y_pred)
        test_auc_roc = roc_auc_score(y_true, y_pred_vote, average='weighted')
        test_jaccard = jaccard_score(y_true, y_pred, average='binary')
        test_cm = confusion_matrix(y_true, y_pred)
        
        test_loss /= len(test_dataloader)
        inference_rate = (total_time / len(test_dataloader)) * 1000  #Convert to ms

        return test_loss, test_f1, test_f1_weighted, test_acc, test_auc_roc, test_jaccard, test_cm, inference_rate




def train_single_epoch_TSVN(model: torch.nn.Module, 
                        TeCNo: torch.nn.Module,
                       feature_extractor: torch.nn.Module,
                       train_dataloader: DataLoader,
                       criterion: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler._LRScheduler,    
                       device: torch.device,
                       exp_kwargs: dict) -> tuple:
    
    """
    Train the model for a single epoch.

    Args:
        model (torch.nn.Module): The model to train.
        feature_extractor (torch.nn.Module): The feature extractor to use.
        train_dataloader (DataLoader): The dataloader for the training set.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        device (torch.device): The device to train on.
        exp_kwargs (dict): Additional experiment parameters.

    Returns:
        tuple: A tuple containing the average loss, F1 score, accuracy, AUC ROC score, and Jaccard index for the training set.
    """
    model.train()
    
    train_all_scores = []
    train_all_preds = []
    train_all_labels = []
    train_loss = 0.0

    for i, batch in enumerate(tqdm.tqdm(train_dataloader, 
                                        total=len(train_dataloader),
                                        )):
        
        #images, kinematics, g_labels, e_labels, task, trial, subject = batch
        images, kinematics, g_labels, e_labels, subject, skill_level = batch
        g_labels = g_labels.to(device).float()
        e_labels_specific = define_error_labels(e_labels = e_labels, exp_kwargs=exp_kwargs)
        e_labels = e_labels_specific.to(device).float()
        
        inputs = define_inputs(images=images,
                               kinematics=kinematics,
                               feature_extractor=feature_extractor,
                               exp_kwargs=exp_kwargs,
                               device=device) #inputs should be of size (batch_size, features, time_steps)
        
        #First pass through TeCNo
        outputs = TeCNo(inputs)[-1].squeeze(1)

        #Then pass through TSVN
        p_classes1 = model(outputs, inputs.transpose(2, 1)).squeeze()

        #Re-define e_labels to match the output shape
        no_e_labels = 1 - e_labels_specific
        no_e_labels = no_e_labels.to(device) #Move to the same device as outputs
        e_labels_complete = torch.cat((no_e_labels, e_labels), dim=0) #Concatenate no error and error labels; shape (2, n_frames)
        e_labels_complete = e_labels_complete.transpose(1, 0)
        
        #Compupte loss and predictions
        loss = criterion(p_classes1, e_labels_complete)
        _, outputs_binary = torch.max(p_classes1.data, 1)
        
        optimizer.zero_grad()
        loss.backward()  #Backwards pass for TSVN
        optimizer.step()  
        train_loss += loss.item()

        #Metrics
        e_labels_specific = e_labels_specific.squeeze(0)  #Remove the extra dimension for frame classification

        for j in range(outputs_binary.shape[0]):
            train_all_scores.append(outputs_binary[j].item())
            train_all_preds.append(outputs_binary[j].item())
            train_all_labels.append(e_labels_specific[j].item())
        

    if scheduler is not None:
        scheduler.step()
    
    train_loss /= len(train_dataloader)
    train_f1 = f1_score(train_all_labels, train_all_preds, average='binary')
    train_f1_weighted = f1_score(train_all_labels, train_all_preds, average='weighted')
    train_acc = accuracy_score(train_all_labels, train_all_preds)
    train_jaccard = jaccard_score(train_all_labels, train_all_preds, average='binary')
    train_cm = confusion_matrix(train_all_labels, train_all_preds)

    return train_loss, train_f1, train_f1_weighted, train_acc, train_jaccard, train_cm


def validate_single_epoch_TSVN(model: torch.nn.Module,
                                TeCNo: torch.nn.Module,
                                feature_extractor: torch.nn.Module,
                                test_dataloader: DataLoader,
                                criterion: torch.nn.Module,
                                device: torch.device,
                                exp_kwargs: dict) -> tuple:
    
    """
    Validate the model for a single epoch.
    Args:
        model (torch.nn.Module): The model to validate.
        TeCNo (torch.nn.Module): The TeCNo model to use.
        feature_extractor (torch.nn.Module): The feature extractor to use.
        test_dataloader (DataLoader): The dataloader for the test set.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to validate on.
        exp_kwargs (dict): Additional experiment parameters.

    Returns:
        tuple: A tuple containing the average loss, F1 score, accuracy, AUC ROC score, and Jaccard index for the test set.

    """

    model.eval()

    test_all_probs = []
    test_all_preds = []
    test_all_labels = []
    test_all_labels_specific = []
    test_all_gest_labels = []
    test_all_subjects = []
    test_loss = 0.0
    total_time = 0.0

    with torch.no_grad():
        for batch in tqdm.tqdm(test_dataloader, desc="Test"):
            images, kinematics, g_labels, e_labels, subject, skill_level = batch
            g_labels = g_labels.to(device).float()
            g_labels = g_labels.squeeze(0)  #Remove the extra dimension for frame classification
            e_labels_specific = define_error_labels(e_labels = e_labels, exp_kwargs=exp_kwargs)
            e_labels_specific = e_labels_specific.to(device).float()

            #Forward pass
            inputs = define_inputs(images=images,
                                   kinematics=kinematics,
                                   feature_extractor=feature_extractor,
                                   exp_kwargs=exp_kwargs,
                                   device=device)

            start_time = time.time()   
            outputs = TeCNo(inputs)[-1].squeeze(1)  #Get the last output of TeCNo
            outputs = model(outputs, inputs.transpose(2, 1)).squeeze()  #Pass through TSVN
            end_time = time.time()
            
            #Loss
            no_e_labels = 1 - e_labels_specific
            no_e_labels = no_e_labels.to(outputs.device) 
            e_labels_complete = torch.cat((no_e_labels, e_labels_specific), dim=0) 
            e_labels_complete = e_labels_complete.transpose(1, 0)
            loss = criterion(outputs, e_labels_complete)
            test_loss += loss.item()

            #Metrics
            _, outputs_binary = torch.max(outputs.data, 1)
            e_labels_specific = e_labels_specific.squeeze(0)

            for j in range(outputs_binary.shape[0]):
                test_all_probs.append(outputs[j, 1].item())
                test_all_preds.append(outputs_binary[j].item())
                test_all_labels.append(e_labels[0, j])
                test_all_labels_specific.append(e_labels_specific[j].item())
                test_all_gest_labels.append(g_labels[j].item())
                test_all_subjects.append(subject)
            
            total_time += (end_time - start_time)

    test_loss /= len(test_dataloader)
    test_f1 = f1_score(test_all_labels_specific, test_all_preds, average='binary')
    test_f1_weighted = f1_score(test_all_labels_specific, test_all_preds, average='weighted')
    test_acc = accuracy_score(test_all_labels_specific, test_all_preds)
    test_jaccard = jaccard_score(test_all_labels_specific, test_all_preds, average='binary')
    test_cm = confusion_matrix(test_all_labels_specific, test_all_preds)
    inference_rate = total_time / images.shape[1] * 1000

    return test_loss, test_f1, test_f1_weighted, test_acc, test_jaccard, test_cm, inference_rate, test_all_preds, test_all_probs, test_all_labels, test_all_labels_specific, test_all_gest_labels, test_all_subjects



def train_single_epoch_COG(model: torch.nn.Module,
                           feature_extractor: torch.nn.Module,
                       train_dataloader: DataLoader,
                       criterion: torch.nn.Module,
                       criterion2: torch.nn.Module, 
                       optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler._LRScheduler,    
                       device: torch.device,
                       exp_kwargs: dict) -> tuple:
    
    """ Train COG for a single epoch.

    Args:
        model (torch.nn.Module): The COG model to train.
        feature_extractor (torch.nn.Module): The feature extractor to use.
        train_dataloader (DataLoader): The dataloader for the training set.
        criterion (torch.nn.Module): The loss function (CrossEntropyLoss).  
        criterion2 (torch.nn.Module): The smooth loss function (MSE)
        optimizer (torch.optim.Optimizer): The optimizer to use.
        device (torch.device): The device to train on.
        exp_kwargs (dict): Additional experiment parameters.
    Returns:
        tuple: A tuple containing the average loss, F1 score, accuracy, and Jaccard index for the training set.

    """


    if exp_kwargs['data_type'] != "kinematics":
        feature_extractor.train()
        model.train()
    
    else:
        model.train()
    
    train_all_probs = []
    train_all_preds = []
    train_all_preds_binary = []
    train_all_labels = []
    train_all_labels_binary = []
    train_all_subjects = []
    train_loss = 0.0

    for i, batch in enumerate(tqdm.tqdm(train_dataloader, 
                                        total=len(train_dataloader),
                                        )):
        
        #images, kinematics, g_labels, e_labels, task, trial, subject = batch
        images, kinematics, g_labels, e_labels, subject, skill_level = batch
        g_labels = g_labels.to(device).float()
        e_labels_specific = define_error_labels(e_labels = e_labels, exp_kwargs=exp_kwargs)
        e_labels_specific = e_labels_specific.to(device).float()

        if exp_kwargs['error_type'] == "global":
            e_labels_specific = e_labels_specific.view(-1,)  #Ensure e_labels is of shape (n_frames,)

        else: #specific error prediction (5/6 classes for CE loss)
            e_labels_specific = torch.argmax(e_labels_specific, dim=2)  #Convert to class labels (0-5) for CE loss
            e_labels_specific = e_labels_specific.view(-1,)

        inputs = define_inputs(images=images,
                               kinematics=kinematics,
                               feature_extractor=feature_extractor,
                               exp_kwargs=exp_kwargs,
                               device=device) #inputs should be of size (batch_size, features, time_steps)

        predicted_list, feature_list = model.forward(inputs)
        all_out, resize_list, labels_list = fusion(predicted_list, e_labels_specific)
        clc_loss = 0.0 #classification loss
        smooth_loss = 0.0 #smooth loss
        
        for p, l in zip(resize_list, labels_list):
            p_classes = p.squeeze(0).transpose(1,0)
            ce_loss = criterion(p_classes.squeeze(), l)
            sm_loss = torch.mean(torch.clamp(criterion2(F.log_softmax(p_classes[1:, :], dim=1), F.log_softmax(p_classes.detach()[:-1, :], dim=1)), min=0, max=16))
            #sm_loss is smooth because it is the difference between the log of the softmax of the next frame 
            # and the log of the softmax of the current frame using MSE loss (criterion2)
            clc_loss += ce_loss 
            smooth_loss += sm_loss

        clc_loss = clc_loss / (exp_kwargs["mstcn_stages"] * 1.0)
        smooth_loss = smooth_loss / (exp_kwargs["mstcn_stages"] * 1.0)

        _, preds = torch.max(resize_list[0].squeeze().transpose(1, 0).data, 1)
        train_p_classes = torch.softmax((resize_list[0].squeeze().transpose(1, 0)), dim=1)
        train_p_classes_positive = train_p_classes[:, 1]

        loss = clc_loss + exp_kwargs["lambda"] * smooth_loss #weighted sum of classification loss and smooth loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Record probabilities if error_type is global
        if exp_kwargs['error_type'] == "global":
            for j in range(len(train_p_classes_positive)):
                train_all_probs.append(float(train_p_classes_positive.data[j]))
        
            for index in range(len(preds)):
                train_all_preds.append(int(preds.data[index]))

            for index in range(len(e_labels_specific)):
                train_all_labels.append(int(e_labels_specific.data[index]))

            for index in range(len(e_labels_specific)):
                train_all_subjects.append(subject)

        #Record binary predictions if error_type is all_errors
        if exp_kwargs['error_type'] == "all_errors":
            
            for index in range(len(preds)):

                pred = preds.data[index]
                label = e_labels_specific.data[index]
                train_all_preds.append(int(pred)) #No error class is the first class (0). Therefore, we are recording the binary prediction (0 or 1) for no error.
                train_all_labels.append(int(label)) #No error class is the first class (0). Therefore, we are recording the binary label (0 or 1) for no error.
                
                if label.sum() > 0:
                    train_all_labels_binary.append(1) #If the label is not zero, it means there is an error
                else:
                    train_all_labels_binary.append(0)
                
                if pred.sum() > 0:
                    train_all_preds_binary.append(1)
                else:
                    train_all_preds_binary.append(0)
        

        train_loss += loss.data.item()

    if scheduler is not None:
        scheduler.step()

    train_average_loss = float(train_loss) / len(train_dataloader)
    
    #Binary metrics for global error prediction
    if exp_kwargs['error_type'] == "global":
        train_jaccard = jaccard_score(train_all_labels, train_all_preds, average='binary', pos_label=1)
        train_f1 = f1_score(train_all_labels, train_all_preds, average='binary', pos_label=1)
        train_f1_weighted = f1_score(train_all_labels, train_all_preds, average='weighted')
        train_acc = accuracy_score(train_all_labels, train_all_preds)
        train_cm = confusion_matrix(train_all_labels, train_all_preds)

        if exp_kwargs['return_train_preds']:
            return (train_average_loss, train_f1, train_f1_weighted, train_acc, 
                    train_jaccard, train_cm, train_all_probs, train_all_preds, train_all_labels, train_all_subjects)
        else:
            return train_average_loss, train_f1, train_f1_weighted, train_acc, train_jaccard, train_cm
        
    else: #Specific error prediction (6 classes) --> use weighted metrics
        train_f1_binary = f1_score(train_all_labels_binary, train_all_preds_binary, average='binary', pos_label=1)
        train_jaccard_binary = jaccard_score(train_all_labels_binary, train_all_preds_binary, average='binary', pos_label=1)
        train_accuracy_binary = accuracy_score(train_all_labels_binary, train_all_preds_binary)
        train_cm_binary = confusion_matrix(train_all_labels_binary, train_all_preds_binary)

        train_f1 = f1_score(train_all_labels, train_all_preds, average='macro')
        train_jaccard = jaccard_score(train_all_labels, train_all_preds, average='macro')
        train_accuracy = accuracy_score(train_all_labels, train_all_preds)
        train_cm = confusion_matrix(train_all_labels, train_all_preds)
        
        if exp_kwargs['return_train_preds']:
            return (train_average_loss, train_f1_binary, train_f1, train_accuracy_binary, 
                    train_accuracy, train_jaccard_binary, train_jaccard, 
                    train_cm_binary, train_cm, 
                    train_all_probs, train_all_preds, train_all_labels,
                    train_all_labels_binary, train_all_preds_binary)
        else:
            return train_average_loss, train_f1_binary, train_f1, train_accuracy_binary, train_accuracy, train_jaccard_binary, train_jaccard, train_cm_binary, train_cm

def validate_single_epoch_COG(model: torch.nn.Module,   
                            feature_extractor: torch.nn.Module,
                            test_dataloader: DataLoader,
                            criterion: torch.nn.Module,
                            criterion2: torch.nn.Module,
                            device: torch.device,
                            exp_kwargs: dict) -> tuple:
    
    """ Validate COG for a single epoch.

    Args:
        model (torch.nn.Module): The COG model to validate.
        feature_extractor (torch.nn.Module): The feature extractor to use.
        test_dataloader (DataLoader): The dataloader for the test set.
        criterion (torch.nn.Module): The loss function (CrossEntropyLoss).
        criterion2 (torch.nn.Module): The smooth loss function (MSE)
        device (torch.device): The device to validate on.
        exp_kwargs (dict): Additional experiment parameters.

    Returns:
        tuple: A tuple containing the average loss, F1 score, accuracy, and Jaccard index for the test set.

    """

    if exp_kwargs['data_type'] != "kinematics":
        model.eval()
        feature_extractor.eval()
    
    else:
        model.eval()

    test_all_probs = []
    test_all_preds = []
    test_all_preds_binary = []
    test_all_labels = []
    test_all_labels_binary = []
    test_all_labels_specific = []
    test_all_gest_labels = []
    test_all_subjects = []
    test_start_time = time.time()
    val_batch_size = 1
    test_loss = 0.0
    total_time = 0.0
    
    with torch.no_grad():
        
        for i, batch in enumerate(tqdm.tqdm(test_dataloader, 
                                        total=len(test_dataloader),
                                        )):
            
            #images, kinematics, g_labels, e_labels, task, trial, subject = batch
            images, kinematics, g_labels, e_labels, subject, skill_level = batch
            g_labels = g_labels.to(device).float()
            g_labels = g_labels.squeeze(0)  #Remove the extra dimension for frame classification
            e_labels_specific = define_error_labels(e_labels = e_labels, exp_kwargs=exp_kwargs)
            e_labels_specific = e_labels_specific.to(device).float()

            if exp_kwargs['error_type'] == "global":
                e_labels_specific = e_labels_specific.view(-1,)
            
            else: #specific error prediction (6 classes for CE loss)
                e_labels_specific = torch.argmax(e_labels_specific, dim=2)
                e_labels_specific = e_labels_specific.view(-1,)  #Convert to class labels (0-5) for CE loss
                e_labels_specific = e_labels_specific.view(-1,)  #Ensure e_labels is of shape (n_frames,)
            
            inputs = define_inputs(images=images,
                                kinematics=kinematics,
                                feature_extractor=feature_extractor,
                                exp_kwargs=exp_kwargs,
                                device=device) #inputs should be of size (batch_size, features, time_steps)
            
            start_time = time.time() 
            predicted_list, feature_list = model.forward(inputs)
            end_time = time.time()

            all_out, resize_list, labels_list = fusion(predicted_list, e_labels_specific)
            
            test_clc_loss = 0.0
            test_smooth_loss = 0.0
            for p, l in zip(resize_list, labels_list):
                p_classes = p.squeeze(0).transpose(1,0)
                ce_loss = criterion(p_classes.squeeze(), l)
                sm_loss = torch.mean(torch.clamp(criterion2(F.log_softmax(p_classes[1:, :], dim=1), F.log_softmax(p_classes.detach()[:-1, :], dim=1)), min=0, max=16))
                test_clc_loss += ce_loss
                test_smooth_loss += sm_loss

            #Average the losses across the stages
            test_clc_loss = test_clc_loss / (exp_kwargs["mstcn_stages"] * 1.0)
            test_smooth_loss = test_smooth_loss / (exp_kwargs["mstcn_stages"] * 1.0)
            test_loss += test_clc_loss + exp_kwargs["lambda"] * test_smooth_loss

            #Compute predictions and scores
            _, preds = torch.max(predicted_list[0].squeeze().transpose(1, 0).data, 1)
            p_classes = torch.softmax((predicted_list[0].squeeze().transpose(1, 0)), dim=1)
            p_classes_positive = p_classes[:, 1]

            #Record probabilities and predictions
            for j in range(len(g_labels)):
                    test_all_gest_labels.append(int(g_labels.data[j]))
                    test_all_subjects.append(subject)

            if exp_kwargs['error_type'] == "global":
                for j in range(len(p_classes_positive)):
                    test_all_probs.append(float(p_classes_positive.data[j]))
                for j in range(len(preds)):
                    test_all_preds.append(int(preds.data[j]))
                for j in range(len(e_labels)):
                    test_all_labels.append(e_labels[j].tolist())
                for j in range(len(e_labels_specific)):
                    test_all_labels_specific.append(int(e_labels_specific.data[j]))
                

            else: #Specific error prediction (6 classes) --> use binary predictions
                for j in range(len(preds)):
                    pred = preds.data[j]
                    label = e_labels_specific.data[j]
                    test_all_probs.append(p_classes.data[j].tolist())
                    test_all_preds.append(int(pred))
                    test_all_labels.append(int(label))

                    if label.sum() > 0:
                        test_all_labels_binary.append(1)
                    else:
                        test_all_labels_binary.append(0)
                    if pred.sum() > 0:
                        test_all_preds_binary.append(1)
                    else:
                        test_all_preds_binary.append(0)    
            
            total_time += (end_time - start_time)

    test_average_loss = float(test_loss) / len(test_dataloader)
    inference_rate = (total_time / images.shape[1]) * 1000  #Convert to ms per frame

    if exp_kwargs['error_type'] == "global":
        
        test_jaccard = jaccard_score(test_all_labels_specific, test_all_preds, average='binary', pos_label=1)
        test_f1 = f1_score(test_all_labels_specific, test_all_preds, average='binary', pos_label=1)
        test_f1_weighted = f1_score(test_all_labels_specific, test_all_preds, average='weighted')
        test_acc = accuracy_score(test_all_labels_specific, test_all_preds)
        test_cm = confusion_matrix(test_all_labels_specific, test_all_preds)
        
        return test_average_loss, test_f1, test_f1_weighted, test_acc, test_jaccard, test_cm, inference_rate, test_all_preds, test_all_probs, test_all_labels, test_all_labels_specific, test_all_gest_labels, test_all_subjects

    else: #Specific error prediction (6 classes) --> use binary metrics
        test_f1_binary = f1_score(test_all_labels_binary, test_all_preds_binary, average='binary', pos_label=1)
        test_jaccard_binary = jaccard_score(test_all_labels_binary, test_all_preds_binary, average='binary', pos_label=1)
        test_accuracy_binary = accuracy_score(test_all_labels_binary, test_all_preds_binary)
        test_cm_binary = confusion_matrix(test_all_labels_binary, test_all_preds_binary)
        test_f1 = f1_score(test_all_labels, test_all_preds, average='macro')
        test_jaccard = jaccard_score(test_all_labels, test_all_preds, average='macro')
        test_accuracy = accuracy_score(test_all_labels, test_all_preds)
        test_cm = confusion_matrix(test_all_labels, test_all_preds) 

        return test_average_loss, test_f1_binary, test_f1, test_accuracy_binary, test_accuracy, test_jaccard_binary, test_jaccard, \
            test_cm_binary, test_cm, inference_rate, test_all_preds, test_all_preds_binary, test_all_probs, test_all_labels, test_all_labels_binary, \
            test_all_gest_labels, test_all_subjects
    

def train_single_epoch_COG_Sequential(model: torch.nn.Module,
                           feature_extractor: torch.nn.Module,
                       train_dataloader: DataLoader,
                       criterion: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler._LRScheduler,   
                       binary_mask: np.ndarray, 
                       binary_subjects: np.ndarray,
                       device: torch.device,
                       exp_kwargs: dict,
                       criterion2: torch.nn.Module = torch.nn.MSELoss()) -> tuple:

    """ Train error-specific COG (sequential to binary model) for a single epoch.

    Args:
        model (torch.nn.Module): The COG model to train.
        feature_extractor (torch.nn.Module): The feature extractor to use.
        train_dataloader (DataLoader): The dataloader for the training set.
        criterion (torch.nn.Module): The loss function (CrossEntropyLoss).  
        optimizer (torch.optim.Optimizer): The optimizer to use.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        binary_mask (np.ndarray): The binary predictions from the previous model, which will act as a mask in the loss function.
        binary_subjects (np.ndarray): The subjects corresponding to the binary predictions.
        device (torch.device): The device to train on.
        exp_kwargs (dict): Additional experiment parameters.
    Returns:
        tuple: A tuple containing the average loss, F1 score, accuracy, and Jaccard index for the training set.

    """


    if exp_kwargs['data_type'] != "kinematics":
        feature_extractor.train()
        model.train()
    
    else:
        model.train()
    
    train_preds_all = []
    train_labels_all = []
    train_preds_error_specific = []
    train_labels_error_specific = []
    train_loss = 0.0

    for i, batch in enumerate(tqdm.tqdm(train_dataloader, 
                                        total=len(train_dataloader),
                                        )):
        
        #images, kinematics, g_labels, e_labels, task, trial, subject = batch
        images, kinematics, g_labels, e_labels, subject, skill_level = batch
        g_labels = g_labels.to(device).float()
        e_labels_specific = define_error_labels(e_labels = e_labels, exp_kwargs=exp_kwargs)
        e_labels_specific = e_labels_specific.to(device).float()

        #Define labels
        e_labels_specific = torch.argmax(e_labels_specific, dim=2)  #Convert to class labels (0-5) for CE loss
        e_labels_specific = e_labels_specific.view(-1,)

        #print(f"Subject: {subject}, e_labels_specific shape: {e_labels_specific.shape}, e_labels_specific unique values: {torch.unique(e_labels_specific)}")  #Debugging line to check the labels

        #Define inputs
        inputs = define_inputs(images=images,
                               kinematics=kinematics,
                               feature_extractor=feature_extractor,
                               exp_kwargs=exp_kwargs,
                               device=device) #inputs should be of size (batch_size, features, time_steps)
        
        #Extract the part of the mask corresponding to the current subject
        subject_indices = np.where(binary_subjects == subject)[0]
        subject_binary_mask = binary_mask[subject_indices]
        subject_binary_mask = torch.tensor(subject_binary_mask, dtype=torch.float32).to(device)

        predicted_list, feature_list = model.forward(inputs)
        all_out, resize_list, labels_list = fusion(predicted_list, e_labels_specific)
        clc_loss = 0.0 #classification loss
        smooth_loss = 0.0 #smooth loss

        #Create label mask, we don't want to compute loss for frames which are not errors but were predicted as errors, as they will never be correct.
        label_mask = (e_labels_specific != 0).float()  #Assuming 0 is the no error class
        label_mask = label_mask.to(device)  #Move to the same device as the model

        if exp_kwargs['use_true_binary_labels_train']:
            #Check all positions are equal
            assert subject_binary_mask.sum() == label_mask.sum(), \
                f"Subject binary mask sum: {subject_binary_mask.sum()}, label mask sum: {label_mask.sum()}. " \
                f"Subject: {subject}, e_labels_specific: {e_labels_specific}, " \
                f"subject_binary_mask: {subject_binary_mask}, label_mask: {label_mask}"
            
            final_mask = subject_binary_mask  #Use the subject mask directly if we are using true binary labels
            final_mask = (final_mask > 0).float()  #Convert to float mask
            
        else:
            #Apply the subject mask to the label mask (AND operation)
            final_mask = label_mask * subject_binary_mask
            final_mask = (final_mask > 0).float() 

        #As the model will only output 5 classes (0-5), we subtract 1 from the labels to match the output shape.
        e_labels_specific = e_labels_specific - 1  #Now the labels are in the range [-1, 4], as the no error class (0) is not included in the output of the model.

        #Modify the log probabilities at the final_mask positions
        final_mask_copy = copy.deepcopy(final_mask)  #Make a copy of the final mask to modify it
        mask_positions = (final_mask_copy > 0).nonzero(as_tuple=True)[0]  #Get the positions where the mask is 1 (i.e., where we want to compute the smooth loss)
        
        for p, l in zip(resize_list, labels_list):

            #a. CE loss masked
            p_classes = p.squeeze(0).transpose(1,0)    
            ce_loss = criterion(p_classes.squeeze(), l)
            ce_loss = ce_loss * final_mask  #Apply the final mask to the classification loss
            ce_loss = ce_loss.sum() / (final_mask.sum() + 1e-6) #Average the loss over the masked elements

            
            #b. #sm_loss is the difference between the log of the softmax of the next frame and the log of the softmax of the current frame using MSE loss (criterion2).
            #Originally: sm_loss = torch.mean(torch.clamp(criterion2(F.log_softmax(p_classes[1:, :], dim=1), F.log_softmax(p_classes.detach()[:-1, :], dim=1)), min=0, max=16))
            #Now, we apply the subject mask to the smooth loss as well
            
            #Compute log-softmax predictions
            log_p = F.log_softmax(p_classes[1:, :], dim=1)
            log_p_prev = F.log_softmax(p_classes[:-1, :].detach(), dim=1)

            #Compute MSE loss per element
            sm_loss = (log_p - log_p_prev) ** 2  # shape: [T-1, C]
            sm_loss = torch.clamp(sm_loss, min=0, max=16)  # clip if needed

            #Apply mask: subject_mask should be shape [T-1] or [T-1, 1]
            #If needed, expand it to match loss shape
            if final_mask.dim() == 1:
                final_mask = final_mask.unsqueeze(1)  # [T-1, 1]

            final_mask_smooth = final_mask[1:] #[T-1, 1] to match the shape of sm_loss
            try:
                sm_loss = sm_loss * final_mask_smooth # mask out irrelevant timesteps

            except: #in last stage, p_classes is of shape [17, 6] but final_mask_smooth is of shape [n_frames, 1], so don't apply mask
                sm_loss = sm_loss  

            # Reduce (mean over valid elements only)
            sm_loss = sm_loss.sum() / (final_mask_smooth.sum() * sm_loss.size(1) + 1e-6)  # normalize over all masked entries
            
            """
            #b. Smooth loss. To compute the smooth loss, we modify p_classes such that, at the final_mask positions,
            #the probability of class 0 (no error) is set to 1, and the probabilities of other classes are set to 0.
            #Therefore, we need to extend the number of classes to 6 (0-5) and set the probabilities accordingly.
            #If needed, expand it to match loss shape

            log_p = F.log_softmax(p_classes[1:, :], dim=1)
            log_p_prev = F.log_softmax(p_classes[:-1, :].detach(), dim=1)

            if p_classes.shape[0] < 20:
                sm_loss = torch.mean(torch.clamp(criterion2(log_p, log_p_prev), min=0, max=16))  #Compute the smooth loss using MSE loss (criterion2)
            
            else:
                #Extend the number of classes to 6 (0-5) by adding a column of zeros for the no error class
                log_p = torch.cat((torch.zeros(log_p.size(0), 1).to(device), log_p), dim=1)  # shape: [T-1, 6]
                log_p_prev = torch.cat((torch.zeros(log_p_prev.size(0), 1).to(device), log_p_prev), dim=1)  # shape: [T-1, 6]
                
                #Assuming close to certainty for no error class at masked positions, no error class will have log probability close to 0, and other classes will have log probabilities close to -inf.
                log_p[mask_positions[1:], 0] = 1e-3  #Set the log probability of the no error class to a small value (close to 0) at the masked positions
                log_p_prev[mask_positions[:-1], 0] = 1e-3
                log_p[mask_positions[1:], 1:] = -1e3  #Set the log probabilities of the other classes to -inf at the masked positions
                log_p_prev[mask_positions[:-1], 1:] = -1e3

                #In the other positions, we keep the log probabilities as they are for the rest of the classes. However, we are certain that the no error class has not happened at these positions, so we set the log probability of the no error class to -inf.
                log_p[~mask_positions[1:], 0] = -1e3  #Set the log probability of the no error class to -inf at the unmasked positions
                log_p_prev[~mask_positions[:-1], 0] = -1e3

                sm_loss = torch.mean(torch.clamp(criterion2(log_p, log_p_prev), min=0, max=16))  #Compute the smooth loss using MSE loss (criterion2)
            """
            clc_loss += ce_loss 
            smooth_loss += sm_loss

        clc_loss = clc_loss / (exp_kwargs["mstcn_stages"] * 1.0)
        smooth_loss = smooth_loss / (exp_kwargs["mstcn_stages"] * 1.0)

        _, preds = torch.max(resize_list[0].squeeze().transpose(1, 0).data, 1)
        loss = clc_loss + exp_kwargs["lambda"] * smooth_loss #weighted sum of classification loss and smooth loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Record:
        #a. Error-specific probabilities and predictions (i.e., OOV-MA-NP- etc.)
        #b. All error predictions (i.e., including no error class) 
        for index in range(len(preds)):

            #i. Predictions
            pred = preds.data[index] + 1 #Add 1 to the prediction to match the label range (0-5) as the model outputs 5 classes (0-4) and we want to include no error class (0).
            #For all frames that were previously predicted as non-errors, we change that prediction to 0 (no error).
            if subject_binary_mask[index] == 0:  #If the subject mask is 0, it means this frame was not predicted as an error
                pred = 0  #Change the prediction to no error (0)
            
            #ii. Labels
            label = e_labels_specific.data[index] + 1 #Add 1 to the label to match the prediction range (0-5) as the model outputs 5 classes (0-4) and we want to include no error class (0).

            train_preds_all.append(int(pred)) 
            train_labels_all.append(int(label))  #Record the label for the frame

            #iii. Error-specific predictions and labels, i.e., metrics for unmasked frames (true errors that were predicted as errors in the binary model,
            #and are now being evaluated for specific error prediction)
            if final_mask[index] > 0:
                if label == 0:
                    print("AI DIOS MÍO, subject: {}, index: {}, label: {}, pred: {}".format(subject, index, label, pred))  #Debugging line to check the labels and predictions
                train_preds_error_specific.append(int(pred))
                train_labels_error_specific.append(int(label))

        train_loss += loss.data.item()

    if scheduler is not None:
        scheduler.step()

    train_average_loss = float(train_loss) / len(train_dataloader)
    
    #All error prediction (6 classes) --> use weighted metrics
    train_f1_all = f1_score(train_labels_all, train_preds_all, average='macro')
    train_jaccard_all = jaccard_score(train_labels_all, train_preds_all, average='macro')
    train_accuracy_all = accuracy_score(train_labels_all, train_preds_all)
    train_cm_all = confusion_matrix(train_labels_all, train_preds_all)

    #Only Error-specific metrics (5 classes) --> use weighted metrics
    train_f1_error_specific = f1_score(train_labels_error_specific, train_preds_error_specific, average='macro')
    train_f1_error_specific_weighted = f1_score(train_labels_error_specific, train_preds_error_specific, average='weighted')
    train_jaccard_error_specific = jaccard_score(train_labels_error_specific, train_preds_error_specific, average='macro')
    train_jaccard_error_specific_weighted = jaccard_score(train_labels_error_specific, train_preds_error_specific, average='weighted')
    train_accuracy_error_specific = accuracy_score(train_labels_error_specific, train_preds_error_specific)
    train_cm_error_specific = confusion_matrix(train_labels_error_specific, train_preds_error_specific)
    
    return train_average_loss, train_f1_all, train_f1_error_specific, train_f1_error_specific_weighted, train_accuracy_all, train_accuracy_error_specific, \
        train_jaccard_all, train_jaccard_error_specific, train_jaccard_error_specific_weighted, train_cm_all, train_cm_error_specific


def validate_single_epoch_COG_Sequential(model: torch.nn.Module,
                            feature_extractor: torch.nn.Module,
                            test_dataloader: DataLoader,
                            criterion: torch.nn.Module,
                            device: torch.device,
                            binary_mask: np.ndarray,
                            binary_subjects: np.ndarray,
                            exp_kwargs: dict,
                            criterion2: torch.nn.Module=torch.nn.MSELoss()) -> tuple:

    """ Validate error-specific COG (sequential to binary model) for a single epoch.
    Args:
        model (torch.nn.Module): The COG model to validate.
        feature_extractor (torch.nn.Module): The feature extractor to use.
        test_dataloader (DataLoader): The dataloader for the test set.
        criterion (torch.nn.Module): The loss function (CrossEntropyLoss).
        binary_mask (np.ndarray): The binary predictions from the previous model, which will act as a mask in the loss function.
        binary_subjects (np.ndarray): The subjects corresponding to the binary predictions.
        device (torch.device): The device to validate on.
        exp_kwargs (dict): Additional experiment parameters.

    Returns:
        tuple: A tuple containing the average loss, F1 score, accuracy, and Jaccard index for the test set.

    """
    model.eval()
    feature_extractor.eval()

    test_preds_all = []
    test_labels_all = []
    test_preds_error_specific = []
    test_labels_error_specific = []
    test_probs_error_specific = []
    test_gest_labels = []
    test_subjects = []
    test_loss = 0.0
    total_time = 0.0

    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(test_dataloader, 
                                        total=len(test_dataloader),
                                        )):
            
            #images, kinematics, g_labels, e_labels, task, trial, subject = batch
            images, kinematics, g_labels, e_labels, subject, skill_level = batch
            g_labels = g_labels.to(device).float()
            g_labels = g_labels.squeeze(0)  #Remove the extra dimension for frame classification
            e_labels_specific = define_error_labels(e_labels = e_labels, exp_kwargs=exp_kwargs)
            e_labels_specific = e_labels_specific.to(device).float()    

            #Define labels
            e_labels_specific = torch.argmax(e_labels_specific, dim=2)  #Convert to class labels (0-5) for CE loss
            e_labels_specific = e_labels_specific.view(-1,)

            #Define inputs
            inputs = define_inputs(images=images,
                                kinematics=kinematics,
                                feature_extractor=feature_extractor,
                                exp_kwargs=exp_kwargs,
                                device=device)
            
            #Extract the part of the mask corresponding to the current subject
            subject_indices = np.where(binary_subjects == subject)[0]
            subject_mask = binary_mask[subject_indices]
            subject_mask = torch.tensor(subject_mask, dtype=torch.float32).to(device)   

            start_time = time.time() 
            predicted_list, feature_list = model.forward(inputs)
            end_time = time.time()
            all_out, resize_list, labels_list = fusion(predicted_list, e_labels_specific)
            test_clc_loss = 0.0 #classification loss
            test_smooth_loss = 0.0 #smooth loss         

            #Create label mask, we don't want to compute loss for frames which are not errors but were predicted as errors, as they will never be correct.
            label_mask = (e_labels_specific != 0).float()  #Assuming 0 is the no error class
            label_mask = label_mask.to(device)  #Move to the same device as the model

            #Apply the subject mask to the label mask
            final_mask = label_mask * subject_mask  #This will zero out the loss for frames that are not errors and not in the subject mask

            #As the model will only output 5 classes (0-5), we subtract 1 from the labels to match the output shape.
            e_labels_specific = e_labels_specific - 1  #Now the labels are in the range [-1, 4], as the no error class (0) is not included in the output of the model.

            #print(f"Final mask shape: {final_mask.shape}, e_labels_specific shape: {e_labels_specific.shape}, subject_mask shape: {subject_mask.shape}")
            #print(f"Unique labels in e_labels_specific: {torch.unique(e_labels_specific)}")
            
            for p, l in zip(resize_list, labels_list):
                
                #a. CE loss masked
                p_classes = p.squeeze(0).transpose(1,0)        
                ce_loss = criterion(p_classes.squeeze(), l)
                ce_loss = ce_loss * final_mask
                ce_loss = ce_loss.sum() / (final_mask.sum() + 1e-6) #Average the loss over the masked elements


                #b. #sm_loss is the difference between the log of the softmax of the next frame and the log of the softmax of the current frame using MSE loss (criterion2).
                #Originally: sm_loss = torch.mean(torch.clamp(criterion2(F.log_softmax(p_classes[1:, :], dim=1), F.log_softmax(p_classes.detach()[:-1, :], dim=1)), min=0, max=16))
                #Now, we apply the subject mask to the smooth loss as well              
                #Compute log-softmax predictions
                log_p = F.log_softmax(p_classes[1:, :], dim=1)
                log_p_prev = F.log_softmax(p_classes[:-1, :].detach(), dim=1)

                #Compute MSE loss per element
                sm_loss = (log_p - log_p_prev) ** 2  # shape:
                sm_loss = torch.clamp(sm_loss, min=0, max=16)  # clip if needed 
                # Apply mask: subject_mask should be shape [T-1] or [T-1, 1]
                #If needed, expand it to match loss shape
                if final_mask.dim() == 1:
                    final_mask = final_mask.unsqueeze(1)
                final_mask_smooth = final_mask[1:] #[T-1, 1] to match the shape of sm_loss
                try:
                    sm_loss = sm_loss * final_mask_smooth # mask out irrelevant timesteps
                except: #in last stage, p_classes is of shape [17, 6] but final_mask_smooth is of shape [n_frames, 1], so don't apply mask
                    sm_loss = sm_loss
                sm_loss = sm_loss.sum() / (final_mask_smooth.sum() * sm_loss.size(1) + 1e-6)
                """
                #b. Smooth loss. To compute the smooth loss, we modify p_classes such that
                #at the final_mask positions, the probability of class 0 (no error) is set to 1, and the probabilities of other classes are set to 0.
                #Therefore, we need to extend the number of classes to 6 (0-5) and set the probabilities accordingly.
                log_p = F.log_softmax(p_classes[1:, :], dim=1)
                log_p_prev = F.log_softmax(p_classes[:-1, :].detach(), dim=1)

                #Extend the number of classes to 6 (0-5) by adding a column of zeros for the no error class
                log_p = torch.cat((torch.zeros(log_p.size(0), 1).to(device), log_p), dim=1)  # shape: [T-1, 6]
                log_p_prev = torch.cat((torch.zeros(log_p_prev.size(0), 1).to(device), log_p_prev), dim=1)  # shape: [T-1, 6]

                #Modify the log probabilities at the final_mask positions
                final_mask_copy = copy.deepcopy(final_mask)  #Make a copy of the final mask to modify it
                mask_positions = (final_mask_copy[1:] > 0).nonzero(as_tuple=True)[0]
                
                #Assuming close to certainty for no error class at masked positions, no error class will have log probability close to 0, and other classes will have log probabilities close to -inf.
                log_p[mask_positions, 0] = 1e-4
                log_p_prev[mask_positions, 0] = 1e-4
                log_p[mask_positions, 1:] = -1e4 
                log_p_prev[mask_positions, 1:] = -1e4

                sm_loss = torch.mean(torch.clamp(criterion2(log_p, log_p_prev), min=0, max=16))  #Compute the smooth loss using MSE loss (criterion2)
                """
                test_clc_loss += ce_loss
                test_smooth_loss += sm_loss

            #Average the losses across the stages
            test_clc_loss = test_clc_loss / (exp_kwargs["mstcn_stages"] * 1.0)
            test_smooth_loss = test_smooth_loss / (exp_kwargs["mstcn_stages"] * 1.0)
            test_loss += test_clc_loss + exp_kwargs["lambda"] * test_smooth_loss

            #Compute predictions and scores
            _, preds = torch.max(predicted_list[0].squeeze().transpose(1, 0).data, 1)
            #print(f"Unique predictions: {torch.unique(preds)}")
            p_classes = torch.softmax((predicted_list[0].squeeze().transpose(1, 0)), dim=1)

            #Record probabilities and predictions
            for j in range(len(g_labels)):
                test_gest_labels.append(int(g_labels.data[j]))
                test_subjects.append(subject)


            #Record error-specific predictions and labels, i.e., metrics for unmasked frames (true errors that were predicted as errors in the binary model,
            #and are now being evaluated for specific error prediction)
            for index in range(len(preds)):
                pred = preds.data[index] + 1  #Add 1 to the prediction to match the label range (0-5) as the model outputs 5 classes (0-4) and we want to include no error class (0).
                label = e_labels_specific.data[index] + 1  #Add 1 to the label to match the prediction range (0-5) as the model outputs 5 classes (0-4) and we want to include no error class (0).

                #If the subject mask is 0, it means this frame was not predicted as an error
                if subject_mask[index] == 0:  
                    pred = 0

                test_preds_all.append(int(pred))
                test_labels_all.append(int(label))

                if final_mask[index] > 0:  #If the frame is an error and was predicted as an error in the binary model
                    test_preds_error_specific.append(int(pred))
                    test_labels_error_specific.append(int(label))
                    test_probs_error_specific.append(p_classes.data[index].tolist())  #Record the probabilities for the error-specific predictions

            total_time += (time.time() - start_time) 

    test_average_loss = float(test_loss) / len(test_dataloader)
    inference_rate = (total_time / images.shape[1]) * 1000  #Convert to ms per frame

    #Specific error prediction (6 classes) --> use weighted metrics
    test_f1_all = f1_score(test_labels_all, test_preds_all, average='macro')
    test_jaccard_all = jaccard_score(test_labels_all, test_preds_all, average='macro')
    test_accuracy_all = accuracy_score(test_labels_all, test_preds_all)
    test_cm_all = confusion_matrix(test_labels_all, test_preds_all) 

    test_f1_error_specific = f1_score(test_labels_error_specific, test_preds_error_specific, average='macro')
    test_f1_error_specific_weighted = f1_score(test_labels_error_specific, test_preds_error_specific, average='weighted')
    test_jaccard_error_specific = jaccard_score(test_labels_error_specific, test_preds_error_specific, average='macro')
    test_jaccard_error_specific_weighted = jaccard_score(test_labels_error_specific, test_preds_error_specific, average='weighted')
    test_accuracy_error_specific = accuracy_score(test_labels_error_specific, test_preds_error_specific)
    test_cm_error_specific = confusion_matrix(test_labels_error_specific, test_preds_error_specific)

    return test_average_loss, test_f1_all, test_f1_error_specific, test_f1_error_specific_weighted, test_accuracy_all, test_accuracy_error_specific, \
        test_jaccard_all, test_jaccard_error_specific, test_jaccard_error_specific_weighted, test_cm_all, test_cm_error_specific, \
        inference_rate, test_preds_all, test_preds_error_specific, test_probs_error_specific, \
        test_labels_all, test_labels_error_specific, test_gest_labels, test_subjects



def fusion(predicted_list,labels):
    all_out_list = []
    resize_out_list = []
    labels_list = []
    all_out = 0

    for out in predicted_list:
        resize_out =F.interpolate(out,size=labels.size(0),mode='nearest')
        resize_out_list.append(resize_out)
        
        if out.size(2)==labels.size(0):
            resize_label = labels
            labels_list.append(resize_label.squeeze().long())
        else:
            # resize_label = max_pool(labels_list[-1].float().unsqueeze(0).unsqueeze(0))
            resize_label = F.interpolate(labels.float().unsqueeze(0).unsqueeze(0),size=out.size(2),mode='nearest')
            
            labels_list.append(resize_label.squeeze().long())
        
        all_out_list.append(out)
    return all_out, all_out_list, labels_list



def load_model_mlflow(out: str,
                      exp_kwargs: dict,
                      setting: str,
                      run_id: str) -> tuple:
    
    """
    Load pytorch mlflow models

    Args:
        out (str): The output type (e.g., 'train', 'test').
        exp_kwargs (dict): Additional experiment parameters.
        setting (str): The setting of the experiment (e.g., 'LOSO').
        run_id (str): The mlflow run ID.

    Returns:

    """
    feature_extractor = None
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model_{setting}_{out}")

    if exp_kwargs['data_type'] != 'kinematics':
        feature_extractor = mlflow.pytorch.load_model(f"runs:/{run_id}/feature_extractor_{setting}_{out}")

    return feature_extractor, model


def load_model_local(model_folder:str,
                     out: str,
                     setting: str,
                     exp_kwargs: dict,
                     in_features: int,
                     window_size: int,
                     device:torch.device) -> tuple:
    
    """
    Load pytorch models from local files

    Args:

        model_folder (str): The folder where the models are saved.
        out (str): The output type (e.g., 'train', 'test').
        setting (str): The setting of the experiment (e.g., 'LOSO').
        exp_kwargs (dict): Additional experiment parameters.
    
    Returns:
        tuple: A tuple containing the feature extractor and model.

    """
    if exp_kwargs['model_name'] == "TransSVNet":
        exp_kwargs2 = exp_kwargs.copy()
        exp_kwargs2['model_name'] = "TeCNo"
        model = instantiate_model(exp_kwargs = exp_kwargs2,
                            in_features = in_features,
                            window_size = window_size).to(device)
    
    else:
        model = instantiate_model(exp_kwargs = exp_kwargs,
                            in_features = in_features,
                            window_size = window_size).to(device)

    
    if exp_kwargs['data_type'] != "kinematics" and exp_kwargs['video_dims'] != 2048:
        feature_extractor = FeatureExtractor(input_dim=2048, output_dim=exp_kwargs['video_dims'], hidden_dims=[512, 256]).to(device)

    else:
        feature_extractor = None

    #Load model
    model_path = os.path.join(model_folder, f'best_model_{setting}_{out}.pt')
    best_model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_model['model'])
    model.to(device)
    
    #Load feature extractor if necessary
    if exp_kwargs['data_type'] != 'kinematics':

        if exp_kwargs['video_dims'] != 2048:
            feature_extractor.load_state_dict(best_model['feature_extractor'])
            feature_extractor.to(device) 

    return feature_extractor, model


def load_binary_model_local(model_folder:str,
                       model_name:str,
                       outs:list,
                       exp_kwargs:dict,
                       device:torch.device):

    model_folder = f'models/{exp_kwargs["data_type"]}/{exp_kwargs["frequency"]}Hz/{model_name}/'
    model_dict = {}
    fe_dict = {}
    for out in outs:

        model = LSTM(in_features=58, window_size=10, 
                        hidden_size=128, num_layers=3,
                        n_classes=1).to(device)
        
        feature_extractor = FeatureExtractor(input_dim=2048, output_dim=exp_kwargs['video_dims'], hidden_dims=[512, 256]).to(device)
        
        model_path = os.path.join(model_folder, f'best_model_LOSO_{out}.pt')
        best_model = torch.load(model_path, weights_only=False)

        feature_extractor_state_dict = best_model['feature_extractor']
        model_state_dict = best_model['model']

        feature_extractor.load_state_dict(feature_extractor_state_dict)
        model.load_state_dict(model_state_dict)

        model_dict[out] = model
        fe_dict[out] = feature_extractor

    print(f"Loaded binary models from {model_folder}.")

    return model_dict, fe_dict


def process_all_labels(all_labels: list,
                       exp_kwargs: dict) -> torch.tensor:

    """
    This function processess the all_labels list. Why?
    all_labels should be a tensor of lenght n_windows/n_frames, where each elements is a tensor of shape (5,) -as there are 5 error types-.
    When printing one of the elements, it shows:
    tensor([1., 1., 0., 0., 1.])
    or 
    'tensor([0, 1, 0, 0, 0, 0, 1], dtype=torch.int32)'
    However, this is a string! Meaning element[0] is 't', element[1] is 'e', and so on. This is due to mlflow's serialization of tensors.
    We need to convert the string to a tensor of floats.

    Args:
        all_labels (list): List of labels to process.
    
    Returns:
        torch.tensor: Processed tensor of labels.
    """

    if exp_kwargs['compute_from_str'] == True:
        
        length = len(all_labels)

        #Manually define the positions of the string where the labels are stored. t is at position 0, e at position 1, and so on. An example string is "tensor([1., 1., 0., 0., 1.])"
        if exp_kwargs['dataset_type'] == 'frame':
            error_label_positions = [8, 12, 16, 20, 24]  
            processed_labels = torch.zeros((length, 5), dtype=torch.float32)
        elif exp_kwargs['dataset_type'] == 'window':
            error_label_positions = [8, 11, 14, 17, 20, 23, 26] 
            processed_labels = torch.zeros((length, 7), dtype=torch.float32)

        for i, label in enumerate(all_labels):
            for j, pos in enumerate(error_label_positions):
                processed_labels[i, j] = float(label[pos])

    else:
        processed_labels = []
        for i in range(len(all_labels)):
            processed_labels.append(np.array(all_labels[i]))
        
        processed_labels = np.concatenate(processed_labels, axis=0)

    return processed_labels

def retrieve_results_mlflow(outs: list,
                            exp_kwargs: dict,
                            setting: str,
                            run_id: str) -> tuple:
    

    #Compute avg and std of accuracy, f1 and jaccard across folds as saved in the best model
    LOSO_f1_train, LOSO_f1_test, LOSO_acc_train, LOSO_acc_test, LOSO_jaccard_train, LOSO_jaccard_test = ([] for _ in range(6))
    LOSO_cm_train, LOSO_cm_test = (np.zeros((2, 2)) for _ in range(2))  #Init confusion matrices 

    return_train_preds = exp_kwargs['return_train_preds'] if 'return_train_preds' in exp_kwargs else False
    test_all_preds, test_all_probs, test_all_labels, test_all_labels_specific, test_all_gest_labels, test_all_subjects = ({} for _ in range(6))

    if return_train_preds:
        train_all_preds, train_all_probs, train_all_labels, train_all_subjects = ({} for _ in range(4))

    for out in outs:

        if exp_kwargs['save_local']:
            model_folder =  f'models/{exp_kwargs["model_name"]}/'
            model_path = os.path.join(model_folder, f'{setting}_{out}_model.pth')

            #Model file contains the state dicts along with performance at that epoch. with open does not work
            if os.path.exists(model_path):

                #with open(model_path, 'rb') as f: does not work. try alternative
                best_model = torch.load(model_path, map_location="cpu", weights_only=False)

                LOSO_f1_train.append(best_model['train_f1_fold'])
                LOSO_f1_test.append(best_model['test_f1_fold'])
                LOSO_acc_train.append(best_model['train_acc_fold'])
                LOSO_acc_test.append(best_model['test_acc_fold'])   
                LOSO_jaccard_train.append(best_model['train_jaccard_fold'])
                LOSO_jaccard_test.append(best_model['test_jaccard_fold'])
                LOSO_cm_train += best_model['train_cm_fold']
                LOSO_cm_test += best_model['test_cm_fold']
            
            else:
                print(f"Model file {model_path} does not exist. Skipping...")

        else:
            #Load best_model_dict from mlflow
            dict_path = f"runs:/{run_id}/best_model_{setting}_{out}.json"
            best_model_dict = mlflow.artifacts.load_dict(dict_path)
            
            LOSO_f1_train.append(best_model_dict['train_f1_fold'])
            LOSO_f1_test.append(best_model_dict['test_f1_fold'])
            LOSO_acc_train.append(best_model_dict['train_acc_fold'])
            LOSO_acc_test.append(best_model_dict['test_acc_fold'])
            LOSO_jaccard_train.append(best_model_dict['train_jaccard_fold'])
            LOSO_jaccard_test.append(best_model_dict['test_jaccard_fold'])
            LOSO_cm_train += np.array(best_model_dict['train_cm_fold'])
            LOSO_cm_test += np.array(best_model_dict['test_cm_fold'])

            if return_train_preds:
                try:
                    train_all_preds[out] = best_model_dict['train_all_preds_fold']
                    train_all_probs[out] = best_model_dict['train_all_probs_fold']
                    train_all_labels[out] = best_model_dict['train_all_labels_fold']
                    train_all_subjects[out] = best_model_dict['train_all_subjects_fold']

                except:
                    train_all_preds[out] = best_model_dict['train_all_preds']
                    train_all_probs[out] = best_model_dict['train_all_probs']
                    train_all_labels[out] = best_model_dict['train_all_labels']
                    train_all_subjects[out] = best_model_dict['train_all_subjects']

            if return_train_preds or exp_kwargs['dataset_type'] == 'frame':
                try:
                
                    test_all_preds[out] = best_model_dict['test_all_preds_fold']
                    try:
                        test_all_probs[out] = best_model_dict['test_all_probs_fold']
                        test_all_labels_specific[out] = best_model_dict['test_all_labels_specific_fold']
                    except:
                        test_all_probs[out] = []
                        test_all_labels_specific[out] = []
                    
                    test_all_labels[out] = best_model_dict['test_all_labels_fold']
                    test_all_gest_labels[out] = best_model_dict['test_all_gest_labels_fold']
                    test_all_subjects[out] = best_model_dict['test_all_subjects_fold']
                
                except:
                    test_all_preds[out] = best_model_dict['test_all_preds']
                    test_all_probs[out] = best_model_dict['test_all_probs']
                    test_all_labels[out] = best_model_dict['test_all_labels']
                    test_all_labels_specific[out] = best_model_dict['test_all_labels_specific']
                    test_all_gest_labels[out] = best_model_dict['test_all_gest_labels']
                    test_all_subjects[out] = best_model_dict['test_all_subjects']

                try:
                    test_all_labels[out] = process_all_labels(test_all_labels[out], exp_kwargs=exp_kwargs)
                
                except:
                    pass

    #Change confusion matrices to integer type
    LOSO_cm_train = LOSO_cm_train.astype(int)
    LOSO_cm_test = LOSO_cm_test.astype(int)

    if return_train_preds:

        return (LOSO_f1_train, LOSO_f1_test,
                    LOSO_acc_train, LOSO_acc_test,
                    LOSO_jaccard_train, LOSO_jaccard_test,
                    LOSO_cm_train, LOSO_cm_test,
                    train_all_preds, train_all_probs, train_all_labels, train_all_subjects,
                    test_all_preds, test_all_probs, test_all_labels, test_all_labels_specific, test_all_gest_labels, test_all_subjects)
        
    else:  
        if exp_kwargs['dataset_type'] == 'frame':
            return (LOSO_f1_train, LOSO_f1_test,
                    LOSO_acc_train, LOSO_acc_test,
                    LOSO_jaccard_train, LOSO_jaccard_test,
                    LOSO_cm_train, LOSO_cm_test,
                    test_all_preds, test_all_probs, test_all_labels, test_all_labels_specific, test_all_gest_labels, test_all_subjects)
        else:
            return (LOSO_f1_train, LOSO_f1_test,
                LOSO_acc_train, LOSO_acc_test,
                LOSO_jaccard_train, LOSO_jaccard_test,
                LOSO_cm_train, LOSO_cm_test)
    

def retrieve_results_mlflow_ES(outs: list,
                               exp_kwargs: dict,
                               setting: str,
                               run_id: str) -> tuple:
    
    """ Retrieve results from mlflow for error specific prediction (i.e., exp_kwargs['error_type'] == 'all_errors').
    
    Args:
        outs (list): List of output types (e.g., ['train', 'test']).
        exp_kwargs (dict): Additional experiment parameters.
        setting (str): The setting of the experiment (e.g., 'LOSO').
        run_id (str): The mlflow run ID.

    Returns:
        tuple: A tuple containing the retrieved results.
    """

    #Compute avg and std of accuracy, f1 and jaccard across folds as saved in the best model
    LOSO_f1_train, LOSO_f1_train_binary, LOSO_f1_test, LOSO_f1_test_binary, LOSO_acc_train, LOSO_acc_train_binary, LOSO_acc_test, LOSO_acc_test_binary, \
    LOSO_jaccard_train, LOSO_jaccard_train_binary, LOSO_jaccard_test, LOSO_jaccard_test_binary = ([] for _ in range(12))

    LOSO_cm_train_binary, LOSO_cm_test_binary = (np.zeros((2, 2)) for _ in range(2))  #Init confusion matrices
    LOSO_cm_train, LOSO_cm_test = (np.zeros((6, 6)) for _ in range(2))  #Init confusion matrices for specific error 

    test_all_preds, test_all_preds_binary, test_all_probs, test_all_labels, test_all_labels_binary, \
    test_all_gest_labels, test_all_subjects = ({} for _ in range(7))

    for out in outs:

        dict_path = f"runs:/{run_id}/best_model_{setting}_{out}.json"
        best_model_dict = mlflow.artifacts.load_dict(dict_path)
        
        LOSO_f1_train.append(best_model_dict['train_f1_fold'])
        LOSO_f1_train_binary.append(best_model_dict['train_f1_binary_fold'])
        LOSO_f1_test.append(best_model_dict['test_f1_fold'])
        LOSO_f1_test_binary.append(best_model_dict['test_f1_binary_fold'])
        LOSO_acc_train.append(best_model_dict['train_acc_fold'])
        LOSO_acc_train_binary.append(best_model_dict['train_accuracy_binary_fold'])
        LOSO_acc_test.append(best_model_dict['test_acc_fold'])
        LOSO_acc_test_binary.append(best_model_dict['test_accuracy_binary_fold'])
        LOSO_jaccard_train.append(best_model_dict['train_jaccard_fold'])
        LOSO_jaccard_train_binary.append(best_model_dict['train_jaccard_binary_fold'])
        LOSO_jaccard_test.append(best_model_dict['test_jaccard_fold'])
        LOSO_jaccard_test_binary.append(best_model_dict['test_jaccard_binary_fold'])

        LOSO_cm_train += np.array(best_model_dict['train_cm_fold'])
        LOSO_cm_train_binary += np.array(best_model_dict['train_cm_binary_fold'])
        LOSO_cm_test += np.array(best_model_dict['test_cm_fold'])
        LOSO_cm_test_binary += np.array(best_model_dict['test_cm_binary_fold'])

        if exp_kwargs['dataset_type'] == 'frame':
            try:
                test_all_preds[out] = best_model_dict['test_all_preds_fold']
                test_all_preds_binary[out] = best_model_dict['test_all_preds_binary_fold']
                test_all_probs[out] = best_model_dict['test_all_probs_fold']
                test_all_labels[out] = best_model_dict['test_all_labels_fold']
                test_all_labels_binary[out] = best_model_dict['test_all_labels_binary_fold']
                test_all_gest_labels[out] = best_model_dict['test_all_gest_labels_fold']
                test_all_subjects[out] = best_model_dict['test_all_subjects_fold']

            except:
                test_all_preds[out] = best_model_dict['test_all_preds']
                test_all_preds_binary[out] = best_model_dict['test_all_preds_binary']
                test_all_probs[out] = best_model_dict['test_all_probs']
                test_all_labels[out] = best_model_dict['test_all_labels']
                test_all_labels_binary[out] = best_model_dict['test_all_labels_binary']
                test_all_gest_labels[out] = best_model_dict['test_all_gest_labels']
                test_all_subjects[out] = best_model_dict['test_all_subjects']

    #Change confusion matrices to integer type
    LOSO_cm_train = LOSO_cm_train.astype(int)
    LOSO_cm_train_binary = LOSO_cm_train_binary.astype(int)
    LOSO_cm_test = LOSO_cm_test.astype(int)
    LOSO_cm_test_binary = LOSO_cm_test_binary.astype(int)

    return (LOSO_f1_train, LOSO_f1_train_binary, LOSO_f1_test, LOSO_f1_test_binary,
                LOSO_acc_train, LOSO_acc_train_binary, LOSO_acc_test, LOSO_acc_test_binary,
                LOSO_jaccard_train, LOSO_jaccard_train_binary, LOSO_jaccard_test, LOSO_jaccard_test_binary,
                LOSO_cm_train, LOSO_cm_train_binary, LOSO_cm_test, LOSO_cm_test_binary,
                test_all_preds, test_all_preds_binary, test_all_probs, test_all_labels, test_all_labels_binary,
                test_all_gest_labels, test_all_subjects)
    

def retrieve_results_mlflow_sequential(outs: list,
                                       exp_kwargs: dict,
                                       setting: str,
                                       run_id: str) -> tuple:
    
    """ Retrieve results from mlflow for sequential error specific prediction.
    Args:
        outs (list): List of output types (e.g., ['train', 'test']).
        exp_kwargs (dict): Additional experiment parameters.
        setting (str): The setting of the experiment (e.g., 'LOSO').
        run_id (str): The mlflow run ID.
    
    Returns:
        tuple: A tuple containing the retrieved results.


    This is the saved dict:
    {'feature_extractor': feature_extractor,
                        'model': model,
                        'epoch': epoch + 1,
                        'train_loss': train_average_loss,
                        'test_loss': test_average_loss,
                        'train_f1': train_f1_all,
                        'test_f1': test_f1_all, 
                        'train_f1_specific': train_f1_error_specific,
                        'test_f1_specific': test_f1_error_specific,
                        'test_f1_specific_weighted': test_f1_error_specific_weighted,
                        'train_accuracy': train_accuracy_all,
                        'test_accuracy': test_accuracy_all,
                        'train_accuracy_specific': train_accuracy_error_specific,
                        'test_accuracy_specific': test_accuracy_error_specific,
                        'train_jaccard': train_jaccard_all,
                        'test_jaccard': test_jaccard_all,
                        'train_jaccard_specific': train_jaccard_error_specific,
                        'test_jaccard_specific': test_jaccard_error_specific,
                        'test_jaccard_specific_weighted': test_jaccard_error_specific_weighted,
                        'train_cm': train_cm_all.tolist(),
                        'test_cm': test_cm_all.tolist(),
                        'train_cm_specific': train_cm_error_specific.tolist(),
                        'test_cm_specific': test_cm_error_specific.tolist(),
                        'inference_rate': inference_rate,
                        'test_preds_all': test_preds_all,
                        'test_preds_error_specific': test_preds_error_specific,
                        'test_probs_error_specific': test_probs_error_specific,
                        'test_labels_all': test_labels_all,
                        'test_labels_error_specific': test_labels_error_specific,
                        'test_gest_labels': test_gest_labels,
                        'test_subjects': test_subjects}
    """

    LOSO_f1_train, LOSO_f1_test, LOSO_f1_train_specific, LOSO_f1_test_specific, LOSO_f1_train_specific_weighted, LOSO_f1_test_specific_weighted,\
    LOSO_acc_train, LOSO_acc_test, LOSO_acc_train_specific, LOSO_acc_test_specific, LOSO_jaccard_train, LOSO_jaccard_test, LOSO_jaccard_train_specific, LOSO_jaccard_test_specific, \
    LOSO_jaccard_train_specific_weighted, LOSO_jaccard_test_specific_weighted = ([] for _ in range(16))

    LOSO_cm_train, LOSO_cm_test = (np.zeros((6, 6)) for _ in range(2))  #Init confusion matrices for specific error
    LOSO_cm_train_specific, LOSO_cm_test_specific = (np.zeros((5, 5)) for _ in range(2))  #Init confusion matrices for specific error

    test_all_preds, test_all_preds_specific, test_all_probs, test_all_labels, test_all_labels_specific, \
    test_all_gest_labels, test_all_subjects = ({} for _ in range(7))

    for out in outs:
        
        dict_path = f"runs:/{run_id}/best_model_{setting}_{out}.json"
        best_model_dict = mlflow.artifacts.load_dict(dict_path)

        LOSO_f1_train.append(best_model_dict['train_f1'])
        LOSO_f1_test.append(best_model_dict['test_f1'])
        LOSO_f1_train_specific.append(best_model_dict['train_f1_specific'])
        LOSO_f1_test_specific.append(best_model_dict['test_f1_specific'])
        LOSO_f1_train_specific_weighted.append(best_model_dict['train_f1_specific_weighted'])
        LOSO_f1_test_specific_weighted.append(best_model_dict['test_f1_specific_weighted'])
        LOSO_acc_train.append(best_model_dict['train_accuracy'])
        LOSO_acc_test.append(best_model_dict['test_accuracy'])
        LOSO_acc_train_specific.append(best_model_dict['train_accuracy_specific'])
        LOSO_acc_test_specific.append(best_model_dict['test_accuracy_specific'])
        LOSO_jaccard_train.append(best_model_dict['train_jaccard'])
        LOSO_jaccard_test.append(best_model_dict['test_jaccard'])   
        LOSO_jaccard_train_specific.append(best_model_dict['train_jaccard_specific'])
        LOSO_jaccard_test_specific.append(best_model_dict['test_jaccard_specific'])
        LOSO_jaccard_train_specific_weighted.append(best_model_dict['train_jaccard_specific_weighted'])
        LOSO_jaccard_test_specific_weighted.append(best_model_dict['test_jaccard_specific_weighted'])   

        LOSO_cm_train += np.array(best_model_dict['train_cm'])
        LOSO_cm_test += np.array(best_model_dict['test_cm'])
        LOSO_cm_train_specific += np.array(best_model_dict['train_cm_specific'])
        LOSO_cm_test_specific += np.array(best_model_dict['test_cm_specific'])

        test_all_preds[out] = best_model_dict['test_preds_all']
        test_all_preds_specific[out] = best_model_dict['test_preds_error_specific']
        test_all_probs[out] = best_model_dict['test_probs_error_specific']
        test_all_labels[out] = best_model_dict['test_labels_all']
        test_all_labels_specific[out] = best_model_dict['test_labels_error_specific']
        test_all_gest_labels[out] = best_model_dict['test_gest_labels']
        test_all_subjects[out] = best_model_dict['test_subjects']

    #Change confusion matrices to integer type
    LOSO_cm_train = LOSO_cm_train.astype(int)   
    LOSO_cm_train_specific = LOSO_cm_train_specific.astype(int)
    LOSO_cm_test = LOSO_cm_test.astype(int)
    LOSO_cm_test_specific = LOSO_cm_test_specific.astype(int)

    #Return the results
    return (LOSO_f1_train, LOSO_f1_test, LOSO_f1_train_specific, LOSO_f1_test_specific, 
            LOSO_f1_train_specific_weighted, LOSO_f1_test_specific_weighted,
            LOSO_acc_train, LOSO_acc_test, LOSO_acc_train_specific, LOSO_acc_test_specific,
            LOSO_jaccard_train, LOSO_jaccard_test, LOSO_jaccard_train_specific, LOSO_jaccard_test_specific,
            LOSO_jaccard_train_specific_weighted, LOSO_jaccard_test_specific_weighted,
            LOSO_cm_train, LOSO_cm_test, LOSO_cm_train_specific, LOSO_cm_test_specific,
            test_all_preds, test_all_preds_specific, test_all_probs, test_all_labels, 
            test_all_labels_specific, test_all_gest_labels, test_all_subjects)


def window_predictions(predictions,
                       e_labels,
                       gestures,
                       subjects,
                       window_size=10,
                        stride=6,
                        binary=True):
    
    """

    Function that windows frame level predictions following the same logic as window_data.

    Args:
        predictions: list of predictions of length n_frames
        e_labels: list of error labels of length n_frames
        gestures: list of gestures of length n_frames
        subjects: list of subjects of length n_frames
        window_size: int, size of the window
        stride: int, stride of the window

    Returns:
        predictions_windows: tensor of shape (n_windows, 1)
        e_labels_windows: tensor of shape (n_windows, 1)
        gestures_windows: tensor of shape (n_windows, 1)
        subjects_windows: DataFrame of shape (n_windows, 1) with subject names
    """

    subjects_unique = np.unique(np.array(subjects))
    subject_indices = {subject: np.where(np.array(subjects) == subject)[0] for subject in subjects_unique}

    predictions_windows = []
    e_labels_windows = []
    gestures_windows = []
    subjects_windows = []

    for subject in subjects_unique:
        subject_index = subject_indices[subject]
        n_frames_subject = len(subject_index)

        #Iterate over gestures
        gesture_indices = np.where(np.array(gestures)[subject_index] != 0)[0]
        start_idx = gesture_indices[0]

        while start_idx < n_frames_subject - window_size:    
            #1. Each gesture type must be at least of window_size to create a window.
            end_idx = start_idx + window_size

            #2. A window cannot contain more than two gesture types, as errors are at the gesture level.
            current_gesture = gestures[subject_index[start_idx]]
            end_gesture = gestures[subject_index[end_idx - 1]]

            if end_idx > n_frames_subject or current_gesture != end_gesture:
                start_idx += 1
                continue

            #3. Create window
            #Binarize prediction by computing the average and threhsolding it at 0.5
            predictions_windows.append(np.mean(predictions[subject_index[start_idx:end_idx]]))
            if binary:
                predictions_windows[-1] = 1.0 if predictions_windows[-1] >= 0.5 else 0.0
            else:
                #In specifc error prediction, we need to retrieve the most common error type (6 in total)
                #By having taken the mean, we can round it to the nearest integer to get the most common error type
                predictions_windows[-1] = np.round(predictions_windows[-1])

            e_labels_windows.append(e_labels[subject_index[start_idx]])
            gestures_windows.append(current_gesture)
            subjects_windows.append(subject)

            #5. Move to next window
            start_idx += stride

    #Convert lists to tensors
    predictions_windows = torch.tensor(predictions_windows).reshape(-1, 1)
    e_labels_windows = torch.tensor(e_labels_windows).reshape(-1, 1)
    gestures_windows = torch.tensor(gestures_windows).reshape(-1, 1)
    subjects_windows = pd.DataFrame(subjects_windows, columns=['subject'])
    
    #Return the windows
    return (predictions_windows, 
            e_labels_windows, 
            gestures_windows, 
            subjects_windows)


def frame2window(outs: list,
                 test_all_preds: dict,
                 test_all_labels: dict,
                 test_all_gest_labels: dict,
                 test_all_subjects: dict,
                 window_size: int = 10,
                 stride: int = 6,
                 binary: bool = True) -> tuple:

    """
    Convert frame level predictions to window level predictions.

    Args:
        outs (list): List of output types (e.g., ['train', 'test']).
        test_all_preds (dict): Dictionary of frame level predictions for each output type.
        test_all_labels (dict): Dictionary of frame level labels for each output type.
        test_all_gest_labels (dict): Dictionary of frame level gesture labels for each output type.
        test_all_subjects (dict): Dictionary of frame level subjects for each output type.
        window_size (int): Size of the window.
        stride (int): Stride of the window.

    Returns:
        tuple: A tuple containing the window level predictions, labels, gesture labels, and subjects for each output type.

    """

    windowed_preds = {}
    windowed_labels = {}
    windowed_gest_labels = {}
    windowed_subjects = {}

    for out in outs:
        if out in test_all_preds:
            predictions = np.array(test_all_preds[out])
            e_labels = np.array(test_all_labels[out])
            gestures = np.array(test_all_gest_labels[out])
            subjects = np.array(test_all_subjects[out])

            #Window the predictions
            windowed_preds[out], windowed_labels[out], windowed_gest_labels[out], windowed_subjects[out] = window_predictions(
                predictions, e_labels, gestures, subjects, window_size=window_size, stride=stride, binary=binary)
            
    return (windowed_preds,
            windowed_labels,
            windowed_gest_labels,
            windowed_subjects)


def compute_window_metrics(outs: list,
                 test_all_preds: dict,
                 test_all_labels: dict,
                 test_all_gest_labels: dict,
                 test_all_subjects: dict,
                 window_size: int = 10,
                 stride: int = 6,
                 binary: bool = True) -> tuple:

    """
    Window frame-level predictions and compute metrics for window level predictions.
    Args:
        windowed_preds (dict): Dictionary of window level predictions for each output type.
        windowed_labels (dict): Dictionary of window level labels for each output type.
        windowed_gest_labels (dict): Dictionary of window level gesture labels for each output type.
        windowed_subjects (dict): Dictionary of window level subjects for each output type.

    Returns:
        tuple: A tuple containing the F1 score, accuracy, Jaccard index, and confusion matrix for each output type.

    """

    #Convert frame level predictions to window level predictions
    windowed_preds, windowed_labels, windowed_gest_labels, windowed_subjects = frame2window(
        outs,
        test_all_preds,
        test_all_labels,
        test_all_gest_labels,
        test_all_subjects,
        window_size=window_size,
        stride=stride,
        binary=binary
    )

    window_f1_scores = []
    window_acc_scores = []
    window_jaccard_scores = []
    window_cm_scores = []
    samples_test = []

    for out in windowed_preds:
        
        if out in windowed_preds:
            preds = windowed_preds[out].numpy().flatten()
            labels = windowed_labels[out].numpy().flatten()
            gest_labels = windowed_gest_labels[out].numpy().flatten()
            subjects = windowed_subjects[out]

            #Compute metrics
            if binary:
                f1 = f1_score(labels, preds, average='binary')
                jaccard = jaccard_score(labels, preds, average='binary')

            else:
                f1 = f1_score(labels, preds, average='weighted')
                jaccard = jaccard_score(labels, preds, average='weighted')
            
            acc = accuracy_score(labels, preds)
            cm = confusion_matrix(labels, preds)

            window_f1_scores.append(f1)
            window_acc_scores.append(acc)
            window_jaccard_scores.append(jaccard)
            window_cm_scores.append(cm)

        samples_test.append(len(windowed_preds[out]))

    #Convert lists to numpy arrays
    window_f1_scores = np.array(window_f1_scores)
    window_acc_scores = np.array(window_acc_scores)
    window_jaccard_scores = np.array(window_jaccard_scores)
    window_cm_scores = np.array(window_cm_scores)

    #Compute mean and std for each metric
    mean_f1 = np.average(window_f1_scores, weights=samples_test)
    std_f1 = np.average((window_f1_scores - mean_f1)**2, weights=samples_test)**0.5
    mean_acc = np.average(window_acc_scores, weights=samples_test)
    std_acc = np.average((window_acc_scores - mean_acc)**2, weights=samples_test)**0.5
    mean_jaccard = np.average(window_jaccard_scores, weights=samples_test)
    std_jaccard = np.average((window_jaccard_scores - mean_jaccard)**2, weights=samples_test)**0.5
    cm_total = np.sum(window_cm_scores, axis=0)

    summary_df = pd.DataFrame({
        'F1': [f"{mean_f1:.3f} ± {std_f1:.3f}"],
        'Accuracy': [f"{mean_acc:.3f} ± {std_acc:.3f}"],
        'Jaccard': [f"{mean_jaccard:.3f} ± {std_jaccard:.3f}"],
    }, index=['Windowed Metrics'])

    return (summary_df,
            cm_total)


def create_binary_mask(preds_binary: list,
                       subjects: list,
                       out: str,
                       fold_data_path: str,
                       exp_kwargs: dict=None):

    """
    Create a binary mask from the predictions and subjects to mask specific error predictions.
    In parallel,  Needle Drop positions from the predictions are dropped as well to match the specific error predictions.

    Args:
        preds_binary (list): List of binary predictions.
        subjects (list): List of subjects corresponding to the predictions.
        out (str): The output type (e.g., 'train', 'test').
        fold_data_path (str): Path to the folder containing the data for the fold.

    Returns:
        binary_mask (np.ndarray): Binary mask indicating the positions of the predictions.
        binary_subjects (np.ndarray): Array of subjects corresponding to the binary mask.
    """
    
    binary_mask = np.array(preds_binary[out]) 
    binary_subjects = np.array(subjects[out])

    mask_position_ND_files = []
    if exp_kwargs['delete_ND']:
        
        for file in os.listdir(fold_data_path):
            if file.startswith('mask_position_ND_') and file.endswith('.pth'):
                subject = file.split('.')[0].replace('mask_position_ND_', '')  #Extract subject from filename
                mask_position_ND = torch.load(os.path.join(fold_data_path, file)) 
                mask_position_ND_files.append((subject, mask_position_ND))
    
    #Pre ii) If mask_position_ND exists, find indices of the subject.
    print(binary_mask.shape)
    if mask_position_ND_files != []:
        
        for subject, mask_position_ND in mask_position_ND_files:
    
            subject_indices_ND = np.where(binary_subjects == subject)[0]
            if len(subject_indices_ND) == 0:
                continue
            else:
                print(f"Found mask_position_ND for subject {subject} in {out} set.")

            #Pre iii) Expand mask_position_ND to match
            expanded_mask_position_ND = np.zeros_like(binary_mask, dtype=bool)
            expanded_mask_position_ND[subject_indices_ND] = mask_position_ND

            #Pre iv) If mask_position_ND exists, remove those positions from binary and multi-class predictions.
            binary_mask = binary_mask[~expanded_mask_position_ND]
            binary_subjects = binary_subjects[~expanded_mask_position_ND]
    
    print(binary_mask.shape)

    return binary_mask, binary_subjects
    
    

def create_summary_df(LOSO_f1_train: list,
                      LOSO_f1_test: list,
                      LOSO_acc_train: list,
                      LOSO_acc_test: list,
                      LOSO_jaccard_train: list,
                      LOSO_jaccard_test: list,
                      samples_train: list,
                      samples_test:list,
                      inference_rates: list,
                      train_times: list) -> pd.DataFrame:


    """
    Create a summary df. Columns: f1, acc, jaccard. Index: Train, Test.
    Each entry is mean +- std format

    Args:
        LOSO_f1_train (list): List of F1 scores for training set.
        LOSO_f1_test (list): List of F1 scores for test set.
        LOSO_acc_train (list): List of accuracy scores for training set.
        LOSO_acc_test (list): List of accuracy scores for test set.
        LOSO_jaccard_train (list): List of Jaccard scores for training set.
        LOSO_jaccard_test (list): List of Jaccard scores for test set.
        samples_train (list): List of number of samples in training set.
        samples_test (list): List of number of samples in test set.
        inference_rates (list): List of inference rates for each fold.
        train_times (list): List of training times for each fold.
    
    Returns:
        pd.DataFrame: A DataFrame containing the summary of results.

    """
    
    #Need to compute weighted mean
    summary_df = pd.DataFrame(index=['Train', 'Test'], columns=['F1', 'Accuracy', 'Jaccard', 'Train Time', 'Inference Rate'])    
    summary_df.loc['Train', 'F1'] = f"{np.average(LOSO_f1_train, weights=samples_train):.3f} ± {np.average((LOSO_f1_train - np.average(LOSO_f1_train, weights=samples_train))**2, weights=samples_train)**0.5:.3f}"
    summary_df.loc['Test', 'F1'] = f"{np.average(LOSO_f1_test, weights=samples_test):.3f} ± {np.average((LOSO_f1_test - np.average(LOSO_f1_test, weights=samples_test))**2, weights=samples_test)**0.5:.3f}"
    summary_df.loc['Train', 'Accuracy'] = f"{np.average(LOSO_acc_train, weights=samples_train):.3f} ± {np.average((LOSO_acc_train - np.average(LOSO_acc_train, weights=samples_train))**2, weights=samples_train)**0.5:.3f}"
    summary_df.loc['Test', 'Accuracy'] = f"{np.average(LOSO_acc_test, weights=samples_test):.3f} ± {np.average((LOSO_acc_test - np.average(LOSO_acc_test, weights=samples_test))**2, weights=samples_test)**0.5:.3f}"
    summary_df.loc['Train', 'Jaccard'] = f"{np.average(LOSO_jaccard_train, weights=samples_train):.3f} ± {np.average((LOSO_jaccard_train - np.average(LOSO_jaccard_train, weights=samples_train))**2, weights=samples_train)**0.5:.3f}"
    summary_df.loc['Test', 'Jaccard'] = f"{np.average(LOSO_jaccard_test, weights=samples_test):.3f} ± {np.average((LOSO_jaccard_test - np.average(LOSO_jaccard_test, weights=samples_test))**2, weights=samples_test)**0.5:.3f}"
    summary_df.loc['Train', 'Train Time'] = f"{np.mean(train_times):.2f} ± {np.std(train_times):.2f}"
    summary_df.loc['Test', 'Train Time'] = np.nan #no test time
    summary_df.loc['Train', 'Inference Rate'] = np.nan #no train inference rate
    summary_df.loc['Test', 'Inference Rate'] = f"{np.mean(inference_rates):.2f} ± {np.std(inference_rates):.2f}"

    return summary_df


def save_model(best_model: dict,
               model_path: str) -> None:
    """
    Save the model to the specified path.

    """
    
    torch.save({
            'feature_extractor': best_model['feature_extractor'],
            'model': best_model['model'],
        }, model_path)

    print(f"Model saved to {model_path}")  


def instantiate_model(exp_kwargs: dict,
                      in_features: int,
                      window_size: int,
                      device: torch.device=None) -> torch.nn.Module:

    model_name = exp_kwargs['model_name']
    if model_name == "SimpleCNN":
        model = CNN(in_features=in_features, window_size=window_size, n_classes=exp_kwargs['out_features'] if 'out_features' in exp_kwargs else 1)
    
    elif model_name == "SimpleLSTM":
        hidden_size = exp_kwargs['hidden_size']
        num_layers = exp_kwargs['num_layers']   
        model = LSTM(in_features=in_features, window_size=window_size, 
                     hidden_size=hidden_size, num_layers=num_layers,
                     n_classes=exp_kwargs['out_features'] if 'out_features' in exp_kwargs else 1)

    elif model_name == "Siamese_CNN":
        model = Siamese_CNN(in_features=in_features, window_size=window_size)
    
    elif model_name == "Siamese_LSTM":
        model = Siamese_LSTM(in_features=in_features, window_size=window_size)
    
    elif model_name == "TeCNo":
        model = MultiStageModel(exp_kwargs['mstcn_stages'],
                                exp_kwargs['mstcn_layers'],
                                exp_kwargs['mstcn_f_maps'],
                                exp_kwargs['mstcn_f_dim'],
                                exp_kwargs['out_features'],
                                exp_kwargs['mstcn_causal_conv'])
        
    elif model_name == "TransSVNet":
        model = Transformer(exp_kwargs['mstcn_f_maps'],
                            exp_kwargs['mstcn_f_dim'],
                            exp_kwargs['out_features'],
                            exp_kwargs['sequence_length'])
        

    elif model_name == "COG":
        
        #model = COG(num_layers_Basic, num_layers_R, num_R, mstcn_f_maps, mstcn_f_dim, out_features, mstcn_causal_conv, d_model, d_q, len_q, device)

        #Using exp_kwargs to pass the parameters
        try:
            model = COG(exp_kwargs['num_layers_Basic'], 
                        exp_kwargs['num_layers_R'], 
                        exp_kwargs['num_R'], 
                        exp_kwargs['mstcn_f_maps'], 
                        exp_kwargs['mstcn_f_dim'], 
                        exp_kwargs['out_features'], 
                        exp_kwargs['mstcn_causal_conv'], 
                        exp_kwargs['d_model'], 
                        exp_kwargs['d_q'], 
                        exp_kwargs['sequence_length'],
                        device=device,
                        SRM=exp_kwargs['SRM'],
                        use_all_gestures=exp_kwargs['use_all_gestures'],
                        use_skill_prompt=exp_kwargs['use_skill_prompt'])
        except:
            model = COG(exp_kwargs['num_layers_Basic'], 
                        exp_kwargs['num_layers_R'], 
                        exp_kwargs['num_R'], 
                        exp_kwargs['mstcn_f_maps'], 
                        exp_kwargs['mstcn_f_dim'], 
                        exp_kwargs['out_features'], 
                        exp_kwargs['mstcn_causal_conv'], 
                        exp_kwargs['d_model'], 
                        exp_kwargs['d_q'], 
                        exp_kwargs['sequence_length'],
                        device=device)


    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    return model