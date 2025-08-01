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
        'all_errors': [0, 1, 2, 3, 4, 5] #list of all error types (and no error)
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
            e_labels_specific = e_labels[:, error_position] if error_position >= 0 else e_labels[:, -1]
        
        elif exp_kwargs['dataset_type'] == "frame":
            #Extract the specific error labels based on the error type
            e_labels_specific = e_labels[:, :, error_position] if error_position >= 0 else e_labels[:, :, -1]         

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
        pos_weight = torch.tensor(class_counts[0] / class_counts[1], device=device, dtype=torch.float32) #downsample the positive class
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    else:
        if exp_kwargs['dataset_type'] == "window":
            criterion = nn.BCEWithLogitsLoss()

        elif exp_kwargs['dataset_type'] == "frame":
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
    train_cm = np.zeros((2, 2), dtype=int)  # Assuming binary classification

    for i, batch in enumerate(tqdm.tqdm(train_dataloader, 
                                        total=len(train_dataloader),
                                        )):
        
        #images, kinematics, g_labels, e_labels, task, trial, subject = batch
        images, kinematics, g_labels, e_labels, subject = batch
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

    if scheduler is not None:
        scheduler.step()
    
    train_loss /= len(train_dataloader)
    train_f1 /= len(train_dataloader)
    train_f1_weighted /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    train_jaccard /= len(train_dataloader)

    return train_loss, train_f1, train_f1_weighted, train_acc, train_jaccard, train_cm



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
            images, kinematics, g_labels, e_labels, subject = batch
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
                
                test_f1 += f1_score(e_labels_specific.detach().cpu().numpy(), outputs_binary.detach().cpu().numpy(), average='binary', pos_label=1)
                test_f1_weighted += f1_score(e_labels_specific.detach().cpu().numpy(), outputs_binary.detach().cpu().numpy(), average='weighted')
                test_acc += accuracy_score(e_labels_specific.detach().cpu().numpy(), outputs_binary.detach().cpu().numpy())
                test_jaccard += jaccard_score(e_labels_specific.detach().cpu().numpy(), outputs_binary.detach().cpu().numpy(), average='binary', pos_label=1)
                test_cm += confusion_matrix(e_labels_specific.detach().cpu().numpy(), outputs_binary.detach().cpu().numpy())


            for j in range(outputs_binary.shape[0]):
                test_all_preds.append(outputs_binary[j].item())
                test_all_probs.append(outputs_sigmoid[j].item())
                test_all_labels.append(e_labels[j])
                test_all_labels_specific.append(e_labels_specific[j].item())
                test_all_gest_labels.append(g_labels[j].item())
                if exp_kwargs['dataset_type'] == "frame":
                    test_all_subjects.append(subject)
                else:
                    test_all_subjects.append(subject[j])

            total_time = (end_time - start_time)

    test_loss /= len(test_dataloader)

    if not exp_kwargs['dataset_type'] == "frame":
        test_f1 = f1_score(test_all_labels_specific, test_all_preds, average='binary')
        test_f1_weighted = f1_score(test_all_labels_specific, test_all_preds, average='weighted')
        test_acc = accuracy_score(test_all_labels_specific, test_all_preds)
        test_jaccard = jaccard_score(test_all_labels_specific, test_all_preds, average='binary')
        test_cm = confusion_matrix(test_all_labels_specific, test_all_preds)

    else:
        test_f1 /= len(test_dataloader)
        test_f1_weighted /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        test_jaccard /= len(test_dataloader)

    inference_rate = total_time * 1000  #Convert to ms per frame

    return test_loss, test_f1, test_f1_weighted, test_acc, test_jaccard, test_cm, inference_rate, test_all_preds, test_all_probs, test_all_labels, test_all_labels_specific, test_all_gest_labels, test_all_subjects



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
        images, kinematics, g_labels, e_labels, subject = batch
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
        no_e_labels = no_e_labels.to(outputs.device) #Move to the same device as outputs
        e_labels_complete = torch.cat((no_e_labels, e_labels_specific), dim=0) #Concatenate no error and error labels; shape (2, n_frames)
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
            images, kinematics, g_labels, e_labels, subject = batch
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
                test_all_probs.append(outputs[j].item())
                test_all_preds.append(outputs_binary[j].item())
                test_all_labels.append(e_labels[j])
                test_all_labels_specific.append(e_labels_specific[j].item())
                test_all_gest_labels.append(g_labels[j].item())
                test_all_subjects.append(subject)
            
            total_time += (end_time - start_time)

    test_loss /= len(test_dataloader)
    test_f1 = f1_score(test_all_labels, test_all_preds, average='binary')
    test_f1_weighted = f1_score(test_all_labels, test_all_preds, average='weighted')
    test_acc = accuracy_score(test_all_labels, test_all_preds)
    test_jaccard = jaccard_score(test_all_labels, test_all_preds, average='binary')
    test_cm = confusion_matrix(test_all_labels, test_all_preds)
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
    
    train_all_scores = []
    train_all_preds = []
    train_all_preds_binary = []
    train_all_labels = []
    train_all_labels_binary = []
    train_loss = 0.0

    for i, batch in enumerate(tqdm.tqdm(train_dataloader, 
                                        total=len(train_dataloader),
                                        )):
        
        #images, kinematics, g_labels, e_labels, task, trial, subject = batch
        images, kinematics, g_labels, e_labels, subject = batch
        g_labels = g_labels.to(device).float()
        e_labels_specific = define_error_labels(e_labels = e_labels, exp_kwargs=exp_kwargs)
        e_labels_specific = e_labels_specific.to(device).float()

        if exp_kwargs['error_type'] == "global":
            e_labels_specific = e_labels_specific.view(-1,)  #Ensure e_labels is of shape (n_frames,)

        else: #specific error prediction (6 classes for CE loss)
            e_labels_specific = torch.argmax(e_labels_specific, dim=1)  #Convert to class labels (0-5) for CE loss

        if i == 0 or i == 20:
            print(e_labels_specific.shape)
            print("Labels unique classes:", torch.unique(e_labels_specific))

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
            print(p_classes.shape, l.shape)
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
                train_all_scores.append(float(train_p_classes_positive.data[j]))
        
        for index in range(len(preds)):
            train_all_preds.append(int(preds.data[index]))

        #Record binary predictions if error_type is all_errors
        if exp_kwargs['error_type'] == "all_errors":
            for index in range(len(preds)):
                train_all_preds_binary.append(int(preds.data[index, 0])) #No error class is the first class (0). Therefore, we are recording the binary prediction (0 or 1) for no error.
                train_all_labels_binary.append(int(e_labels_specific.data[index, 0])) #No error class is the first class (0). Therefore, we are recording the binary label (0 or 1) for no error.

        for index in range(len(e_labels_specific)):
            train_all_labels.append(int(e_labels_specific.data[index]))

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

        return train_average_loss, train_f1, train_f1_weighted, train_acc, train_jaccard, train_cm
        
    else: #Specific error prediction (6 classes) --> use weighted metrics
        train_f1_binary = f1_score(train_all_labels_binary, train_all_preds_binary, average='binary', pos_label=0)
        train_jaccard_binary = jaccard_score(train_all_labels_binary, train_all_preds_binary, average='binary', pos_label=0)
        train_accuracy_binary = accuracy_score(train_all_labels_binary, train_all_preds_binary)
        train_cm_binary = confusion_matrix(train_all_labels_binary, train_all_preds_binary)

        train_f1 = f1_score(train_all_labels, train_all_preds, average='weighted')
        train_jaccard = jaccard_score(train_all_labels, train_all_preds, average='weighted')
        train_accuracy = accuracy_score(train_all_labels, train_all_preds)
        train_cm = confusion_matrix(train_all_labels, train_all_preds)
        
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
    test_all_labels = []
    test_all_labels_specific = []
    test_all_gest_labels = []
    test_all_subjects = []
    test_start_time = time.time()
    test_progress = 0
    val_batch_size = 1
    test_loss = 0.0
    total_time = 0.0
    
    with torch.no_grad():
        
        for i, batch in enumerate(tqdm.tqdm(test_dataloader, 
                                        total=len(test_dataloader),
                                        )):
            
            #images, kinematics, g_labels, e_labels, task, trial, subject = batch
            images, kinematics, g_labels, e_labels, subject = batch
            g_labels = g_labels.to(device).float()
            g_labels = g_labels.squeeze(0)  #Remove the extra dimension for frame classification
            e_labels_specific = define_error_labels(e_labels = e_labels, exp_kwargs=exp_kwargs)
            e_labels_specific = e_labels_specific.to(device).float()
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

            for j in range(len(p_classes_positive)):
                test_all_probs.append(float(p_classes_positive.data[j]))
            for j in range(len(preds)):
                test_all_preds.append(int(preds.data[j]))
            for j in range(len(e_labels)):
                test_all_labels.append(e_labels[j].tolist())
            for j in range(len(e_labels_specific)):
                test_all_labels_specific.append(int(e_labels_specific.data[j]))
            for j in range(len(g_labels)):
                test_all_gest_labels.append(int(g_labels.data[j]))
                test_all_subjects.append(subject)
            
            test_progress += 1
            total_time += (end_time - start_time)

    test_average_loss = float(test_loss) / len(test_dataloader)
    test_jaccard = jaccard_score(test_all_labels_specific, test_all_preds, average='binary', pos_label=1)
    test_f1 = f1_score(test_all_labels_specific, test_all_preds, average='binary', pos_label=1)
    test_f1_weighted = f1_score(test_all_labels_specific, test_all_preds, average='weighted')
    test_acc = accuracy_score(test_all_labels_specific, test_all_preds)
    test_cm = confusion_matrix(test_all_labels_specific, test_all_preds)
    inference_rate = (total_time / images.shape[1]) * 1000  #Convert to ms per frame

    return test_average_loss, test_f1, test_f1_weighted, test_acc, test_jaccard, test_cm, inference_rate, test_all_preds, test_all_probs, test_all_labels, test_all_labels_specific, test_all_gest_labels, test_all_subjects


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
    model.load_state_dict(best_model['model_state_dict'])
    model.to(device)
    
    #Load feature extractor if necessary
    if exp_kwargs['data_type'] != 'kinematics':

        if exp_kwargs['video_dims'] != 2048:
            feature_extractor.load_state_dict(best_model['feature_extractor_state_dict'])
            feature_extractor.to(device) 

    return feature_extractor, model


def process_all_labels(all_labels: list,
                       exp_kwargs: dict) -> torch.tensor:

    """
    This function processess the all_labels list. Why?
    all_labels should be a tensor of lenght n_windows/n_frames, where each elements is a tensor of shape (5,) -as there are 5 error types-.
    When printing one of the elements, it shows:
    tensor([1., 1., 0., 0., 1.])
    However, this is a string! Meaning element[0] is 't', element[1] is 'e', and so on. This is due to mlflow's serialization of tensors.
    We need to convert the string to a tensor of floats.

    Args:
        all_labels (list): List of labels to process.
    
    Returns:
        torch.tensor: Processed tensor of labels.
    """

    if exp_kwargs['compute_from_str'] == True:
        
        length = len(all_labels)
        processed_labels = torch.zeros((length, 5), dtype=torch.float32)

        #Manually define the positions of the string where the labels are stored. t is at position 0, e at position 1, and so on. An example string is "tensor([1., 1., 0., 0., 1.])"
        error_label_positions = [8, 12, 16, 20, 24]  #Positions of t, e, n, a, l in the string

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

    if exp_kwargs['dataset_type'] == 'frame':
        test_all_preds, test_all_probs, test_all_labels, test_all_labels_specific, test_all_gest_labels, test_all_subjects = ({} for _ in range(6))

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

            if exp_kwargs['dataset_type'] == 'frame':
                try:
                    test_all_preds[out] = best_model_dict['test_all_preds_fold']
                    test_all_probs[out] = best_model_dict['test_all_probs_fold']
                    test_all_labels[out] = best_model_dict['test_all_labels_fold']
                    test_all_labels_specific[out] = best_model_dict['test_all_labels_specific_fold']
                    test_all_gest_labels[out] = best_model_dict['test_all_gest_labels_fold']
                    test_all_subjects[out] = best_model_dict['test_all_subjects_fold']
                
                except:
                    test_all_preds[out] = best_model_dict['test_all_preds']
                    test_all_probs[out] = best_model_dict['test_all_probs']
                    test_all_labels[out] = best_model_dict['test_all_labels']
                    test_all_labels_specific[out] = best_model_dict['test_all_labels_specific']
                    test_all_gest_labels[out] = best_model_dict['test_all_gest_labels']
                    test_all_subjects[out] = best_model_dict['test_all_subjects']

                test_all_labels[out] = process_all_labels(test_all_labels[out], exp_kwargs=exp_kwargs)

    #Change confusion matrices to integer type
    LOSO_cm_train = LOSO_cm_train.astype(int)
    LOSO_cm_test = LOSO_cm_test.astype(int)

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
    

def window_predictions(predictions,
                       e_labels,
                       gestures,
                       subjects,
                       window_size=10,
                          stride=6):
    
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
            predictions_windows[-1] = 1.0 if predictions_windows[-1] >= 0.5 else 0.0
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
                 stride: int = 6) -> tuple:
    
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
                predictions, e_labels, gestures, subjects, window_size=window_size, stride=stride)
            
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
                 stride: int = 6) -> tuple:
    
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
        stride=stride
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
            f1 = f1_score(labels, preds, average='binary')
            acc = accuracy_score(labels, preds)
            jaccard = jaccard_score(labels, preds, average='binary')
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
            'feature_extractor_state_dict': best_model['feature_extractor'],
            'model_state_dict': best_model['model'],
            'epoch': best_model['epoch'],
            'train_f1_fold': best_model['train_f1_fold'],
            'test_f1_fold': best_model['test_f1_fold'],
            'train_acc_fold': best_model['train_acc_fold'],
            'test_acc_fold': best_model['test_acc_fold'],
            'train_jaccard_fold': best_model['train_jaccard_fold'],
            'test_jaccard_fold': best_model['test_jaccard_fold'],
            'train_cm_fold': best_model['train_cm_fold'],
            'test_cm_fold': best_model['test_cm_fold'],
        }, model_path)

    print(f"Model saved to {model_path}")  


def instantiate_model(exp_kwargs: dict,
                      in_features: int,
                      window_size: int,
                      device: torch.device=None) -> torch.nn.Module:
    
    model_name = exp_kwargs['model_name']
    if model_name == "SimpleCNN":
        model = CNN(in_features=in_features, window_size=window_size)
    
    elif model_name == "SimpleLSTM":
        hidden_size = exp_kwargs['hidden_size']
        num_layers = exp_kwargs['num_layers']   
        model = LSTM(in_features=in_features, window_size=window_size, 
                     hidden_size=hidden_size, num_layers=num_layers)

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