import torch
import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, jaccard_score, confusion_matrix
import numpy as np

def train_single_epoch(model: torch.nn.Module, 
                       feature_extractor: torch.nn.Module,
                       train_dataloader: DataLoader,
                       criterion: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
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

    feature_extractor.train()
    model.train()
    
    train_loss, train_f1, train_acc, train_auc_roc, train_jaccard = 0.0, 0.0, 0.0, 0.0, 0.0
    train_cm = np.zeros((2, 2), dtype=int)  # Assuming binary classification

    for i, batch in enumerate(tqdm.tqdm(train_dataloader, 
                                        total=len(train_dataloader),
                                        )):
        
        #images, kinematics, g_labels, e_labels, task, trial, subject = batch
        images, kinematics, g_labels, e_labels, subject = batch
        
        if not exp_kwargs['siamese']:
            if exp_kwargs['binary_error']:
                e_labels = e_labels[:, 4] #Last column is the binary overall error label
    

        images = images.to(device)
        kinematics = kinematics.to(device)
        g_labels = g_labels.to(device).float()
        e_labels = e_labels.to(device).float()

        #Forward pass
        image_features = feature_extractor(images)
        inputs = torch.cat((image_features, kinematics), dim = 2).permute(0, 2, 1)
        
        if inputs.size(0) < 2:
            continue
        
        outputs = model(inputs)
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
        train_f1 += f1_score(e_labels.detach().cpu().numpy(), outputs_binary.detach().cpu().numpy(), average='weighted')
        train_acc += accuracy_score(e_labels.detach().cpu().numpy(), outputs_binary.detach().cpu().numpy())
        train_auc_roc += roc_auc_score(e_labels.detach().cpu().numpy(), outputs_sigmoid.detach().cpu().numpy(), average='weighted')
        train_jaccard += jaccard_score(e_labels.detach().cpu().numpy(), outputs_binary.detach().cpu().numpy(), average='weighted')
        train_cm += confusion_matrix(e_labels.detach().cpu().numpy(), outputs_binary.detach().cpu().numpy())
    
    train_loss /= len(train_dataloader)
    train_f1 /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    train_auc_roc /= len(train_dataloader)
    train_jaccard /= len(train_dataloader)

    return train_loss, train_f1, train_acc, train_auc_roc, train_jaccard, train_cm



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
    model.eval()
    feature_extractor.eval()

    test_loss, test_f1, test_acc, test_auc_roc, test_jaccard = 0.0, 0.0, 0.0, 0.0, 0.0
    test_cm = np.zeros((2, 2), dtype=int)  # Assuming binary classification

    with torch.no_grad():
        for batch in tqdm.tqdm(test_dataloader, desc="Test"):
            images, kinematics, g_labels, e_labels, subject = batch
            images = images.to(device)
            kinematics = kinematics.to(device)
            g_labels = g_labels.to(device).float()
            e_labels = e_labels.to(device).float()

            if exp_kwargs['binary_error']:
                e_labels = e_labels[:, 4]

            #Forward pass
            image_features = feature_extractor(images)
            inputs = torch.cat((image_features, kinematics), dim=2).permute(0,2,1)                    
            outputs = model(inputs)
            outputs = outputs.squeeze(1)

            #Loss
            loss = criterion(outputs, e_labels)
            test_loss += loss.item()

            #Metrics
            outputs_sigmoid = torch.sigmoid(outputs)
            outputs_binary = (outputs_sigmoid > 0.5).float()
            test_f1 += f1_score(e_labels.detach().cpu().numpy(), outputs_binary.cpu().numpy(), average='weighted')
            test_acc += accuracy_score(e_labels.detach().cpu().numpy(), outputs_binary.cpu().numpy())
            test_auc_roc += roc_auc_score(e_labels.detach().cpu().numpy(), outputs_sigmoid.cpu().numpy(), average='weighted')
            test_jaccard += jaccard_score(e_labels.detach().cpu().numpy(), outputs_binary.cpu().numpy(), average='weighted')
            test_cm += confusion_matrix(e_labels.detach().cpu().numpy(), outputs_binary.cpu().numpy())

    test_loss /= len(test_dataloader)
    test_f1 /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    test_jaccard /= len(test_dataloader)
    test_auc_roc /= len(test_dataloader)

    return test_loss, test_f1, test_acc, test_auc_roc, test_jaccard, test_cm


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
            'train_auc_roc_fold': best_model['train_auc_roc_fold'],
            'test_auc_roc_fold': best_model['test_auc_roc_fold'],
            'train_jaccard_fold': best_model['train_jaccard_fold'],
            'test_jaccard_fold': best_model['test_jaccard_fold'],
            'train_cm_fold': best_model['train_cm_fold'].tolist(),
            'test_cm_fold': best_model['test_cm_fold'].tolist(),
        }, model_path)

    print(f"Model saved to {model_path}")  