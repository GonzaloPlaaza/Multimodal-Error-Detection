import torch
import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, jaccard_score, confusion_matrix
import numpy as np

from .models import CNN, LSTM, Siamese_CNN, Siamese_LSTM

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


def train_single_epoch_siamese(model: torch.nn.Module, 
                       feature_extractor: torch.nn.Module,
                       train_dataloader: DataLoader,
                       criterion: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
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

    feature_extractor.train()
    model.train()
    
    train_loss, train_f1, train_acc, train_auc_roc, train_jaccard = 0.0, 0.0, 0.0, 0.0, 0.0
    train_cm = np.zeros((2, 2), dtype=int) 

    for i, batch in enumerate(tqdm.tqdm(train_dataloader, 
                                        total=len(train_dataloader),
                                        )):
        
        paired_images, paired_kinematics, position_1, position_2, e_labels = batch #The shape of data is (2, 30, 2048) for images and (2, 30, 26) for kinematics. label is 1D

        paired_images = paired_images.to(device)
        paired_kinematics = paired_kinematics.to(device)
        e_labels = e_labels.to(device).float()

        #Forward pass
        paired_image_features = feature_extractor(paired_images)
        #Concatenate img and kin features to obtain shape ((batch_size, 2, 30, 58))
        inputs = torch.cat((paired_image_features, paired_kinematics), dim=3).permute(0, 1, 3, 2) #Permute to have shape (batch_size, 2, 58, 30)
        
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
    
        #Evaluate on test set
        model.eval()
        feature_extractor.eval()
    
        test_loss, test_f1, test_acc, test_auc_roc, test_jaccard = 0.0, 0.0, 0.0, 0.0, 0.0
        test_cm = np.zeros((2, 2), dtype=int) 

        #Store the ids of test samples to then compute the average prediction across the paired training samples.
        id_test = np.empty((0, 1), dtype=int) 
        all_true = np.empty((0, 1), dtype=int) #Store the true labels for each sample
        all_preds = np.empty((0, 1), dtype=int) #Store the predictions for each sample
    
        with torch.no_grad():
            for batch in tqdm.tqdm(test_dataloader, desc="Test"):
                paired_images, paired_kinematics, position_1, position_2, e_labels = batch
    
                paired_images = paired_images.to(device)
                paired_kinematics = paired_kinematics.to(device)
                e_labels = e_labels.to(device).float()
    
                #Forward pass
                paired_image_features = feature_extractor(paired_images)
                inputs = torch.cat((paired_image_features, paired_kinematics), dim=3).permute(0, 1, 3, 2)
                input1, input2 = inputs[:, 0, :, :], inputs[:, 1, :, :]
                outputs = model(input1, input2)
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
        test_f1 = f1_score(y_true, y_pred, average='weighted')
        test_acc = accuracy_score(y_true, y_pred)
        test_auc_roc = roc_auc_score(y_true, y_pred_vote, average='weighted')
        test_jaccard = jaccard_score(y_true, y_pred, average='weighted')
        test_cm = confusion_matrix(y_true, y_pred)
        
        test_loss /= len(test_dataloader)

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


def instantiate_model(model_name: str,
                      in_features: int,
                      window_size: int):
    
    if model_name == "SimpleCNN":
        model = CNN(in_features=in_features, window_size=window_size)
    
    elif model_name == "SimpleLSTM":
        model = LSTM(in_features=in_features, window_size=window_size)

    elif model_name == "Siamese_CNN":
        model = Siamese_CNN(in_features=in_features, window_size=window_size)
    
    elif model_name == "Siamese_LSTM":
        model = Siamese_LSTM(in_features=in_features, window_size=window_size)
    
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    return model