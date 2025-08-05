import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import os


class CustomFrameDataset(Dataset):
    
    def __init__(self,
                 fold_data_path:str,
                 video_data_path:str = None,
                 error_type:str = 'global',
                 csv_filename: str = 'trains.csv',
                 delete_ND: bool = True):
        
        self.fold_data_path = fold_data_path
        self.video_data_path = video_data_path
        self.error_type = error_type
        self.feature_standardization_dict = self.load_feature_standardization_dict()
        self.csv_file = pd.read_csv(os.path.join(fold_data_path, csv_filename), header=None, names=['files'])
        self.delete_ND = delete_ND

        self.skill_level_dict = {
            'B': 'Novice', #Novice
            'C': 'Intermediate', #Intermediate
            'D': 'Expert', #Expert
            'E': 'Expert', #Expert
            'F': 'Intermediate',
            'G': 'Novice', 
            'H': 'Novice',
            'I': 'Expert'}
     

    def __len__(self):
        
        return len(self.csv_file)           

    def __getitem__(self, idx):
        
        file_name = self.csv_file['files'].iloc[idx]
        
        if self.video_data_path is not None:
            file_path = os.path.join(self.video_data_path, file_name)
            file_path2 = os.path.join(self.fold_data_path, file_name)

            with open(file_path2, 'rb') as f:
                data2 = pickle.load(f)
        
        else:
            file_path = os.path.join(self.fold_data_path, file_name)
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        if self.video_data_path is not None:
            
            n_frames_in_trial = data2['kinematics_feats'].shape[0]
            
            images = torch.tensor(data['feature'].astype('float32'))
            kinematics =  data2['kinematics_feats'].clone().detach()  
            e_labels = data2['e_labels'].clone().detach()
            g_labels = torch.tensor(data2['g_labels'].astype('float32').reshape(n_frames_in_trial, 1))
            subject = file_name[:-4]

        else:
            n_frames_in_trial = data['image_feats'].shape[0]
            
            #Extract features and labels
            images = data['image_feats'].reshape(n_frames_in_trial, 2048).clone().detach()
            kinematics = data['kinematics_feats'].clone().detach()
            g_labels = torch.tensor(data['g_labels'].reshape(n_frames_in_trial, 1))
            e_labels = data['e_labels'].reshape(n_frames_in_trial, 5).clone().detach()

            #Extract subject
            file_name = file_name[:-4]
            subject = file_name

            #Standardize features if needed

        #Transform error labels to powerset format
        e_labels, mask_position_needle_drop = self.powerset_error_labels(e_labels, delete_ND=True) #Convert error labels to powerset format

        if self.delete_ND:
            images = images[~mask_position_needle_drop]
            kinematics = kinematics[~mask_position_needle_drop]
            g_labels = g_labels[~mask_position_needle_drop]
            e_labels = e_labels[~mask_position_needle_drop]
        

        for key, value in self.feature_standardization_dict.items():
            if key == 'kinematics':
                kinematics = (kinematics - value['mean']) / value['std']

        #Skill level: convert skill level to numerical value: novice = [1, 0, 0], intermediate = [0, 1, 0], expert = [0, 0, 1]
        skill_level = torch.zeros((kinematics.size(0), 3))

        #Subject is Needle_Passing_B001
        subject_letter = subject[-4] #B, C, D, E, F, G, H, I
        skill = self.skill_level_dict[subject_letter]
        if skill == 'Novice':
            skill_level[:, 0] = 1
        elif skill == 'Intermediate':
            skill_level[:, 1] = 1
        elif skill == 'Expert':
            skill_level[:, 2] = 1
        else:
            raise ValueError(f"Skill level {skill} not recognized.")

        return images, kinematics, g_labels, e_labels, subject, skill_level
    
    def load_feature_standardization_dict(self):
        """
        Load the feature standardization dictionary from the specified path.
        """
        image_mean = torch.load(os.path.join(self.fold_data_path, 'mean_features.pth'))
        image_std = torch.load(os.path.join(self.fold_data_path, 'std_features.pth'))
        kinematics_mean = torch.load(os.path.join(self.fold_data_path, 'mean_kinematics.pth'))
        kinematics_std = torch.load(os.path.join(self.fold_data_path, 'std_kinematics.pth'))
        feature_standardization_dict = {
            'image': {'mean': image_mean, 'std': image_std},
            'kinematics': {'mean': kinematics_mean, 'std': kinematics_std}
        }

        return feature_standardization_dict
    

    def get_n_frames(self):

        """
        Calculate the total number of frames across all files in the dataset.
        Returns:
            int: Total number of frames.

        """
        frame_length = 0
        for file in self.csv_file['files']:
            if not file.endswith('.pkl'):
                raise ValueError(f"File {file} does not end with .pkl. Please check the CSV file.")
            
            file_path = os.path.join(self.fold_data_path, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} does not exist. Please check the path.")
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                if 'image_feats' in data:
                    n_frames = data['image_feats'].shape[0]
                
                elif 'feature' in data:
                    n_frames = data['feature'].shape[0]
                
                else:
                    raise KeyError(f"Neither 'image_feats' nor 'feature' found in {file}. Please check the data format.")
                
            frame_length += n_frames
            
        return frame_length  
    
    def powerset_error_labels(self, e_labels_data: torch.tensor,
                          delete_ND: bool = True) -> tuple:

        """
        Process error labels to create a powerset for specific error classification algorithm, since some smaples have more than one error label.

        Args:
            e_labels_data (torch.tensor): Tensor of error labels.
        
        Returns:
            e_labels_data_powerset (torch.tensor): Tensor of error labels in powerset format.

        Description:
            e_labels_data is a tensor of shape (n_frames, 5), where each row corresponds to a frame and each column corresponds to an error label.
            Namely, the error labels are: Out_of_View, Needle_Drop, Multiple_Attempts, Needle_Position and Error (which is the global error label).
            After processing, and since some combinations don't exist or are less frequent than 2% across folds (e.g., Needle_Drop is not considered), we want the following labels:
            0: No Error
            1: Out_of_View (which includes Out_of_View_Needle_Drop)
            2: Multiple_Attempts (which includes Multiple_Attempts_Needle_Drop)
            3: Needle_Position (which includes Out_of_View_Needle_Position)
            4: Out_of_View_Multiple_Attempts
            5: Multiple_Attempts_Needle_Position
            6: Error (which is the global error label, i.e., any error)
            7: Needle_Drop (which is not considered in the powerset, but we want to keep track of its positions)
        """

        #Create a new tensor with the same shape as e_labels_data, but with 8 columns (for the 8 error labels)
        e_labels_data_powerset = torch.zeros((e_labels_data.shape[0], 7), dtype=torch.int)

        #Create a boolean mask to track positions of Needle Drop, if True (i.e., Needle Drop is present), it will be deleted afterwards from e_labels_data_powerset
        mask_positions_needle_drop = torch.zeros((e_labels_data.shape[0],), dtype=torch.bool)

        #Iterate over each frame and assign the appropriate label
        for i in range(e_labels_data.shape[0]):
            
            #Error
            if e_labels_data[i, 4] == 1: 
                e_labels_data_powerset[i, 6] = 1
                e_labels_data_powerset[i, 0] = 0
            
                #OOV: includes only OOV (i.e., sum from 0 to 3 is 1, only one error), OOV + ND (simultaneous) but does not consider OOV + NP (as it goes to NP)
                if (e_labels_data[i, 0] == 1 and e_labels_data[i, :4].sum() == 1) \
                    or (e_labels_data[i, 0] == 1 and e_labels_data[i, 1] == 1):
                    
                    e_labels_data_powerset[i, 1] = 1
                
                #Multiple Attempts: includes only MA (i.e., sum from 0 to 3 is 1, only one error), MA + ND (simultaneous) but does not consider MA + NP (as it goes to NP)
                elif (e_labels_data[i, 2] == 1 and e_labels_data[i, :4].sum() == 1) \
                    or (e_labels_data[i, 2] == 1 and e_labels_data[i, 1] == 1):
                    
                    e_labels_data_powerset[i, 2] = 1

                #Needle Position: includes only NP (i.e., sum from 0 to 3 is 1, only one error), NP + OOV (simultaneous) but does not consider NP + MA (as it goes alone)
                elif (e_labels_data[i, 3] == 1 and e_labels_data[i, :4].sum() == 1) \
                    or (e_labels_data[i, 3] == 1 and e_labels_data[i, 0] == 1):

                    e_labels_data_powerset[i, 3] = 1

                #OOV + Multiple Attempts: includes OOV + MA (simultaneous) 
                elif e_labels_data[i, 0] == 1 and e_labels_data[i, 2] == 1:
                    e_labels_data_powerset[i, 4] = 1

                #Multiple Attempts + Needle Position: includes MA + NP (simultaneous)
                elif e_labels_data[i, 2] == 1 and e_labels_data[i, 3] == 1:
                    e_labels_data_powerset[i, 5] = 1

                elif e_labels_data[i, 1] == 1:
                    
                    if delete_ND:
                        #Needle Drop is not considered in the powerset, but we want to keep track of its positions
                        e_labels_data_powerset[i, 0] = 0 
                        e_labels_data_powerset[i, 6] = 0 
                        mask_positions_needle_drop[i] = True

                    else: 
                        continue

                else:
                    print(f"Error label not recognized for frame {i}: {e_labels_data[i]}. Setting to No Error.")

            #No error
            else:
                e_labels_data_powerset[i, 0] = 1 
                e_labels_data_powerset[i, 6] = 0 

        return e_labels_data_powerset, mask_positions_needle_drop
        