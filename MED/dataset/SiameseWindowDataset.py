from torch.utils.data import Dataset
import torch

class SiameseWindowDataset(Dataset):

    """
    Class to create a Siamese window dataset for multimodal error detection.

    Input:
        image_data: Tensor of paired image features. Shape: n_samples x 2 x 30 x 2048
        kinematics_data: Tensor of paired kinematics features. Shape: n_samples x 2 x 30 x 26
        pairs_df: DataFrame containing pairs of indices and labels.
        train: Boolean indicating if the dataset is for training or testing.
        feature_standardization_dict: Dictionary for feature standardization.

    Returns: None
    """

    def __init__(self, 
                 image_data_train, 
                 kinematics_data_train, 
                 pairs_df,
                 image_data_test=None,
                 kinematics_data_test=None,  
                 train=True,
                 feature_standardization_dict={}):
        
        self.image_data_train = image_data_train
        self.kinematics_data_train = kinematics_data_train
        #pairs_df = pd.DataFrame(columns=['subject_1', 'gesture_label_1', 'position_1', 'subject_2', 'gesture_label_2', 'position_2', 'label])
        #subject_1 is train and subject_2 is test in case of the test set
        self.pairs_df = pairs_df 
        self.image_data_test = image_data_test
        self.kinematics_data_test = kinematics_data_test
        self.feature_standardization_dict = feature_standardization_dict

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):

        position_1 = self.pairs_df.iloc[idx]['position_1']
        position_2 = self.pairs_df.iloc[idx]['position_2']

        paired_images = torch.empty((2, 30, 2048), dtype=torch.float32)
        paired_kinematics = torch.empty((2, 30, 26), dtype=torch.float32)

        if self.train:
            paired_images[0] = self.image_data_train[position_1]
            paired_images[1] = self.image_data_train[position_2]
            paired_kinematics[0] = self.kinematics_data_train[position_1]
            paired_kinematics[1] = self.kinematics_data_train[position_2]
        
        else:
            #Check image_data_test and kinematics_data_test are not None
            if self.image_data_test is not None and self.kinematics_data_test is not None:
                paired_images[0] = self.image_data_train[position_1] #subject_1 is train, as we are comparing each test window with n_comparisons in train set
                paired_images[1] = self.image_data_test[position_2]
                paired_kinematics[0] = self.kinematics_data_train[position_1]
                paired_kinematics[1] = self.kinematics_data_test[position_2]
                
            else:
                raise ValueError("Test data is not available.")

        for key, value in self.feature_standardization_dict.items():
            if key == 'image':
                paired_images = (paired_images - value['mean']) / value['std']
            elif key == 'kinematics':
                paired_kinematics = (paired_kinematics - value['mean']) / value['std']
        
        paired_indices = self.pairs_df.iloc[idx][['position_1', 'position_2']].values
        label = self.pairs_df.iloc[idx]['label']

        return (paired_images,
                paired_kinematics,
                paired_indices[0],  # position_1
                paired_indices[1],  # position_2
                label)
