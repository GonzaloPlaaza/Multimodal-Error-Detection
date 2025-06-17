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
                 train=True,
                 feature_standardization_dict={}):
        
        self.image_data_train = image_data_train
        self.kinematics_data_train = kinematics_data_train
        #pairs_df = pd.DataFrame(columns=['subject_1', 'gesture_label_1', 'position_1', 'subject_2', 'gesture_label_2', 'position_2', 'label])
        #subject_1 is train and subject_2 is test in case of the test set
        self.pairs_df = pairs_df 
        self.feature_standardization_dict = feature_standardization_dict
        self.train = train

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):

        position_1 = self.pairs_df.iloc[idx]['position_1']
        position_2 = self.pairs_df.iloc[idx]['position_2']

        paired_images = torch.empty((2, 30, 2048), dtype=torch.float32)

        if self.train:
            paired_images = self.image_data[idx] #shape (2, 30, 2048) (2 images, 30 frames, 2048 features)
            paired_kinematics = self.kinematics_data[idx] #shape (2, 30, 26) 

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
