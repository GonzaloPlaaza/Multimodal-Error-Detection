from torch.utils.data import Dataset

class SiameseWindowDataset(Dataset):

    """
    Class to create a Siamese window dataset for multimodal error detection.
    """

    def __init__(self, 
                 image_data, 
                 kinematics_data, 
                 pairs_df,
                 train=True,
                 feature_standardization_dict={}):
        
        self.image_data = image_data
        self.kinematics_data = kinematics_data
        #pairs_df = pd.DataFrame(columns=['subject_1', 'gesture_label_1', 'position_1', 'subject_2', 'gesture_label_2', 'position_2', 'label])
        #subject_1 is train and subject_2 is test in case of the test set
        self.pairs_df = pairs_df 
        self.feature_standardization_dict = feature_standardization_dict

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        
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
