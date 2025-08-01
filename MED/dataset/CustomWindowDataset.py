from torch.utils.data import Dataset

class CustomWindowDataset(Dataset):

    """
    Custom dataset to load data and arrange it in windows.
    Input:
        image_data: Tensor of image features. Shape: n_samples x 2048
        kinematics_data: Tensor of kinematics features. Shape: n_samples x 26
        g_labels_data: Tensor of gesture labels. Shape: n_samples x 1   
        e_labels_data: Tensor of error labels. Shape: n_samples x 5
        task_data: Tensor of task labels. Shape: n_samples x 1
        trial_data: Tensor of trial labels. Shape: n_samples x 1
        subject_data: DataFrame of subject labels. Shape: n_samples x 1
        window_size: Size of the window for arranging data.   
        stride: Stride for the sliding window.  

    Returns: None

    """

    def __init__(self, 
                 image_data, 
                 kinematics_data, 
                 g_labels_data, 
                 e_labels_data, 
                 #task_data, 
                 #trial_data, 
                 subject_data,
                 feature_standardization_dict={}):
        
        self.image_data = image_data
        self.kinematics_data = kinematics_data
        self.g_labels_data = g_labels_data
        self.e_labels_data = e_labels_data
        self.subject_data = subject_data
        self.feature_standardization_dict = feature_standardization_dict  
        #self.task_data = task_data
        #self.trial_data = trial_data

        #Compute error class balance
        self.binary_error_distribution = (1 - self.e_labels_data[:, -1].sum() / len(self.e_labels_data), 
                                          self.e_labels_data[:, -1].sum() / len(self.e_labels_data))

    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        
        image = self.image_data[idx]
        kinematics = self.kinematics_data[idx]
        
        for key, value in self.feature_standardization_dict.items():
            if key == 'image':
                image = (image - value['mean']) / value['std']
            elif key == 'kinematics':
                kinematics = (kinematics - value['mean']) / value['std']  
                
        
        g_labels = self.g_labels_data[idx]
        e_labels = self.e_labels_data[idx]
        subject = self.subject_data.iloc[idx]['subject']

        #task = self.task_data[idx]
        #trial = self.trial_data[idx]

        return (image, 
                kinematics, 
                g_labels, 
                e_labels,  
                subject)
                #task, 
                #trial,