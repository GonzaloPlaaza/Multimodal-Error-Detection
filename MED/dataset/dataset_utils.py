import os
import pickle
import pandas as pd
import torch

from torch.utils.data import DataLoader
from .CustomWindowDataset import CustomWindowDataset
from .SiameseWindowDataset import SiameseWindowDataset

def compute_n_frames(fold_data_path:str,
                      csv_file: pd.DataFrame) -> int:
    """
    Compute the number of frames based on the .pkl files in the specified directory.
    Args:
        root_dir (str): Path to the directory containing .pkl files.
    Returns:
        int: Number of frames.
    """
    n_frames = 0
    for file in csv_file['files']:
        pkl_path = os.path.join(fold_data_path, file)
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            try:
                n_frames += data['image_feats'].shape[0]
            except:
                n_frames += data['feature'].shape[0]
        else:
            print(f"Warning: {pkl_path} does not exist.")

    return n_frames


def load_data(fold_data_path:str,
               csv_filename:str,
               video_data_path:str = None) -> tuple:

    """
    Load data from the specified directory and return tensors for image features, 
    kinematics features, gesture labels, error labels, task labels, trial labels, and subject labels.

    Parameters:
    root_dir (str): The directory containing the data files.

    Returns:
    tuple: A tuple containing tensors/dfs for image features, kinematics features, 
           gesture labels, error labels, task labels, trial labels, and subject labels.
    """

    csv_file = pd.read_csv(os.path.join(fold_data_path, csv_filename), header=None, names=['files'])
    if video_data_path is not None:
        n_frames = compute_n_frames(video_data_path, csv_file)
    
    else:
        n_frames = compute_n_frames(fold_data_path, csv_file)

    image_data = torch.empty((n_frames, 2048))
    kinematics_data = torch.empty((n_frames, 26))
    g_labels_data = torch.empty((n_frames, 1))
    e_labels_data = torch.empty((n_frames, 5))
    #task_data = torch.empty((n_frames, 1))
    #trial_data = torch.empty((n_frames, 1))
    subject_data = pd.DataFrame(columns=['subject'], index=range(n_frames))

    frame_index = 0

    for pkl_file in csv_file['files']:
        
        if pkl_file.endswith('.pkl'):
  
            if video_data_path is not None:
                
                pkl_path = os.path.join(video_data_path, pkl_file)
                pkl_path_2 = os.path.join(fold_data_path, pkl_file)

                with open(pkl_path, 'rb') as file:
                    data = pickle.load(file)
                
                with open(pkl_path_2, 'rb') as file:
                    data2 = pickle.load(file)

                n_frames_in_trial = data['feature'].shape[0] #this should be equal except for subjects 4-5Out

                #i. Image features
                image_data[frame_index : frame_index + n_frames_in_trial] = torch.tensor(data['feature'].reshape(n_frames_in_trial, 2048), dtype=torch.float32)

                #ii. Kinematics features
                kinematics_data[frame_index : frame_index + n_frames_in_trial] = data2['kinematics_feats'].clone().detach()

                #iii. Gesture labels
                g_labels_data[frame_index : frame_index + n_frames_in_trial] = torch.tensor(data2['g_labels'].reshape(n_frames_in_trial, 1))

                #iv. Error labels
                e_labels_data[frame_index : frame_index + n_frames_in_trial] = torch.tensor(data2['e_labels'].reshape(n_frames_in_trial, 5), dtype=torch.float32)

                #v. Task labels
                #if pkl_file.startswith("Needle"):
                    #task_data[frame_index : frame_index + n_frames_in_trial] = 0 #Needle_Passing
                
                #elif pkl_file.startswith("Suturing"):
                    #task_data[frame_index : frame_index + n_frames_in_trial] = 1

                #vi. Trial labels
                #Remove .pkl
                pkl_file = pkl_file[:-4]
                #trial_data[frame_index : frame_index + n_frames_in_trial] = int(pkl_file[-1])

                #vii. Subject labels (literally the pkl_file)
                subject = pkl_file
                subject_data.iloc[frame_index : frame_index + n_frames_in_trial] = subject

                frame_index += n_frames_in_trial

            
            
            else:
                
                pkl_path = os.path.join(fold_data_path, pkl_file)
                with open(pkl_path, 'rb') as file:
                    data = pickle.load(file)

                n_frames_in_trial = data['image_feats'].shape[0]

                #i. Image features
                image_data[frame_index : frame_index + n_frames_in_trial] = data['image_feats'].reshape(n_frames_in_trial, 2048).clone().detach()

                #ii. Kinematics features
                kinematics_data[frame_index : frame_index + n_frames_in_trial] = data['kinematics_feats'].clone().detach()

                #iii. Gesture labels
                g_labels_data[frame_index : frame_index + n_frames_in_trial] = torch.tensor(data['g_labels'].reshape(n_frames_in_trial, 1))

                #iv. Error labels
                e_labels_data[frame_index : frame_index + n_frames_in_trial] = data['e_labels'].reshape(n_frames_in_trial, 5).clone().detach()

                #v. Task labels
                #if pkl_file.startswith("Needle"):
                    #task_data[frame_index : frame_index + n_frames_in_trial] = 0 #Needle_Passing
                
                #elif pkl_file.startswith("Suturing"):
                    #task_data[frame_index : frame_index + n_frames_in_trial] = 1

                #vi. Trial labels
                #Remove .pkl
                pkl_file = pkl_file[:-4]
                #trial_data[frame_index : frame_index + n_frames_in_trial] = int(pkl_file[-1])

                #vii. Subject labels (literally the pkl_file)
                subject = pkl_file
                subject_data.iloc[frame_index : frame_index + n_frames_in_trial] = subject

                frame_index += n_frames_in_trial

    #return image_data, kinematics_data, g_labels_data, e_labels_data, task_data, trial_data, subject_data
    return image_data, kinematics_data, g_labels_data, e_labels_data, subject_data



def window_data(image_data, 
                kinematics_data, 
                g_labels_data, 
                e_labels_data, 
                #task_data, 
                #trial_data, 
                subject_data,
                window_size = 10,
                stride = 6):

    """
    Create windows of data.

    The data is of size (n_frames, x) (where x depends on the data type). However, not all the data comes from the same subject (determined by 
    subject_data). Therefore, we need to create windows per subject, and not across subjects. Data is nevertheless ordered, i.e., if the first subject video 
    has 100 frames, the second subject video will start at frame 101.

    Secondly, windows can only be created within the same gesture, as errors are at the gesture level. If a gesture is shorter than window length,
    a window is not created for that gesture. 

    Output: 
    image_windows: shape (n_windows, window_size, 2048)
    kinematics_windows: shape (n_windows, window_size, 26)
    g_labels_windows: shape (n_windows, 1) #no need to explicit gesture labels, as they are the same for all frames in a window
    e_labels_windows: shape (n_windows, 5)
    task_windows: shape (n_windows, 1)
    trial_windows: shape (n_windows, 1)
    subject_windows: shape (n_windows, 1)

    """

    #Determine number of subjects and their position in the matrices
    subjects = subject_data['subject'].unique()
    subject_indices = {subject: subject_data[subject_data['subject'] == subject].index for subject in subjects}
    n_windows_total = 0

    image_windows = []
    kinematics_windows = []
    g_labels_windows = []
    e_labels_windows = []
    #task_windows = []
    #trial_windows = []
    subject_windows = []

    #Iterate over subjects
    for subject in subjects:
        subject_index = subject_indices[subject]
        n_frames_subject = len(subject_index)

        #Iterate over gestures
        gesture_indices = g_labels_data[subject_index].nonzero(as_tuple=True)[0]
        start_idx = gesture_indices[0].item()

        while start_idx < n_frames_subject - window_size:

            #1. Each gesture type must be at least of window_size to create a window.
            end_idx = start_idx + window_size

            #2. A window cannot contain more than two gesture types, as errors are at the gesture level.
            current_gesture = g_labels_data[subject_index[start_idx]].item()
            end_gesture = g_labels_data[subject_index[end_idx - 1]].item()

            if end_idx > n_frames_subject or current_gesture != end_gesture:
                #print(f"Skipping gesture {current_gesture} at subject {subject}, start gesture {g_labels_data[subject_index[start_idx]].item()} \
                #      (frame: {start_idx}) to end gesture {end_gesture} (frame: {end_idx - 1})")
                start_idx += 1
                continue

            #3. Create window
            image_windows.append(image_data[subject_index[start_idx:end_idx]])
            kinematics_windows.append(kinematics_data[subject_index[start_idx:end_idx]])
            g_labels_windows.append(g_labels_data[subject_index[start_idx]])
            e_labels_windows.append(e_labels_data[subject_index[start_idx]])
            #task_windows.append(task_data[subject_index[start_idx]])
            #trial_windows.append(trial_data[subject_index[start_idx]])
            subject_windows.append(subject)
            
            #4. Move to next window
            start_idx += stride
            n_windows_total += 1

    #Convert lists to tensors
    image_windows = torch.stack(image_windows)
    kinematics_windows = torch.stack(kinematics_windows)
    g_labels_windows = torch.tensor(g_labels_windows).reshape(-1, 1)
    e_labels_windows = torch.stack(e_labels_windows)
    #task_windows = torch.tensor(task_windows).reshape(-1, 1)
    #trial_windows = torch.tensor(trial_windows).reshape(-1, 1)
    subject_windows = pd.DataFrame(subject_windows, columns=['subject'])

    #Return the windows
    return (image_windows, 
            kinematics_windows, 
            g_labels_windows, 
            e_labels_windows, 
            #task_windows, 
            #trial_windows, 
            subject_windows)



def compute_window_size_stride(frequency: int = 30) -> tuple:

    """
    Compute the window size and stride.
    Original data is at 30Hz. We want windows of 2s, and strides 1.33s.
    For instance, at a frequency of 15Hz, window size is 30 and stride is 20.

    Args:
        frequency (int): The frequency of the data.

    Returns:
        tuple: A tuple containing the window size and stride.
    """

    window_size = int(2 * frequency)  # 2 seconds window
    stride = int(4/3 * frequency)      # 1.33 seconds stride

    return (window_size, stride)


def load_siamese_pairs(pairs_df: pd.DataFrame,
                       image_train_data: torch.Tensor,
                        kinematics_train_data: torch.Tensor,  
                        image_data_test: torch.Tensor = None,
                        kinematics_data_test: torch.Tensor = None,
                        subject_data_test: pd.DataFrame = None,
                        train: bool = True,
                        exp_kwargs: dict = {},
                        window_size: int = 30
                        ) -> tuple:
        
        """
        Create pairs of windows for Siamese networks.

        The input dfs are of the form: 
            train_pairs_df = pd.DataFrame(columns=['subject_1', 'gesture_label_1', 'position_1', 'subject_2', 'gesture_label_2', 'position_2', 'label])

        We want to create a pre-defined set number of pairs (exp_kwargs['n_pairs']) for training.
        Given the data is imbalanced (especially towards the dissimilar/positive pairs), we will create a balanced set of pairs.

        Returns:

            tuple: A tuple containing the pairs of images, pairs of kinematics, labels, and a DataFrame with the pairs information.
        """
        
        labels = []

        if train:
            n_pairs = exp_kwargs['n_pairs']

            #Create a balanced set of pairs based on the computed class balance
            df_0 = pairs_df[pairs_df['label'] == 0].sample(n=n_pairs//2, replace=True, random_state=42)
            df_1 = pairs_df[pairs_df['label'] == 1].sample(n=n_pairs//2, replace=True, random_state=42)
            new_pairs_df = pd.concat([df_0, df_1], ignore_index=True)
        
        else: 
            #For testing, we use all the pairs in the dataframe
            new_pairs_df = pairs_df.copy()

        #Create image pairs, kinematics pairs, position (#window) pairs, and labels
        image_pairs = torch.empty((len(new_pairs_df), 2, window_size, 2048)) #n_pairs, 2 pairs, 30 frames, 2048 features
        kinematics_pairs = torch.empty((len(new_pairs_df), 2, window_size, 26)) #n_pairs, 2 pairs, 30 frames, 26 features
        labels = torch.empty((len(new_pairs_df), 1))
        
        for i, row in new_pairs_df.iterrows():
            #Get the positions of the pairs
            pos_1 = row['position_1']
            pos_2 = row['position_2']

            #Get the label
            label = row['label']

            #Get the image and kinematics data for the pairs
            if train:
                image_pairs[i, 0] = image_train_data[pos_1]
                image_pairs[i, 1] = image_train_data[pos_2]
                kinematics_pairs[i, 0] = kinematics_train_data[pos_1]
                kinematics_pairs[i, 1] = kinematics_train_data[pos_2]

            else:
                #For test data, we use the train data (first position) and test data (second position)
                image_pairs[i, 0] = image_train_data[pos_1]
                image_pairs[i, 1] = image_data_test[pos_2]
                kinematics_pairs[i, 0] = kinematics_train_data[pos_1]
                kinematics_pairs[i, 1] = kinematics_data_test[pos_2]

            labels[i] = label

        return (image_pairs,
                kinematics_pairs,
                labels,
                new_pairs_df)



def load_and_window(fold_data_path:str,
                    window_size:int = 30,
                    stride:int = 20,
                    video_data_path:str = None) -> tuple:
    
    """
    Load the data from the specified directory and window it.

    Args:
        fold_data_path (str): Path to the data for the fold.
        window_size (int): Size of the window for the data.
        stride (int): Stride for the sliding window.

    Returns:
        tuple: A tuple containing the windowed data for training and testing.
    """


    image_train, kinematics_train, g_labels_train, e_labels_train, subject_train = load_data(fold_data_path, 
                                                                                             'train.csv',
                                                                                             video_data_path=video_data_path)
    
    image_test, kinematics_test, g_labels_test, e_labels_test, subject_test = load_data(fold_data_path, 
                                                                                        'test.csv',
                                                                                        video_data_path=video_data_path)
    
    print("Windowing data...")
    image_data, kinematics_data, g_labels_data, e_labels_data, subject_data = window_data(image_train,
                                                                                            kinematics_train,
                                                                                            g_labels_train,
                                                                                            e_labels_train,
                                                                                            subject_train,
                                                                                            window_size= window_size,
                                                                                            stride=stride)
    
    image_test_data, kinematics_test_data, g_labels_test_data, e_labels_test_data, subject_test_data = window_data(image_test,
                                                                                                                    kinematics_test,
                                                                                                                    g_labels_test,
                                                                                                                    e_labels_test,
                                                                                                                    subject_test,
                                                                                                                    window_size=window_size,
                                                                                                                    stride=stride)


    return image_data, kinematics_data, g_labels_data, e_labels_data, subject_data, \
              image_test_data, kinematics_test_data, g_labels_test_data, e_labels_test_data, subject_test_data


def retrieve_dataloaders_window(fold_data_path:str,
                        exp_kwargs:dict,
                        window_size:int = 30,
                        stride:int = 20,
                        video_data_path:str = None
                         ):
    

    """
    Retrieve the dataloaders for the experiment.

    Args:
        fold_data_path (str): Path to the data for the fold.
        exp_kwargs (dict): Dictionary containing experiment parameters.
        window_size (int): Size of the window for the data.
        stride (int): Stride for the sliding window.  

    Returns:
        tuple: A tuple containing the train and test dataloaders.
    """

    #a.1. Load data / b. Window data and standardize features
    image_data, kinematics_data, g_labels_data, e_labels_data, subject_data, \
              image_test_data, kinematics_test_data, g_labels_test_data, e_labels_test_data, subject_test_data = load_and_window(fold_data_path,
                                                                                                                    video_data_path=video_data_path,
                                                                                                                    window_size=window_size,
                                                                                                                    stride=stride)

    #a.2. Create powerset of error labels, i.e., all possible combinations of error labels.
    #i. Train data
    e_labels_data, mask_positions_needle_drop = powerset_error_labels(e_labels_data=e_labels_data, delete_ND = exp_kwargs['delete_ND'])

    #ii. Test data
    e_labels_test_data, mask_positions_needle_drop_test = powerset_error_labels(e_labels_data=e_labels_test_data, 
                                                                                    delete_ND=exp_kwargs['delete_ND'])
    
    #iii. Delete positions where Needle Drop is present
    if exp_kwargs['delete_ND']:
        image_data = image_data[~mask_positions_needle_drop]
        kinematics_data = kinematics_data[~mask_positions_needle_drop]
        g_labels_data = g_labels_data[~mask_positions_needle_drop]
        e_labels_data = e_labels_data[~mask_positions_needle_drop]
        subject_data = subject_data[~mask_positions_needle_drop.numpy()]

        image_test_data = image_test_data[~mask_positions_needle_drop_test]
        kinematics_test_data = kinematics_test_data[~mask_positions_needle_drop_test]
        g_labels_test_data = g_labels_test_data[~mask_positions_needle_drop_test]
        e_labels_test_data = e_labels_test_data[~mask_positions_needle_drop_test]
        subject_test_data = subject_test_data[~mask_positions_needle_drop_test.numpy()]
    
    
    #b. Load feature standardization parameters
    image_mean = torch.load(os.path.join(fold_data_path, 'mean_features.pth'))
    image_std = torch.load(os.path.join(fold_data_path, 'std_features.pth'))
    kinematics_mean = torch.load(os.path.join(fold_data_path, 'mean_kinematics.pth'))
    kinematics_std = torch.load(os.path.join(fold_data_path, 'std_kinematics.pth'))
    feature_standardization_dict = {
        'image': {'mean': image_mean, 'std': image_std},
        'kinematics': {'mean': kinematics_mean, 'std': kinematics_std}
    }

    
    #c. Create datasets and dataloaders
    if exp_kwargs['siamese']: #Siamese networks require a different type of dataset.
        print("Creating Siamese pairs...")

        pairs_df_train = pd.read_csv(os.path.join(fold_data_path, 'train_pairs.csv'))
        pairs_df_test = pd.read_csv(os.path.join(fold_data_path, f'test_pairs_{exp_kwargs["n_comparisons"]}.csv'))

        image_pairs_train, kinematics_pairs_train, labels_train, train_pairs_df = load_siamese_pairs(pairs_df=pairs_df_train,
                                                                                                    image_train_data= image_data,
                                                                                                    kinematics_train_data= kinematics_data, 
                                                                                                    train=True,
                                                                                                    window_size=window_size,
                                                                                                    exp_kwargs=exp_kwargs)
        
        image_pairs_test, kinematics_pairs_test, labels_test, test_pairs_df = load_siamese_pairs(pairs_df=pairs_df_test,
                                                                                                image_train_data=image_data,
                                                                                                kinematics_train_data=kinematics_data,
                                                                                                image_data_test=image_test_data,
                                                                                                kinematics_data_test=kinematics_test_data,
                                                                                                subject_data_test=subject_test_data,
                                                                                                train=False,
                                                                                                window_size=window_size,
                                                                                                exp_kwargs=exp_kwargs)
                                                                                            
        train_dataset = SiameseWindowDataset(image_data=image_pairs_train,
                                            kinematics_data=kinematics_pairs_train, 
                                            e_labels=labels_train,
                                            pairs_df=train_pairs_df,
                                            train=True,
                                            feature_standardization_dict=feature_standardization_dict)
        
        test_dataset = SiameseWindowDataset(image_data=image_pairs_test,
                                            kinematics_data=kinematics_pairs_test, 
                                            e_labels=labels_test,
                                            pairs_df=test_pairs_df,
                                            train=False,
                                            feature_standardization_dict=feature_standardization_dict)
 
           
    else:
        print("Creating datasets and dataloaders...")
        train_dataset = CustomWindowDataset(image_data, 
                                            kinematics_data, 
                                            g_labels_data, 
                                            e_labels_data, 
                                            #task_train, 
                                            #trial_train, 
                                            subject_data,
                                            feature_standardization_dict=feature_standardization_dict)
        
        test_dataset = CustomWindowDataset(image_test_data,
                                            kinematics_test_data, 
                                            g_labels_test_data, 
                                            e_labels_test_data, 
                                            #task_test, 
                                            #trial_test, 
                                            subject_test_data,
                                            feature_standardization_dict=feature_standardization_dict)
    
    train_dataloader = DataLoader(train_dataset, batch_size=exp_kwargs["batch_size"], shuffle=True, generator= torch.Generator().manual_seed(42))
    test_dataloader = DataLoader(test_dataset, batch_size=exp_kwargs["batch_size"], shuffle=False, generator=torch.Generator().manual_seed(42))
    print(f"Number of training windows: {len(train_dataset)}")
    print(f"Number of testing windows: {len(test_dataset)}")

    return train_dataloader, test_dataloader


def create_siamese_pairs(fold_data_path:str,
                         video_data_path:str = None,
                         window_size:int = 30,  
                         stride:int = 20,
                         train=True,
                         position_to_instance_df: pd.DataFrame = None,
                         exp_kwargs: dict = {}
                        ) -> tuple:
        
        """
        Creates a .csv file with pairs of windows (i.e., positions in the data vectors) and their labels for Siamese networks.

        Args:
            g_labels_data (torch.Tensor): Tensor of gesture labels.
            e_labels_data (torch.Tensor): Tensor of error labels.
            subject_data (pd.DataFrame): DataFrame containing subject labels.
            train (bool): Whether the data is for training or testing.
            g_labels_data_test (torch.Tensor, optional): Tensor of gesture labels for the test set. Defaults to None.
            e_labels_data_test (torch.Tensor, optional): Tensor of error labels for the test set. Defaults to None.
            subject_data_test (pd.DataFrame, optional): DataFrame containing subject labels for the test set. Defaults to None.
            exp_kwargs (dict): Dictionary containing experiment parameters, such as whether to use binary error labels.


        Returns:

            tuple: A tuple containing the pairs of images, pairs of kinematics, labels, and a DataFrame with the pairs information.
        """

        #Load data
        image_data_train, kinematics_data_train, g_labels_data_train, e_labels_data_train, subject_data_train, \
            image_data_test, kinematics_data_test, g_labels_data_test, e_labels_data_test, subject_data_test = load_and_window(
                    video_data_path=video_data_path,
                    fold_data_path=fold_data_path,
                    window_size=window_size,
                    stride=stride)
    
        
        torch.manual_seed(42)

        if exp_kwargs['binary_error']:
            #If binary error, we only consider the last error label (index 4)
            e_labels_data_train = e_labels_data_train[:, 4]
            if not train:
                e_labels_data_test = e_labels_data_test[:, 4]

        #Empty list to store the subject, gesture labels, and position in vector of the training pairs
        if train:
            train_pairs_df = []
            
        else:
            test_pairs_df = []

        n_pairs = 0
        create = False
        if train:

            #The TRAINING data is paired in the following way.
            #a. When two windows are not erroneous, they are labeled as 0.
            #b. When one window is erroneous and the other is not, they are paired and labeled as 1.
            #c. An instance is defined as a gesture performed by a subject, i.e., a sequence of frames with the same gesture label.
            #To create a pair, they must not belong to the same gesture **instance** or be contiguous.

            n_windows = len(g_labels_data_train)
            instance_count_1 = 0
            for i in range(n_windows):

                #Subject change --> sreset instance_count_1
                if i != 0: #to avoid index error
                    if subject_data_train['subject'][i] != subject_data_train['subject'][i - 1]:
                        instance_count_1 = 0 
                    
                    else:
                        if g_labels_data_train[i] != g_labels_data_train[i - 1]:
                            #Gesture change --> sum instance_count_1
                            instance_count_1 += 1

                instance_count_2 = 1 #start always from the first instance of the subject
                for j in range(i + 2, n_windows): #+2 to ensure non-contiguity
                    #Update subject count:
                    #Subject change --> sum subject_count_2 and reset instance_count_2
                    if j != i + 2: #to avoid looking at i + 1 window which we are not interested in.
                        if subject_data_train['subject'][j] != subject_data_train['subject'][j - 1]:
                            instance_count_2 = 0
                        else:
                            if g_labels_data_train[j] != g_labels_data_train[j - 1]:
                                #Gesture change --> sum instance_count_2
                                instance_count_2 += 1
                    
                    #Ensure that the pairs do not belong to the same gesture instance or are contiguous
                    #1st case: is it the same subject?
                    if subject_data_train['subject'][i] == subject_data_train['subject'][j]:
                        
                        #2nd case: is it the same gesture?
                        if g_labels_data_train[i] == g_labels_data_train[j]:
                            
                            #3rd case: if it is the same gesture, do they belong to the same gesture instance?
                            #Check if in between the windows there is a different gesture
                            for k in range(i + 1, j):
                                if g_labels_data_train[k] != g_labels_data_train[i]: #not the same gesture
                                    create = True
                                    break

                        else: #not the same gesture
                            create = True
                            
                    else: #not the same subject
                        create = True

                    if create:

                        #Labeling
                        if e_labels_data_train[i] == 0 and e_labels_data_train[j] == 0:
                            label = 0
                        
                        elif e_labels_data_train[i] == 1 and e_labels_data_train[j] == 0 or \
                            e_labels_data_train[i] == 0 and e_labels_data_train[j] == 1:
                            label = 1

                        else: continue #Both windows are erroneous, skip this pair  

                        #Add a new row in df at loc n_pairs
                        train_pairs_df.append({
                            'subject_1': subject_data_train['subject'][i],
                            'gesture_label_1': g_labels_data_train[i].item(),
                            'position_1': i,
                            'instance_1': instance_count_1,
                            'subject_2': subject_data_train['subject'][j],
                            'gesture_label_2': g_labels_data_train[j].item(),
                            'position_2': j,
                            'instance_2': instance_count_2,
                            'label': label
                        })

                        n_pairs += 1

                        create = False #Reset for the next pair

                if i % 200 == 0:
                    print(f"Processed {i}/{n_windows} training windows.")

        else: #Testing data

            #The TESTING data is paired in the following way.
            #Each test window is paired with "n_comparisons" non-erroneous (0) training set windows.

            #Find all non-erroneous training windows
            train_indices = (e_labels_data_train == 0).nonzero(as_tuple=True)[0]
            train_indices = train_indices.tolist()
            train_pairs_df = None

            instance_count_2 = 0
            for i in range(len(g_labels_data_test)):
                
                #Subject change --> reset instance_count_2
                if i != 0: #to avoid index error
                    if subject_data_test['subject'][i] != subject_data_test['subject'][i - 1]:
                        instance_count_2 = 0 
                    
                    else:
                        if g_labels_data_test[i] != g_labels_data_test[i - 1]:
                            #Gesture change --> sum instance_count_2
                            instance_count_2 += 1
                
                #Sample n_comparisons random training windows
                n_comparisons = exp_kwargs['n_comparisons']
                if len(train_indices) < n_comparisons:
                    print(f"Warning: Not enough training windows to compare with test window {i}.")
                    continue
                
                sampled_indices = torch.randperm(len(train_indices))[:n_comparisons]
                train_indices_chosen = [train_indices[idx] for idx in sampled_indices]
                
                for j in train_indices_chosen:
                    #Create pair
                    #Labeling
                    if e_labels_data_test[i] == 0 and e_labels_data_train[j] == 0:
                        label = 0
                    
                    elif e_labels_data_test[i] == 1 and e_labels_data_train[j] == 0:
                        label = 1

                    else: continue

                    try:
                        instance_1 = position_to_instance_df[position_to_instance_df['position'] == j]['instance'].values[0]
                    
                    except IndexError:
                        instance_1 
                        continue

                    test_pairs_df.append({
                            'subject_1': subject_data_train['subject'][j],
                            'gesture_label_1': g_labels_data_train[j].item(),
                            'position_1': j,
                            'instance_1': instance_1,
                            'subject_2': subject_data_test['subject'][i],
                            'gesture_label_2': g_labels_data_test[i].item(),
                            'position_2': i,
                            'instance_2': instance_count_2,
                            'label': label
                        })

                if i % 200 == 0:
                    print(f"Processed {i}/{len(g_labels_data_test)} test windows.")

        print(f"Number of pairs created: {len(train_pairs_df) if train else len(test_pairs_df)}")

        if train:
            train_pairs_df = pd.DataFrame(train_pairs_df)
            train_pairs_df['label'] = train_pairs_df['label'].astype(int)
            position_to_instance_df_1 = train_pairs_df[['position_1', 'instance_1']].drop_duplicates()
            position_to_instance_df_2 = train_pairs_df[['position_2', 'instance_2']].drop_duplicates()

            df1 = position_to_instance_df_1.rename(columns={'position_1': 'position', 'instance_1': 'instance'})
            df2 = position_to_instance_df_2.rename(columns={'position_2': 'position', 'instance_2': 'instance'})

            position_to_instance_df = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
                                                      
            return train_pairs_df, position_to_instance_df

        else:
            test_pairs_df = pd.DataFrame(test_pairs_df)
            test_pairs_df['label'] = test_pairs_df['label'].astype(int)
            return test_pairs_df


def powerset_error_labels(e_labels_data: torch.tensor,
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
                
