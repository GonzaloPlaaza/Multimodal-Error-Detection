import cv2
import os
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import torch 
from torch import nn
from torchvision import transforms
import torchvision.models as models
from torchvision import models, transforms


def convert_videos_to_frames(video_folder, output_folder, frequency=30):
    '''
    This function converts .avi videos in the specified folder into frames saved as .png images.

    For each .avi file, we create a folder named after the video which contains the frames. For instance,

    Name: raw/Needle_Passing/Needle_Passing_B001_capture2.avi
    Output: xHz/Needle_Passing/Needle_Passing_B001/{0000.png, Needle_Passing_B001/capture2/0001.png, etc.}
    
    Args:
    video_folder (str): Path to the folder containing .avi videos.
    output_folder (str): Path to the folder where frames will be saved.
    frequency (int): Frequency at which to extract frames. Default is 30 Hz.

    Returns:
    None: The function saves frames as .png images in the specified output folder.
    
    '''

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if frequency > 30 or frequency < 1:
        raise ValueError("Frequency must be between 1 and 30 Hz.")

    for filename in os.listdir(video_folder):


    
        if filename.endswith('.avi') and "capture2" in filename:
            print(f"Processing {filename}...")
            video_path = os.path.join(video_folder, filename)
            video_capture = cv2.VideoCapture(video_path)
            frame_count = 1 # Start frame count from 1

            filename_parts = filename.split('_')
            filename_parts = filename_parts[:-1]  # Remove the last part (e.g., 'capture2.avi')
            final_filename = '_'.join(filename_parts)

            video_output_folder = os.path.join(output_folder, final_filename)
            if not os.path.exists(video_output_folder):
                os.makedirs(video_output_folder)
            
            while True:
                ret, frame = video_capture.read()

                if frequency != 30:
                    #Original data is at 30Hz. If frequency is 5, then we keep one in every 6 frames.
                    if frame_count % (30 / frequency) != 1:
                        frame_count += 1
                        continue

                if not ret:
                    break
                
                frame_filename = f"{frame_count:04d}.png"
                frame_path = os.path.join(video_output_folder, frame_filename)

                #Resize to 240x240
                frame = cv2.resize(frame, (240, 240))

                #Center-crop at 224x224
                height, width, _ = frame.shape
                start_x = (width - 224) // 2
                start_y = (height - 224) // 2
                frame = frame[start_y:start_y + 224, start_x:start_x + 224]

                cv2.imwrite(frame_path, frame)
                print(f"Saved frame {frame_count} to {frame_path}")
                frame_count += 1

            video_capture.release()

        print("Conversion complete.")



def rotation_matrix_to_euler_angles(R):

    """
    Convert a rotation matrix to Euler angles (roll, pitch, yaw).
    The input R is a 3x3 rotation matrix.

    Args:
    R (numpy.ndarray): A 3x3 rotation matrix.

    Returns:
    tuple: A tuple containing the Euler angles (roll, pitch, yaw) in radians.
    """
    assert R.shape == (3, 3), "Input must be a 3x3 rotation matrix."
    
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    
    return x, y, z


def process_kinematics(input_folder, output_folder, frequency = 30):

    '''

        kinematic variables
        1-3    (3) : Master left tooltip xyz                    
        4-12   (9) : Master left tooltip R    
        13-15  (3) : Master left tooltip trans_vel x', y', z'   
        16-18  (3) : Master left tooltip rot_vel                
        19     (1) : Master left gripper angle                  
        20-38  (19): Master right
        39-41  (3) : Slave left tooltip xyz
        42-50  (9) : Slave left tooltip R
        51-53  (3) : Slave left tooltip trans_vel x', y', z'   
        54-56  (3) : Slave left tooltip rot_vel
        57     (1) : Slave left gripper angle                   
        58-76  (19): Slave right


        We are interested in the slave variables. Moreover, we will transform the rotational matrix R into Euler angles (roll, pitch, yaw).
        This results in 26 total variables.

    Args:
    input_folder (str): Path to the folder containing kinematics data files.
    output_folder (str): Path to the folder where processed data will be saved.
    frequency (int): Frequency at which to sample the data. Default is 30 Hz.

    Returns:
    None: The function saves processed data as .csv files in the specified output folder.
'''
    kinematic_columns_original = [ "Slave_left_tooltip_x", "Slave_left_tooltip_y", "Slave_left_tooltip_z",
                     "Slave_left_tooltip_R11", "Slave_left_tooltip_R12", "Slave_left_tooltip_R13",
                        "Slave_left_tooltip_R21", "Slave_left_tooltip_R22", "Slave_left_tooltip_R23",
                        "Slave_left_tooltip_R31", "Slave_left_tooltip_R32", "Slave_left_tooltip_R33",
                        "Slave_left_tooltip_x'", "Slave_left_tooltip_y'", "Slave_left_tooltip_z'",
                        "Slave_left_tooltip_rot_vel_x", "Slave_left_tooltip_rot_vel_y", "Slave_left_tooltip_rot_vel_z",
                        "Slave_left_gripper_angle",
                        "Slave_right_tooltip_x", "Slave_right_tooltip_y", "Slave_right_tooltip_z",
                        "Slave_right_tooltip_R11", "Slave_right_tooltip_R12", "Slave_right_tooltip_R13",
                        "Slave_right_tooltip_R21", "Slave_right_tooltip_R22", "Slave_right_tooltip_R23",
                        "Slave_right_tooltip_R31", "Slave_right_tooltip_R32", "Slave_right_tooltip_R33",
                        "Slave_right_tooltip_x'", "Slave_right_tooltip_y'", "Slave_right_tooltip_z'",
                        "Slave_right_tooltip_rot_vel_x", "Slave_right_tooltip_rot_vel_y", "Slave_right_tooltip_rot_vel_z",
                        "Slave_right_gripper_angle"]

    kinematic_columns_processed = ["Slave_left_tooltip_x", "Slave_left_tooltip_y", "Slave_left_tooltip_z",
                                    "Slave_left_tooltip_roll", "Slave_left_tooltip_pitch", "Slave_left_tooltip_yaw",
                                    "Slave_left_tooltip_x'", "Slave_left_tooltip_y'", "Slave_left_tooltip_z'",
                                    "Slave_left_tooltip_rot_vel_x", "Slave_left_tooltip_rot_vel_y", "Slave_left_tooltip_rot_vel_z",
                                    "Slave_left_gripper_angle",
                                    "Slave_right_tooltip_x", "Slave_right_tooltip_y", "Slave_right_tooltip_z",
                                    "Slave_right_tooltip_roll", "Slave_right_tooltip_pitch", "Slave_right_tooltip_yaw",
                                    "Slave_right_tooltip_x'", "Slave_right_tooltip_y'", "Slave_right_tooltip_z'",
                                    "Slave_right_tooltip_rot_vel_x", "Slave_right_tooltip_rot_vel_y", "Slave_right_tooltip_rot_vel_z",
                                    "Slave_right_gripper_angle"]

    
    for filename in os.listdir(input_folder):

        print(f"Processing {filename}...")
        
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            data = pd.read_csv(file_path, sep='\s+', header=None, engine='python')
        
            #Select columns 39 to 76
            kinematics_data = data.iloc[:, 38:76]
            kinematics_data.columns = kinematic_columns_original    

            #Add data that is not from R into the processed dataframe
            kinematics_data_processed = pd.DataFrame(columns=kinematic_columns_processed)
            for col in kinematic_columns_processed:
                if col in kinematics_data.columns:
                    kinematics_data_processed[col] = kinematics_data[col]
                                                        

            #Convert R matrices to Euler angles
            for side in ["left", "right"]:
                R_side = kinematics_data[[
                    f"Slave_{side}_tooltip_R11", f"Slave_{side}_tooltip_R12", f"Slave_{side}_tooltip_R13",
                    f"Slave_{side}_tooltip_R21", f"Slave_{side}_tooltip_R22", f"Slave_{side}_tooltip_R23",
                    f"Slave_{side}_tooltip_R31", f"Slave_{side}_tooltip_R32", f"Slave_{side}_tooltip_R33"
                ]].values

                euler_angles_side = np.array([rotation_matrix_to_euler_angles(R.reshape(3, 3).astype(np.float64)) for R in R_side])
                kinematics_data_processed[f"Slave_{side}_tooltip_roll"] = euler_angles_side[:, 0]
                kinematics_data_processed[f"Slave_{side}_tooltip_pitch"] = euler_angles_side[:, 1]
                kinematics_data_processed[f"Slave_{side}_tooltip_yaw"] = euler_angles_side[:, 2]

            if frequency != 30:
                #Original data is at 30Hz. If frequency is 5, then we keep one in every 6 frames.
                kinematics_data_processed_final = kinematics_data_processed.iloc[::(30//frequency), :]
                
                try:
                    #Try to add the last frame if it exists

                    print(kinematics_data_processed.iloc[kinematics_data_processed.index[-1] + 6, :])
                    kinematics_data_processed_final = pd.concat([kinematics_data_processed_final, kinematics_data_processed.iloc[kinematics_data_processed_final.index[-1] + 6, :]])
                    
                except IndexError:
                    #If it does not exist, do nothing
                    pass

                kinematics_data_processed = kinematics_data_processed_final

            #Sum 1 to the frame count to match the frame numbering in the video frames
            kinematics_data_processed.index += 1

            #Save processed data as .csv
            output_filename = filename.replace('.txt', '.csv')
            output_path = os.path.join(output_folder, output_filename)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            kinematics_data_processed.to_csv(output_path, index_label='frame', float_format='%.6f')




def process_gestures(input_folder, output_folder, frequency=30):

    '''
    In each transcription file, we have the following format:
    Column 1: Frame start of gesture
    Column 2: Frame end of gesture
    Column 3: Gesture (G1, G2, etc.)

    We want to create a vector of gestures (1,1,1,2,2,2,3,3,3,3,3,4,4,4,4,4,4) where each number corresponds to the gesture at that frame.

    Args:
    input_folder (str): Path to the folder containing gesture transcription files.
    output_folder (str): Path to the folder where processed gesture vectors will be saved.
    frequency (int): Frequency at which to sample the data. Default is 30 Hz.

    Returns:
    None: The function saves processed gesture vectors as .npy files in the specified output folder.

    '''

    for filename in os.listdir(input_folder):

        print(f"Processing {filename}...")
        
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            gestures_data = pd.read_csv(file_path, sep='\s+', header=None, names=['start', 'end', 'gesture'])
            output_filename = filename.replace('.txt', '.npy')# Save gesture vector as .npy file
            
            # Create a vector of gestures
            gesture_vector = []
            if frequency == 30:
                for _, row in gestures_data.iterrows():
                    
                    gesture_int = int(row['gesture'].replace('G', ''))
                    gesture_vector.extend([gesture_int] * (row['end'] - row['start'] + 1))

                #Check if length of gesture_vector matches the last end frame - first start frame + 1
                if len(gesture_vector) != (gestures_data['end'].max() - gestures_data['start'].min() + 1):
                    print(f"Warning: Length of gesture vector does not match expected length for {filename}")

            
            else:
                #Open kinematics data and check that the length of the gesture vector matches the number of frames
                kinematics_path = os.path.join(output_folder.replace('gestures', 'kinematics'), output_filename.replace('.npy', '.csv'))
                kinematics_data = pd.read_csv(kinematics_path)
                
                #Keep gestures of frames that are present in the kinematics data
                frame_counter = 0
                for frame in kinematics_data["frame"]:
                    #Find the gesture for this frame
                    gesture_row = gestures_data[(gestures_data['start'] <= frame) & (gestures_data['end'] >= frame)]
                    if not gesture_row.empty:
                        gesture_int = int(gesture_row['gesture'].values[0].replace('G', ''))
                        gesture_vector.append(gesture_int)
                    else:
                        #If no gesture is found, append 0
                        frame_counter += 1
                    
                #Check that the length of the gesture vector matches the number of frames in the kinematics data
                if len(gesture_vector) + frame_counter != len(kinematics_data):
                    print(f"Warning: Length of gesture vector does not match number of frames in \
                          kinematics data for {filename}. Expected {len(kinematics_data)}, got {len(gesture_vector) + frame_counter}.")
            
        
            output_path = os.path.join(output_folder, output_filename)
            np.save(output_path, np.array(gesture_vector))

            #Check that the file contains integers
            gesture_vector = np.load(output_path)
            if not np.issubdtype(gesture_vector.dtype, np.integer):
                print(f"Warning: {output_filename} does not contain integers.")



def process_errors(folder_errors,
                    output_folder,
                    task_type,
                    error_dict,
                    transcription_folder,
                    frequency=30, 
                    output_folder_kinematics= None):
    '''
    Process errors for Needle Passing and Suturing tasks.

    For each trial, we need a dataframe with the following columns:

    Frame, Out_Of_View, Needle_Drop, Multiple_Attempts, Needle_Position, Error (each of the errors with 0 or 1)

    For each trial (Needle_Passing_B001, Suturing_B001, etc.), we will create a .csv file with the error labels.

    The data is in the following format:

    ExecProc_Error_Analysis-main/Error_Labels/
    Consensus_error_labels_needle_passing/
        Error_specific/
            G{i}_{name of error}.csv/
                Columns: name, label_err1_nor0
                where name is trial (Needle_Passing_B001) + _startframe + _endframe + .avi

        error_NP_G{i}.csv/
            Same organization of the .csv
        
    Consensus_error_labels_suturing/
        ...

    We will iterate through the trials, generating an empty dataframe with the columns mentioned above, and the rows extracted from the transcription files 
    (min of start column and max of end column). Then, we will iterate through the errors. For each error, we will then iterate throgh the .csv files, and
    will try to find the matching trial in the "name" column of each .csv. If found, then the corresponding frames will be filled with the error label (0 or 1).


    '''

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #a. Iterate through the transcription files in the input folder
    for filename in os.listdir(transcription_folder):

        #Open file and retrieve start and end frames
        file = os.path.join(transcription_folder, filename)
        if not filename.endswith('.txt'):
            continue
        
        print(f"Processing {filename}...")

        #Read the transcription file
        data = pd.read_csv(file, sep='\s+', header=None, names=['start', 'end', 'gesture'])
        start_frame = data['start'].min()
        end_frame = data['end'].max()
        index = range(start_frame, end_frame + 1)  # Include end frame

        #Create trial name from filename
        trial_name = filename.replace('.txt', '')

        #Create empty dataframe with columns: Frame, Out_Of_View, Needle_Drop, Multiple_Attempts, Needle_Position, Error
        error_columns = ['frame', 'Out_Of_View', 'Needle_Drop', 'Multiple_Attempts', 'Needle_Position', 'Error']

        error_df = pd.DataFrame(columns=error_columns)
        error_df['frame'] = index


        #b. Iterate through the error files
        for error, error_name in error_dict.items():

            if error != 'Error':
                error_folder = os.path.join(folder_errors, "Error_specific/")

                #Find files that have error_name and .csv in the name
                error_files = [f for f in os.listdir(error_folder) if error_name in f and f.endswith('.csv')]

                #Iterate through error files
                for error_file in error_files:

                    error_file_path = os.path.join(error_folder, error_file)
                    error_data = pd.read_csv(error_file_path)

                    if task_type == "NP":
                        column_title = 'name'
                        #Delete 'Needle_Passing' from trial_name if it exists
                        trial_name = trial_name.replace('Needle_Passing', 'NeedlePassing') 
                    
                    elif task_type == "S":
                        column_title = 'files'
                    
                    name_column = error_data[column_title].values            
                    
                    #Check if trial_name is in the 'name' column
                    #where error_data['name'] is trial (Needle_Passing_B001) + _startframe + _endframe + .avi
                    for name in name_column:

                        #Check if trial_name is in the name
                        if trial_name in name:

                            if task_type == "NP":
                                row = error_data[error_data[column_title] == name]
                            
                            elif task_type == "S":
                                row = error_data[error_data[column_title] == name]

                            if not row.empty:
                                start = int(row[column_title].str.extract(r'_(\d+)_')[0].values[0])
                                end = int(row[column_title].str.extract(r'_(\d+)\.avi')[0].values[0])

                                #Get error value (0 or 1)
                                error_value = int(row['label_err1_nor0'].values[0])

                                #Fill the corresponding frames in the error_df
                                error_df.loc[(error_df['frame'] >= start) & (error_df['frame'] <= end), error] = error_value


                            else: #this case happens when the error file does not contain the trial name
                                print(f"Warning: No matching row found for {trial_name} in {error_file_path}")
                                continue         
            
            else:
                #The error files are like "error_{task_type}_G{i}.csv"
                error_files = [os.path.join(folder_errors, f) for f in os.listdir(folder_errors) if f.startswith(f'error_{task_type}') and f.endswith('.csv')]

                #Iterate through error files
                for error_file in error_files:
                    
                    error_data = pd.read_csv(error_file)
                    if task_type == "NP":
                        column_title = 'name'
                        try:
                            trial_name = trial_name.replace('NeedlePassing', 'Needle_Passing')
                        
                        except:
                            trial_name = trial_name

                    elif task_type == "S":
                        column_title = 'files'
                        
                    name_column = error_data[column_title].values

                    #Check if trial_name is in the 'name' column
                    #where error_data['name'] is trial (Needle_Passing_B001) + _startframe + _endframe + .avi
                    for name in name_column:
                        
                        #Check if trial_name is in the name
                        if trial_name in name:
                            
                            row = error_data[error_data[column_title] == name]

                            if not row.empty:
                                start = int(row[column_title].str.extract(r'_(\d+)_')[0].values[0])
                                end = int(row[column_title].str.extract(r'_(\d+)\.avi')[0].values[0])

                                #Get error value (0 or 1)
                                error_value = int(row['label_err1_nor0'].values[0])

                                #Fill the corresponding frames in the error_df
                                error_df.loc[(error_df['frame'] >= start) & (error_df['frame'] <= end), error] = error_value

                            else:
                                print(f"Warning: No matching row found for {trial_name} in {error_file}")
                                continue

        #c. Fill NaN values with 0 (nans occur when a specific error type does not appear for a gesture).
        error_df.fillna(0, inplace=True)

        #Save error_df as .csv
        if task_type == "NP":
            trial_name = trial_name.replace('NeedlePassing', 'Needle_Passing')  #Restore original name format
        output_file = os.path.join(output_folder, f"{trial_name}.csv")
        
        if frequency == 30:
            error_df.to_csv(output_file, index=False)

        else:
            #Keep frames that are present in the kinematics data
            kinematics_path = os.path.join(output_folder_kinematics, f"{trial_name}.csv")
            kinematics_data = pd.read_csv(kinematics_path)
            error_df_final = error_df[error_df['frame'].isin(kinematics_data['frame'])]
            error_df_final.to_csv(output_file, index=False)



def delete_unmatched_frames(image_folder, transcription_folder):

    '''
    This function deletes frames from the image folder that are not present in the transcription files.
    It checks each frame number against the start and end frames in the transcription files.
    

    Args:
    image_folder (str): Path to the folder containing image frames.
    transcription_folder (str): Path to the folder containing transcription files.

    Returns:
    None: The function deletes unmatched frames from the image folder.
    '''

    for filename in os.listdir(image_folder):

        print(f"Processing {filename}...")

        if ".DS_Store" in filename:
            continue

        for frame_number in os.listdir(os.path.join(image_folder, filename)):
            if frame_number == ".DS_Store":
                continue
            frame_number = int(frame_number.split('.')[0])
            
            #Check if frame_number is in the transcription file
            transcription_file = os.path.join(transcription_folder, filename + '.txt')
            transcription_data = pd.read_csv(transcription_file, sep='\s+', header=None, names=['start', 'end', 'gesture'])

            min_frame = transcription_data['start'].min()
            max_frame = transcription_data['end'].max()

            if frame_number < min_frame or frame_number > max_frame:
                #Delete the frame
                frame_path = os.path.join(image_folder, filename, f"{frame_number:04d}.png")
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                else:
                    print(f"Frame {frame_path} does not exist.")
                    



def delete_unmatched_kinematics(kinematics_folder, transcription_folder):
    '''
    This function deletes kinematics data from the kinematics folder that are not present in the transcription files.
    It checks each frame number against the start and end frames in the transcription files.

    Args:
    kinematics_folder (str): Path to the folder containing kinematics data files.
    transcription_folder (str): Path to the folder containing transcription files.

    Returns:
    None: The function deletes unmatched kinematics data from the kinematics folder.

    '''
    for filename in os.listdir(kinematics_folder):

        file_path = os.path.join(kinematics_folder, filename)
        if not filename.endswith('.csv'):
            continue
        
        print(f"Processing {filename}...")

        kinematics_data = pd.read_csv(file_path)

        #Read transcription file
        transcription_file = os.path.join(transcription_folder, filename.replace('.csv', '.txt'))
        if not os.path.exists(transcription_file):
            print(f"Transcription file {transcription_file} does not exist. Skipping {filename}.")
            continue
        transcription_data = pd.read_csv(transcription_file, sep='\s+', header=None, names=['start', 'end', 'gesture'])
        min_frame = transcription_data['start'].min()
        max_frame = transcription_data['end'].max()

        #Eliminate rows (frames) that are not in the transcription file
        kinematics_data = kinematics_data[(kinematics_data['frame'] >= min_frame) & (kinematics_data['frame'] <= max_frame)]

        #Save the processed kinematics data
        kinematics_data.to_csv(file_path, index=False)
    

#Delete frames that correspond to gestures 10 or 11

def delete_gesture_frames(image_folder, kinematics_folder, error_folder, transcriptions_folder):
    '''
    This function deletes frames from the image folder that correspond to gestures 10 or 11.
    It checks each frame number against the transcription files and deletes the frames accordingly.

    Args:
    image_folder (str): Path to the folder containing image frames.
    kinematics_folder (str): Path to the folder containing kinematics data files.
    error_folder (str): Path to the folder containing error data files.
    transcriptions_folder (str): Path to the folder containing transcription files.

    Returns:
    None: The function deletes frames from the image folder and updates the kinematics and error data files.
    '''

    if not os.path.exists(image_folder):
        print(f"Image folder {image_folder} does not exist.")
        return

    if not os.path.exists(transcriptions_folder):
        print(f"Transcriptions folder {transcriptions_folder} does not exist.")
        return
    
    for filename in os.listdir(image_folder):
            
            if ".DS_Store" in filename:
                continue
            
            
            #Read the transcription file for this trial
            transcriptions_file = os.path.join(transcriptions_folder, filename + '.txt')

            #Read kinematics, gestures, and error files
            kinematics_file = os.path.join(kinematics_folder, filename + '.csv')
            kinematics = pd.read_csv(kinematics_file)
            error_file = os.path.join(error_folder, filename + '.csv')
            errors = pd.read_csv(error_file)

            if not os.path.exists(transcriptions_file):
                print(f"Transcriptions file {transcriptions_file} does not exist. Skipping {filename}.")
                continue
            
            transcriptions = pd.read_csv(transcriptions_file, sep='\s+', header=None, names=['start', 'end', 'gesture'])

            #Delete frames that correspond to gestures 10 or 11
            for frame_number in kinematics['frame']:
                 
                gesture = transcriptions[(transcriptions['start'] <= frame_number) & (transcriptions['end'] >= frame_number)]
                if not gesture.empty:
                    gesture_value = int(gesture['gesture'].values[0].replace('G', ''))
                    if gesture_value in [10, 11]:
                        #Delete the frame from the image folder
                        print(frame_number)
                        frame_path = os.path.join(image_folder, filename, f"{frame_number:04d}.png")
                        if os.path.exists(frame_path):
                            os.remove(frame_path)
                            print(f"Deleted frame {frame_number} from {frame_path}")
                        
                        #Delete the row from kinematics and errors
                        kinematics = kinematics[kinematics['frame'] != frame_number]
                        errors = errors[errors['frame'] != frame_number]

            #Save the updated kinematics, gestures, and errors
            kinematics.to_csv(kinematics_file, index=False)
            errors.to_csv(error_file, index=False)


def delete_gesture_vectors(gestures_folder):
    '''
    This function deletes the gestures 10 and 11 from the gesture vectors.
    It iterates through all the .npy files in the gestures folder and removes the frames corresponding to gestures 10 and 11.

    Args:
    gestures_folder (str): Path to the folder containing gesture vectors (.npy files).
    Returns:
    None: The function updates the gesture vectors by removing gestures 10 and 11 and saves them back to the same folder.
    '''

    for filename in os.listdir(gestures_folder):
        if not filename.endswith('.npy'):
            continue
        
        print(f"Processing {filename}...")

        #Load the gesture vector
        gesture_vector = np.load(os.path.join(gestures_folder, filename))

        #Remove gestures 10 and 11
        gesture_vector = gesture_vector[(gesture_vector != 10) & (gesture_vector != 11)]

        #Check if 10 or 11 are still in the vector
        if 10 in gesture_vector or 11 in gesture_vector:
            print(f"Warning: Gestures 10 or 11 still present in {filename}.")

        else:
            #Save the updated gesture vector
            np.save(os.path.join(gestures_folder, filename), gesture_vector)
            

#Create a .pkl file for each trial.

#The .pkl contains the following data:
    #image_feats = np.array of shape (n_frames, 220, 220, 3) with the image frames
    #kinematics_feats = np.array of shape (n_frames, 26) with the kinematics data
    #g_labels = np.array of shape (n_frames,) with the gesture labels
    #e_labels = np.array of shape (n_frames, 5) with the error labels (Out_Of_View, Needle_Drop, Multiple_Attempts, Needle_Position, Error)
    #frames = np.array of shape (n_frames,) with the frame numbers

import pickle

def image_transform(mean, std):
    
    #Inputs: mean and std of the images
    #Outputs: transformation object:
        # Normalize and Standardize
    return transforms.Compose([
        lambda x: x/255.0,
        transforms.Normalize(mean=mean, std=std),
        ])

def create_pkl_files(image_folder, kinematics_folder, gestures_folder, errors_folder, output_folder, task_type, device,
                     folds = ['1Out', '2Out', '3Out', '4Out', '5Out']):
    '''
    This function creates a .pkl file for each trial in the output folder.
    The .pkl file contains the image frames, kinematics data, gesture labels, error labels, and frame numbers.
    '''

    if not os.path.exists(output_folder):
        os.makedirs(output_folder) 

    #For each data fold, we need to create a .pkl file with the images, kinematics, gestures, errors, and frames. 
    #However, the images need to be passed through a fine-tuned ResNet50 model specific to each fold.
    #Before processing the files, let us load all the fine-tuned ResNet50 models for each fold and define the transforms.

    #1. Load models and transforms
    model_dict = {}
    transform_dict = {}
    for fold in folds:
        #a. Load the fine-tuned ResNet50 model for the specific fold
        model_path = f"../models/ResNet50/{fold}.pth"
        if not os.path.exists(model_path):
            print(f"Model {model_path} does not exist. Skipping fold {fold}.")
            continue
        
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(in_features=2048, out_features=1)
        model.load_state_dict(torch.load(model_path))
        model.fc = torch.nn.Identity() #Delete last layer
        model_dict[fold] = model #Save in dictionary

        #b. Define the transforms for the images
        mean_fold = torch.load(os.path.join(output_folder, f"{fold}/mean.pth"))
        std_fold = torch.load(os.path.join(output_folder, f"{fold}/std.pth"))
        transform_dict[fold] = image_transform(mean=mean_fold, std=std_fold)

    #2. Process files
    for filename in os.listdir(image_folder):

        print(f"Processing {filename}...")
        if filename.endswith('.DS_Store'):
            continue
        
        #Get trial name
        if task_type == "NP":
            trial_name = filename.split('_')[0] + '_' + filename.split('_')[1] + "_" + filename.split('_')[2].replace('.csv', '')

        elif task_type == "S":
            trial_name = filename.split('_')[0] + '_' + filename.split('_')[1].replace('.csv', '')

        #2.a. Load all data
        #i. Load image frames
        image_frames_folder = os.path.join(image_folder, filename)
        image_frames = []
        for frame_file in sorted(os.listdir(image_frames_folder)):
            if frame_file.endswith('.png'):
                frame_path = os.path.join(image_frames_folder, frame_file)
                frame = cv2.imread(frame_path)
                image_frames.append(frame)

        #Convert list to torch tensor
        image_feats = torch.tensor(image_frames, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert to (N, C, H, W) format
        if image_feats.shape[1:] != (3, 224, 224):
                print(f"Warning: Image frames in {trial_name} do not have the expected shape (3, 224, 224). Found shape {image_feats.shape[1:]}.")
    
        #ii. Load kinematics data
        kinematics_file = os.path.join(kinematics_folder, f"{trial_name}.csv")
        kinematics_data = pd.read_csv(kinematics_file)
        kinematics_feats = kinematics_data.iloc[:, 1:].values  #exclude 'frame' column
        kinematics_feats = torch.tensor(kinematics_feats, dtype=torch.float32) #Save as torch tensor

        if kinematics_feats.shape[1] != 26: #Check size of kinematics_feats
            print(f"Warning: Kinematics data in {trial_name} does not have the expected shape (n_frames, 26). Found shape {kinematics_feats.shape}.")

        #iii. Load gesture labels
        gestures_file = os.path.join(gestures_folder, f"{trial_name}.npy")
        g_labels = np.load(gestures_file)

        #iv. Load error labels
        errors_file = os.path.join(errors_folder, f"{trial_name}.csv")
        error_data = pd.read_csv(errors_file)
        e_labels = error_data.iloc[:, 1:].values  # Exclude 'frame' column

        #Check size of e_labels
        if e_labels.shape[1] != 5:
            print(f"Warning: Error labels in {trial_name} do not have the expected shape (n_frames, 5). Found shape {e_labels.shape}.")

        #Convert e_labels to torch tensor
        e_labels = torch.tensor(e_labels, dtype=torch.float32)

        #v.Get frame numbers
        frames = kinematics_data['frame'].values

        #3. Process images and save .pkl for each fold
        for fold, model in model_dict.items():  
            
            final_images = torch.empty(image_feats.shape[0], 2048)
            model.eval()
            model.to(device)
            
            #Transform the images
            transform = transform_dict[fold]
            image_feats_transformed = transform(image_feats)

            #Pass the images through the model
            with torch.no_grad():
                for i in range(0, image_feats_transformed.shape[0], 32):
                    batch = image_feats_transformed[i:i+32].to(device)
                    feats = model(batch)
                    final_images[i:i+32] = feats.cpu()

            #Create a dictionary with the data
            data_dict = {
                'image_feats': final_images,
                'kinematics_feats': kinematics_feats,
                'g_labels': g_labels,
                'e_labels': e_labels,
                'frames': frames
            }

            #Save the dictionary as a .pkl file
            output_path = os.path.join(output_folder, fold, f"{trial_name}.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(data_dict, f)

            torch.mps.empty_cache()
    
    print("All .pkl files created.")