import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['figure.dpi'] = 100 #high resolution
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def plot_results_LOSO(train_f1_fold, 
                        test_f1_fold, 
                        train_loss_fold,
                        test_loss_fold,
                        setting, out, image_folder):

    """
    Plot the results of the LOSO cross-validation.
    Inputs:   
    - train_f1_fold: List of F1 scores for the training set for each fold.
    - test_f1_fold: List of F1 scores for the test set for each fold.
    - train_loss_fold: List of loss values for the training set for each fold.
    - test_loss_fold: List of loss values for the test set for each fold.

    """

    #Display evolution of loss, f1
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(train_f1_fold, label='Train F1', marker='o')
    plt.plot(test_f1_fold, label='Test F1', marker='o')
    plt.title(f'{setting} - Fold {out} - F1 Score')
    plt.xlabel('Fold')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(train_loss_fold, label='Train Loss', marker='o')
    plt.plot(test_loss_fold, label='Test Loss', marker='o')
    plt.title(f'{setting} - Fold {out} - Loss')
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(image_folder, f'{setting}_fold_{out}_results.png'))
    plt.show()


def plot_cm(LOSO_cm_train: np.ndarray,
            LOSO_cm_test: np.ndarray,
            image_folder: str,
            binary: str = None):
    """
    Plot the confusion matrices for the LOSO cross-validation results.

    Inputs:
    - LOSO_cm_train: Confusion matrix for the training set.
    - LOSO_cm_test: Confusion matrix for the test set.
    - image_folder: Folder where the confusion matrix images will be saved.
    - binary: Optional string to specify the binary classification setting.

    Returns: None
    """

    print("Confusion Matrix for Train:")
    plt.figure(figsize=(15, 15))
    
    if binary:
        ConfusionMatrixDisplay(LOSO_cm_train, display_labels=['No Error', 'Error']).plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - Train - {binary}')
        plt.savefig(os.path.join(image_folder, f'LOSO_Train_Confusion_Matrix_{binary}.png'))

    else:
        ConfusionMatrixDisplay(LOSO_cm_train, display_labels=['No Error', 'OOV', 'MA', 'NP', ' OOV + MA', 'MA + NP']).plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix - Train')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(image_folder, 'LOSO_Train_Confusion_Matrix.png'))

    plt.show()
    
    print("Confusion Matrix for Test:")
    plt.figure(figsize=(15, 15))
    
    if binary:
        ConfusionMatrixDisplay(LOSO_cm_test, display_labels=['No Error', 'Error']).plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - Test - {binary}')
        plt.savefig(os.path.join(image_folder, f'LOSO_Test_Confusion_Matrix_{binary}.png'))
    else:
        ConfusionMatrixDisplay(LOSO_cm_test, display_labels=['No Error', 'OOV', 'MA', 'NP', ' OOV + MA', 'MA + NP']).plot(cmap=plt.cm.Blues)
        plt.xticks(rotation=45)
        plt.title('Confusion Matrix - Test')
        plt.savefig(os.path.join(image_folder, 'LOSO_Train_Confusion_Matrix.png'))

    plt.show()
