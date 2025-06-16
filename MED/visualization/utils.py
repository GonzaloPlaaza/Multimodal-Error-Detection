import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['figure.dpi'] = 100 #high resolution
import matplotlib.pyplot as plt
import os

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