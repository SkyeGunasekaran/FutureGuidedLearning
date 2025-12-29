import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from utils.model_stuff import EarlyStopping
from utils.load_signals_teacher import PrepDataTeacher
from utils.prep_data_teacher import train_val_test_split_continual_t
from models.model import CNN_LSTM_Model


def train_teacher_model(target, epochs, optimizer_type, patience):
    """
    Trains and evaluates a seizure detection model for a given patient.

    Args:
        target (str): Patient identifier.
        epochs (int): Number of training epochs.
        optimizer_type (str): Optimizer type, either 'SGD' or 'Adam'.
        patience (int): Early stopping patience.

    Returns:
        float: AUC-ROC score of the trained model.
    """
    print(f'\nTraining Teacher Model: Patient {target} | Epochs: {epochs} | Optimizer: {optimizer_type} | Patience: {patience}')
    torch.cuda.empty_cache()

    # Load teacher model settings
    with open('teacher_settings.json') as k:
        teacher_settings = json.load(k)

    early_stopping = EarlyStopping(patience=patience, mode='min')
    teacher_losses = []

    # Load ictal and interictal data
    ictal_X, ictal_y = PrepDataTeacher(target, type='ictal', settings=teacher_settings).apply()
    interictal_X, interictal_y = PrepDataTeacher(target, type='interictal', settings=teacher_settings).apply()

    # Split into training and testing sets
    X_train, y_train, X_test, y_test = train_val_test_split_continual_t(
        ictal_X, ictal_y, interictal_X, interictal_y, 0.35
    )

    # Initialize the teacher model
    teacher = CNN_LSTM_Model(X_train.shape).to('cuda')

    # Convert training data to PyTorch tensors and move to GPU
    Y_train = torch.tensor(y_train).long().to('cuda')
    X_train = torch.tensor(X_train).float().to('cuda')

    # Create a DataLoader for the training set
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)

    # Define loss function
    ce_loss = nn.CrossEntropyLoss()

    # Select optimizer
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(teacher.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
    else:  # Default to SGD
        optimizer = torch.optim.SGD(teacher.parameters(), lr=5e-4, momentum=0.9)

    pbar = tqdm(total=epochs, desc=f"Training Teacher Model for Patient {target}")

    # Training loop
    for epoch in range(epochs):
        teacher.train()
        total_loss = 0

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to("cuda"), Y_batch.to("cuda")

            # Forward pass
            teacher_logits = teacher(X_batch)
            loss = ce_loss(teacher_logits, Y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        pbar.update(1)

        # Track loss for early stopping
        avg_loss = total_loss / len(train_loader)
        teacher_losses.append(avg_loss)
        early_stopping.step(avg_loss, epoch)

        if early_stopping.is_stopped():
            print(f"Early stopping at epoch {epoch} with best loss {early_stopping.best_loss:.4f}")
            break

    pbar.close()

    # Evaluate the model on the test set
    teacher.eval()
    X_tensor = torch.tensor(X_test).float().to('cuda')
    y_tensor = torch.tensor(y_test).long().to('cuda')

    with torch.no_grad():
        predictions = teacher(X_tensor)

    # Convert model outputs to probabilities and compute AUC score
    predictions = F.softmax(predictions, dim=1)[:, 1].cpu().numpy()
    auc_test = roc_auc_score(y_tensor.cpu(), predictions)

    print(f'Patient {target} - Test AUC: {auc_test:.4f}')

    # Save the trained teacher model
    model_path = f'pytorch_models/Patient_{target}_detection'
    torch.save(teacher, model_path)

    # Plot training loss
    plt.plot(teacher_losses, label=f'Patient {target} Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss for Patient {target}')
    # Uncomment the line below to display the plot:
    # plt.show()

    return auc_test


if __name__ == "__main__":
    """
    Main execution loop that takes command-line arguments for:
    - Patient ID (or 'all' for multiple patients)
    - Number of epochs
    - Optimizer type (SGD or Adam)
    - Early stopping patience
    """

    parser = argparse.ArgumentParser(description="Seizure Detection Model Training")
    parser.add_argument("--patient", type=str, required=True,
                        help="Target patient ID (use 'all' to train for all patients)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--optimizer", type=str, choices=['SGD', 'Adam'], default='SGD',
                        help="Optimizer type: 'SGD' or 'Adam' (default: SGD)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")

    args = parser.parse_args()

    # List of patients to train models for
    patient_list = ['1', '2', '3', '5', '9', '10', '13', '18', '19', '20', '21', '23']
    if args.patient != 'all':
        patient_list = [args.patient]

    results = {}

    for patient in patient_list:
        auc = train_teacher_model(patient, args.epochs, args.optimizer, args.patience)
        results[patient] = auc

    # Save results to file
    with open('Detection_results.txt', 'a') as f:
        f.write(f'{str(results)}\n')