import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils.load_signals_student import PrepDataStudent
from utils.prep_data_student import train_val_test_split_continual_s
from models.model import CNN_LSTM_Model


def distill_student_model(target, epochs, trials, optimizer_type, alpha, temperature):
    """
    Performs knowledge distillation for a given patient.

    Args:
        target (str): Patient identifier.
        epochs (int): Number of training epochs.
        trials (int): Number of training trials.
        optimizer_type (str): Optimizer type ('SGD' or 'Adam').
        alpha (float): Weight for cross-entropy loss (1 - alpha for distillation loss).
        temperature (float): Temperature parameter for distillation.

    Returns:
        list: AUC scores for each trial.
    """
    print(f'\nKnowledge Distillation: Patient {target} | Alpha: {alpha:.2f} | Epochs: {epochs} | Trials: {trials} | Optimizer: {optimizer_type}')
    torch.cuda.empty_cache()

    beta = 1 - alpha  # Complementary weight for distillation loss
    auc_list = []

    # Load student settings from JSON file
    with open('student_settings.json') as k:
        student_settings = json.load(k)

    # Load the teacher model for the given patient
    path = f'pytorch_models/Patient_{target}_detection'
    teacher = torch.load(path)

    # Prepare ictal and interictal data
    ictal_X, ictal_y = PrepDataStudent(target, type='ictal', settings=student_settings).apply()
    interictal_X, interictal_y = PrepDataStudent(target, type='interictal', settings=student_settings).apply()

    # Split the data into training and testing sets
    X_train, y_train, X_test, y_test = train_val_test_split_continual_s(
        ictal_X, ictal_y, interictal_X, interictal_y, 0.35
    )

    # Convert training data to PyTorch tensors and move to GPU
    Y_train = torch.tensor(y_train).long().to('cuda')
    X_train = torch.tensor(X_train).float().to('cuda')

    # Create a DataLoader for the training set
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)

    for trial in range(trials):
        print(f'\nPatient {target} | Alpha: {alpha:.2f} | Trial {trial + 1}/{trials}')

        # Instantiate a new student model for each trial
        student = CNN_LSTM_Model(X_train.shape).to('cuda')

        ce_loss = nn.CrossEntropyLoss()
        if optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(student.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
        else:  # Default to SGD
            optimizer = torch.optim.SGD(student.parameters(), lr=5e-4, momentum=0.9)

        pbar = tqdm(total=epochs, desc=f"Training Student Model (Alpha: {alpha:.2f}) for Patient {target}")

        for epoch in range(epochs):
            student.train()
            teacher.eval()
            total_loss = 0

            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to("cuda"), Y_batch.to("cuda")

                student_logits = student(X_batch)
                with torch.no_grad():
                    teacher_logits = teacher(X_batch)

                # Compute soft targets using temperature scaling
                soft_targets = F.softmax(teacher_logits / temperature, dim=1)
                soft_prob = F.log_softmax(student_logits / temperature, dim=1)
                distillation_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)

                # Compute total loss
                loss = alpha * ce_loss(student_logits, Y_batch) + beta * distillation_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            pbar.update(1)

        pbar.close()

        # Evaluate the student model on the test set
        student.eval()
        X_tensor = torch.tensor(X_test).float().to('cuda')
        y_tensor = torch.tensor(y_test).long().to('cuda')

        with torch.no_grad():
            predictions = student(X_tensor)

        predictions = F.softmax(predictions, dim=1)[:, 1].cpu().numpy()
        auc_test = roc_auc_score(y_tensor.cpu(), predictions)

        print(f'Patient {target}, Alpha {alpha:.2f} | Test AUC: {auc_test:.4f}')
        auc_list.append(auc_test)

    # Save results to file
    with open("FGL_results.txt", 'a') as f:
        f.write(f'Patient_{target}_Alpha_{alpha:.2f}_Results= {str(auc_list)}\n')

    return auc_list


if __name__ == "__main__":
    """
    Main execution loop that takes command-line arguments for:
    - Patient ID (or 'all' for multiple patients)
    - Number of epochs
    - Optimizer type (SGD or Adam)
    - Number of trials
    - Alpha value for loss balancing
    - Temperature parameter for knowledge distillation
    """

    parser = argparse.ArgumentParser(description="Knowledge Distillation for Seizure Prediction")
    parser.add_argument("--patient", type=str, required=True,
                        help="Target patient ID (use 'all' to train for all patients)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--optimizer", type=str, choices=['SGD', 'Adam'], default='SGD',
                        help="Optimizer type: 'SGD' or 'Adam' (default: SGD)")
    parser.add_argument("--trials", type=int, default=3, help="Number of training trials (default: 3)")
    parser.add_argument("--alpha", type=float, required=True, 
                        help="Alpha value for loss weighting (cross-entropy weight)")
    parser.add_argument("--temperature", type=float, default=4, help="Temperature for distillation (default: 4)")

    args = parser.parse_args()

    # List of patients to process
    patient_list = ['1', '2', '3', '5', '9', '10', '13', '18', '19', '20', '21', '23']
    if args.patient != 'all':
        patient_list = [args.patient]

    results = {}

    for patient in patient_list:
        results[patient] = distill_student_model(
            patient, args.epochs, args.trials, args.optimizer, args.alpha, args.temperature
        )

    # Save all results to file
    with open("FGL_results.txt", 'a') as f:
        f.write(f'{str(results)}\n')