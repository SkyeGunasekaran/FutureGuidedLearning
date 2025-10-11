# A Predictive Approach to Enhance Time-Series Forecasting

[![Paper](https://img.shields.io/badge/paper-nature_communications-B31B1B.svg)](https://doi.org/10.1038/s41467-025-63786-4)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation for the paper: **"A predictive approach to enhance time-series forecasting"**.

Future-Guided Learning (FGL) is a novel approach that enhances time-series event forecasting using a dynamic feedback mechanism inspired by predictive coding. FGL utilizes a "teacher" model that analyzes future data to identify key events and a "student" model that learns to predict these events from current data. By minimizing the discrepancy between the teacher's near-future insights and the student's long-horizon predictions, FGL allows the forecasting model to dynamically adapt and improve its accuracy.

---

## Key Results

Our experiments demonstrate that FGL significantly improves forecasting performance across different domains:

* [cite_start]**EEG-based Seizure Prediction:** Achieved a **44.8%** average increase in AUC-ROC on the CHBMIT dataset[cite: 15, 64].
* [cite_start]**Nonlinear Dynamical Systems:** Reduced Mean Squared Error (MSE) by **23.4%** in forecasting the Mackey-Glass time series (outlier excluded)[cite: 15, 50].

![Overview of FGL](fig3.png)
<details>
<summary><b>Figure 3: Overview of FGL and its applications. (Click to expand)</b></summary>
<b>A</b> In the FGL framework, a teacher model operates in the relative future of a student model that focuses on long-term forecasting. After training the teacher on its future-oriented task, both models perform inference during the student’s training phase. The probability distributions from the teacher and student are extracted, and a loss is computed based on Eq. (1) [cite_start][cite: 232, 233, 234]. [cite_start]<b>A1</b> Knowledge distillation transfers information via the Kullback–Leibler (KL) divergence between class distributions[cite: 235]. <b>B</b> In an event prediction setting, the teacher is trained directly on the events themselves, while the student is trained to forecast these events. [cite_start]Future labels are distilled from the teacher to the student, guiding the student to align more closely with the teacher model’s predictions, despite using data from the relative past[cite: 249, 250]. <b>C</b> In a regression forecasting scenario, the teacher and student perform short-term and long-term forecasting, respectively. [cite_start]Similar to event prediction, the student gains insights from the teacher during training, enhancing its ability to predict further into the future[cite: 251, 252].
</details>

---

## 1. Setup and Installation

The experiments were conducted using Python 3.11.7 and CUDA 12.1.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/FutureGuidedLearning.git](https://github.com/your-username/FutureGuidedLearning.git)
    cd FutureGuidedLearning
    ```

2.  **Install dependencies:**
    All required packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

---

## 2. Dataset Preparation

This repository supports three datasets. For the EEG datasets, please download the data and place the patient folders inside a `Dataset/` directory at the root of this project.

### AES (American Epilepsy Society)
* [cite_start]**Description:** Intracranial EEG (iEEG) recordings from 5 dogs and 2 human patients[cite: 277].
* **Source:** [Kaggle Seizure Prediction Challenge](https://www.kaggle.com/competitions/seizure-prediction).
* **Setup:** Download and place the patient data folders into the `Dataset/` directory.

### CHB-MIT (Children's Hospital Boston - MIT)
* **Description:** Intracranial EEG recordings from 23 pediatric patients. We train only on select patients with sufficient preictal data.
* **Source:** [PhysioNet](https://physionet.org/content/chbmit/1.0.0/).
* **Setup:** Download and place the patient data inside the `Dataset/` folder.

### Mackey-Glass
* [cite_start]**Description:** A synthetic time-series dataset generated from the Mackey-Glass delay differential equation, used for controlled forecasting experiments[cite: 401, 402].
* **Setup:** This dataset is generated automatically by the scripts.

---

## 3. Running the Experiments

To fully reproduce the results from the paper, follow the three main stages below: training the teacher models, training the baseline models (without FGL), and finally, training the student models with FGL.

### Stage 1: Train the Teacher Models 🧑‍🏫

Teacher models are trained for **seizure detection** (identifying a seizure as it happens) and serve as the source of "future" knowledge.

* **For CHB-MIT (Patient-Specific Teachers):**
    Train a separate detection model for each patient you wish to evaluate.
    ```bash
    # Example for Patient 1. Repeat for other patients (e.g., 3, 5, 10, etc.).
    python seizure_detection.py --patient 1 --epochs 50 --patience 5
    ```

* **For AES (Universal Teachers):**
    [cite_start]The AES dataset lacks seizure labels for training, so we create two "universal" teacher models using an external dataset: one for dogs and one for humans[cite: 275, 276, 278].
    ```bash
    # This script trains and saves both the dog and human universal teachers.
    python create_teacher.py --epochs 50
    ```

### Stage 2: Train the Baseline Models (No FGL)  baseline

These models are trained for **seizure prediction** without any guidance from a teacher. This provides the baseline performance against which FGL is compared.

* **For CHB-MIT & AES:**
    The `seizure_prediction.py` script is used to train the baseline models.
    ```bash
    # Example for a CHB-MIT patient
    python seizure_prediction.py --subject Patient_1 --model MViT --trials 3

    # Example for an AES subject
    python seizure_prediction.py --subject Dog_1 --model CNN_LSTM --trials 3
    ```

### Stage 3: Train the Student Models with FGL 🧠

This is the main experiment where student models are trained for seizure prediction while being guided by the pre-trained teacher models.

* **For CHB-MIT (using patient-specific teachers):**
    ```bash
    python FGL_CHBMIT.py --target 1 --epochs 30 --trials 3 --optimizer Adam --alpha 0.5 --temperature 4
    ```

* **For AES (using universal teachers):**
    ```bash
    python FGL_AES.py --subject Dog_1 --epochs 25 --trials 3 --optimizer Adam --alpha 0.7 --temperature 4
    ```

---

### Mackey-Glass Forecasting Experiments 📈

For the Mackey-Glass dataset, a single script handles the entire pipeline: training the teacher (1-step forecast), the baseline (H-step forecast), and the FGL student (H-step forecast with guidance).

* **Standard FGL Experiment:**
    This command runs the full pipeline for a forecast horizon of 5 steps.
    ```bash
    python FGL_MG.py --horizon 5 --num_bins 50 --alpha 0.5 --epochs 20
    ```
* **FGL with Page-Hinkley Drift Detection:**
    [cite_start]To evaluate FGL with online adaptation to data drift, add the `--use_ph` flag[cite: 143].
    ```bash
    python FGL_MG.py --horizon 5 --num_bins 50 --alpha 0.5 --epochs 20 --use_ph
    ```

---

## 4. Hyperparameter Tuning

Future-Guided Learning leverages Knowledge Distillation, which is primarily controlled by two hyperparameters:

* [cite_start]**Alpha (`α`):** A float between 0 and 1 that balances the standard cross-entropy loss and the KL divergence distillation loss[cite: 247]. A higher `α` relies more on the ground-truth labels.
* [cite_start]**Temperature (`T`):** A positive value that softens the probability distributions from the teacher and student models before calculating the KL divergence[cite: 247]. Higher temperatures create softer distributions, encouraging the student to learn generalized patterns.

Optimal values for these parameters can be found via hyperparameter sweeps, as detailed in the paper's supplementary materials.

---

## 5. Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{Gunasekaran2025,
  author = {Gunasekaran, Skye and Kembay, Assel and Ladret, Hugo and Zhu, Rui-Jie and Perrinet, Laurent and Kavehei, Omid and Eshraghian, Jason},
  title = {A predictive approach to enhance time-series forecasting},
  journal = {Nature Communications},
  year = {2025},
  volume = {16},
  number = {8645},
  pages = {1--7},
  doi = {10.1038/s41467-025-63786-4}
}
