# OLIVES Dataset Biomarker Detection using EfficientNetV2

This repository contains a Jupyter Notebook (`oct.ipynb`) demonstrating the process of training a deep learning model to detect biomarkers in Optical Coherence Tomography (OCT) scans using the OLIVES dataset. The model leverages a pre-trained EfficientNetV2 architecture combined with clinical tabular data.

## Overview

The project aims to perform multi-label classification on OCT images to identify the presence or absence of 6 different biomarkers (B1-B6). It utilizes the `gOLIVES/OLIVES_Dataset` hosted on the Hugging Face Hub.

The approach involves:

1.  Loading and preprocessing the OLIVES dataset (images and tabular data).
2.  Defining a custom PyTorch model that combines features from an image CNN (EfficientNetV2-M modified for grayscale) and a small MLP for tabular data (BCVA, CST).
3.  Training the model using Binary Cross-Entropy loss and Adam optimizer with a Cosine Annealing learning rate scheduler.
4.  Evaluating the model on a held-out validation set using Macro F1-score during training.
5.  Evaluating the final trained model on the predefined test set and saving the predictions.

## Features

-   **Dataset Handling:** Uses the `datasets` library to load and manage the `gOLIVES/OLIVES_Dataset`.
-   **Data Preprocessing:**
    -   Resizes images to 224x224.
    -   Normalizes image pixel values (dividing by 255.0).
    -   Handles NaN values in biomarker labels and tabular features (imputing with 0.0).
    -   Filters out data samples where all biomarker labels are NaN.
-   **Model Architecture:**
    -   Uses `timm` to load an `efficientnetv2_m` backbone.
    -   Modifies the input layer of the CNN to accept 1-channel (grayscale) images.
    -   Extracts image features (1280 dimensions).
    -   Processes tabular features (BCVA, CST) using a 2-layer MLP (outputting 16 dimensions).
    -   Concatenates image and tabular features (1280 + 16 = 1296 dimensions).
    -   Uses a final classifier head (Linear -> ReLU -> Linear) to output 6 logits for the biomarkers.
-   **Training:**
    -   Uses `BCEWithLogitsLoss` for multi-label classification.
    -   Employs the Adam optimizer and CosineAnnealingLR scheduler.
    -   Trains for 100 epochs, saving checkpoints every 5 epochs.
-   **Evaluation:**
    -   Calculates Macro F1-score on the validation set during training.
    -   Calculates Macro F1-score on the test set after training.
-   **Prediction Saving:** Saves the final binary predictions on the test set to a CSV file.

## Dataset

-   **Source:** `gOLIVES/OLIVES_Dataset` from Hugging Face Hub.
-   **Configuration:** `biomarker_detection`.
-   **Splits:**
    -   `train`: Used for training and validation (split 80/20 internally within the notebook).
    -   `test`: Used for final model evaluation.
-   **Features Used:** `Image`, `B1`, `B2`, `B3`, `B4`, `B5`, `B6`, `BCVA`, `CST`.

## Setup

1.  **Clone the repository (optional):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    _Alternatively, install manually based on the notebook imports:_

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Adjust CUDA version if needed
    pip install datasets timm scikit-learn numpy pandas Pillow
    ```

    _Note: The notebook upgrades Pillow (`pip install --upgrade Pillow`). Some dependency conflict messages might appear during installation (as seen in the notebook output), but the core functionality should work._

4.  **Hardware:** A CUDA-enabled GPU is highly recommended for reasonable training times. The notebook automatically detects and uses CUDA if available.

## Usage

1.  Ensure all dependencies are installed.
2.  Open and run the Jupyter Notebook `oct.ipynb` sequentially.
3.  The notebook will:
    -   Download the OLIVES dataset (this might take time and disk space, caching is enabled in the notebook's directory).
    -   Preprocess the data.
    -   Define the model.
    -   Train the model for 100 epochs (progress and metrics will be printed). Model checkpoints (e.g., `biomarker_resnet50_epoch_100.pth`) will be saved locally. _Note: The saved filenames in the notebook output use `resnet50`, but the loaded model architecture for testing is EfficientNet._
    -   Load the best/final checkpoint (`biomarker_epoch_100.pth` is loaded in the notebook).
    -   Evaluate the model on the test set.
    -   Save the test set predictions to `oct_test_predictions_binary.csv`.

## Results

The final trained model (EfficientNetV2-M backbone) achieved the following performance on the OLIVES test set:

-   **Macro F1 Score:** ~0.4994

_(Note: The validation F1 score reached much higher values (~0.97) during training, suggesting potential overfitting to the training/validation split or a difference in distribution between the validation and test sets.)_

## Key Files

-   `oct.ipynb`: The main Jupyter Notebook containing all the code.
-   `biomarker_epoch_100.pth` (or similar): Saved PyTorch model state dictionary after training (generated by the notebook).
-   `oct_test_predictions_binary.csv`: CSV file containing the binary predictions (0 or 1) for each biomarker on the test set (generated by the notebook).
-   `README.md`: This file.
