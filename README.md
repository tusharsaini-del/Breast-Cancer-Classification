# Breast Cancer Classification Using Neural Networks

## 1. Project Overview

This project uses a Neural Network classifier to predict whether a breast cancer sample is **malignant** or **benign** based on features computed from digitized images of fine needle aspirates (FNA) of breast masses.

The project uses the Breast Cancer Wisconsin Diagnostic dataset available in Scikit-learn.

**Note:** This is an educational machine learning project and is not intended for clinical diagnosis or medical decision-making.

## 2. Objectives

- Build a Neural Network classification model.
- Preprocess the dataset using feature standardization.
- Train and evaluate the model on separate training and testing sets.
- Measure performance using accuracy, precision, recall, F1-score, and ROC-AUC.
- Visualize the confusion matrix, ROC curve, and training loss.
- Save the trained model for future use.

## 3. Dataset

- **Dataset:** Breast Cancer Wisconsin Diagnostic
- **Total samples:** 569
- **Input features:** 30
- **Classes:** Malignant and Benign
- **Training samples:** 455
- **Testing samples:** 114

The dataset is loaded directly using `sklearn.datasets.load_breast_cancer()`; no manual dataset download is required.

## 4. Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## 5. Methodology

1. Load the built-in dataset.
2. Separate input features and target labels.
3. Split the data into training and testing sets using stratified sampling.
4. Standardize the features using `StandardScaler`.
5. Train an MLP Neural Network with hidden layers of 64 and 32 neurons.
6. Generate predictions on the test dataset.
7. Evaluate performance using classification metrics and ROC-AUC.
8. Save evaluation results, plots, and the trained model.

## 6. Model Configuration

- **Algorithm:** Multi-Layer Perceptron (MLP) Classifier
- **Hidden layers:** 64 and 32 neurons
- **Activation function:** ReLU
- **Optimizer:** Adam
- **Maximum iterations:** 1000
- **Early stopping:** Enabled
- **Random state:** 42

## 7. Results

Results obtained from the executed model:

| Metric | Result |
|---|---:|
| Test Accuracy | 94.74% |
| ROC-AUC | 0.9927 |
| Correct test predictions | 108 / 114 |
| Test samples | 114 |

The classification report and confusion matrix provide additional information about precision, recall, and F1-score for each class.

## 8. Project Structure

```text
Breast Cancer Classification/
├── main.py
├── requirements.txt
├── README.md
├── models/
│   └── breast_cancer_model.joblib
└── results/
    ├── metrics.txt
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── training_loss.png
```

## 9. Installation and Execution

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "Breast Cancer Classification"
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### Step 2: Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the project

```bash
python main.py
```

The evaluation metrics will appear in the terminal. The plots, metrics file, and trained model will be saved in the `results/` and `models/` directories.

## 10. Limitations

- The model was evaluated on a single train-test split.
- Results may vary with different data splits or model configurations.
- High ROC-AUC does not guarantee reliable clinical performance.
- This model is not validated for real-world medical diagnosis.

## 11. Future Improvements

- Perform cross-validation and hyperparameter tuning.
- Compare the Neural Network with other classification algorithms.
- Add a simple user interface for educational demonstrations.
- Improve reproducibility and document additional experiments.

## 12. Conclusion

The project demonstrates how a Neural Network can be trained and evaluated for binary classification using a medical research dataset. The implemented pipeline includes data preprocessing, model training, evaluation, visualization, and model saving.

**Disclaimer:** This project is for academic and educational purposes only and must not be used as a substitute for professional medical advice or diagnosis.
