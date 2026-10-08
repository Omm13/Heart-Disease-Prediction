# AI-Based Heart Disease Risk Prediction Using Machine Learning

An end-to-end AI for Healthcare mini-project that predicts heart disease risk using Machine Learning and an interactive Streamlit web application.

## Overview

This project uses the UCI Heart Disease Dataset (Cleveland subset) to build a binary classification model that predicts whether a patient belongs to the higher-risk or lower-risk category based on selected health attributes.

The project covers the complete Machine Learning workflow, including data preprocessing, exploratory data analysis (EDA), model development, performance evaluation, model serialization, and deployment through Streamlit.

**Disclaimer:** This project is intended for academic and educational purposes only. Its predictions are not medical diagnoses and must not replace professional medical advice.

## Objectives

* Analyze patient health data related to heart disease.
* Perform data cleaning and exploratory data analysis.
* Handle missing values and preprocess numerical and categorical features.
* Train a Logistic Regression classification model.
* Evaluate model performance using standard classification metrics.
* Save and reuse the trained model.
* Develop an interactive Streamlit application for predictions.
* Validate consistency between the saved model and the application.

## Dataset

**Source:** UCI Machine Learning Repository — Heart Disease Dataset

**Link:** https://archive.ics.uci.edu/dataset/45/heart+disease

The project uses the Cleveland subset, which contains 303 patient records and 13 input attributes.

The original target variable contains values from 0 to 4. For this project, it was converted into a binary target:

* `0` — No Heart Disease
* `1` — Presence of Heart Disease (original values 1–4)

The resulting dataset contains 303 records and 14 columns, including the target variable.

### Input Features

| Feature    | Description                           |
| ---------- | ------------------------------------- |
| `age`      | Age of the patient                    |
| `sex`      | Sex                                   |
| `cp`       | Chest pain type                       |
| `trestbps` | Resting blood pressure                |
| `chol`     | Serum cholesterol                     |
| `fbs`      | Fasting blood sugar indicator         |
| `restecg`  | Resting electrocardiographic results  |
| `thalach`  | Maximum heart rate achieved           |
| `exang`    | Exercise-induced angina               |
| `oldpeak`  | ST depression induced by exercise     |
| `slope`    | Slope of the peak exercise ST segment |
| `ca`       | Number of major vessels               |
| `thal`     | Thalassemia-related test result       |
| `target`   | Binary classification label           |

## Technologies Used

* **Programming Language:** Python
* **Data Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn
* **Model Serialization:** Joblib
* **Web Application:** Streamlit
* **Development Environment:** Jupyter Notebook, VS Code

## Project Workflow

```text
UCI Heart Disease Dataset
          |
          v
    Data Loading
          |
          v
 Data Cleaning and EDA
          |
          v
 Feature Preprocessing
          |
          v
 Train-Test Split (80:20)
          |
          v
   Logistic Regression
          |
          v
 Model Evaluation
          |
          v
    Save Model (.pkl)
          |
          v
 Streamlit Web Application
          |
          v
  Patient Input and Prediction
```

## Exploratory Data Analysis (EDA)

The following analyses were performed to understand the dataset:

* Target class distribution
* Age distribution
* Age versus heart disease
* Cholesterol versus heart disease
* Maximum heart rate versus heart disease
* Feature correlation heatmap
* Descriptive statistics
* Unique value analysis
* Comparison of selected numerical features across target classes

### Target Distribution

The dataset contains the following target classes:

| Class            | Number of Records |
| ---------------- | ----------------: |
| No Heart Disease |               164 |
| Heart Disease    |               139 |
| **Total**        |           **303** |

## Data Preprocessing

Missing values were identified in two features:

* `ca`: 4 missing values
* `thal`: 2 missing values

For the final model, preprocessing was implemented within a Scikit-learn Pipeline to ensure that imputation and transformation steps were fitted using the training data rather than the entire dataset.

### Numerical Features

The following numerical features were processed using:

1. Median imputation
2. Standardization using `StandardScaler`

Features: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`

### Categorical Features

The categorical features were processed using:

1. Most-frequent-value imputation
2. One-Hot Encoding

Features: `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`

`OneHotEncoder(handle_unknown="ignore")` was used to handle previously unseen categorical values safely during transformation.

## Machine Learning Model

**Algorithm:** Logistic Regression

Logistic Regression was selected because it is suitable for binary classification, computationally efficient, and straightforward to integrate into a structured-data prediction application.

The final Scikit-learn Pipeline combines:

* Missing-value imputation
* Feature scaling
* One-Hot Encoding
* Logistic Regression classification

### Train-Test Split

* Training set: 80% (242 records)
* Testing set: 20% (61 records)
* Random state: `42`
* Stratification: Applied to preserve the target class distribution

## Model Performance

The model was evaluated on the held-out test set of 61 samples.

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 88.52% |
| Precision | 83.87% |
| Recall    | 92.86% |
| F1-Score  | 88.14% |

### Classification Report

| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------: | -----: | -------: | ------: |
| No Heart Disease |      0.93 |   0.85 |     0.89 |      33 |
| Heart Disease    |      0.84 |   0.93 |     0.88 |      28 |
| **Accuracy**     |         — |      — | **0.89** |  **61** |
| Macro Average    |      0.89 |   0.89 |     0.89 |      61 |
| Weighted Average |      0.89 |   0.89 |     0.89 |      61 |

The model achieved 92.86% recall for the heart disease class, correctly identifying 26 of the 28 positive cases in the held-out test set.

These results apply only to this particular test split and do not establish clinical reliability.

### Confusion Matrix

| Actual / Predicted | No Heart Disease | Heart Disease |
| ------------------ | ---------------: | ------------: |
| No Heart Disease   |               28 |             5 |
| Heart Disease      |                2 |            26 |

* **True Negatives:** 28
* **False Positives:** 5
* **False Negatives:** 2
* **True Positives:** 26

## Streamlit Web Application

The project includes an interactive Streamlit application that loads the saved model and accepts patient health attributes through a user-friendly interface.

### Application Features

* Interactive patient input fields
* Human-readable categorical selections
* Prediction of the model's risk category
* Display of the estimated model probability
* Clear educational-use disclaimer

The application displays one of two results:

* **Higher Risk of Heart Disease**
* **Lower Risk of Heart Disease**

The probability shown is generated by the trained model. It should not be interpreted as a clinically calibrated probability or a confirmed diagnosis.

## Application Validation

The saved model was tested directly in the Jupyter Notebook and through the Streamlit application using the same patient inputs.

### Example Validation Result

* Age: 65
* Sex: Male
* Chest Pain Type: Asymptomatic
* Resting Blood Pressure: 150
* Cholesterol: 280
* Fasting Blood Sugar: No
* Resting ECG: Normal
* Maximum Heart Rate: 110
* Exercise-Induced Angina: Yes
* Oldpeak: 2.5
* Slope: Flat
* Number of Major Vessels: 2
* Thalassemia: Reversible Defect

**Prediction:** Higher Risk of Heart Disease

**Estimated Model Probability:** 99.53%

Both the notebook and Streamlit application returned the same prediction and probability for these inputs. This validates consistency between the saved model and the application for this test case; it does not establish clinical accuracy.

## Project Structure

```text
Heart-Disease-Prediction/
│
├── data/
│   ├── processed.cleveland.data
│   ├── heart_disease.csv
│   └── heart_disease_cleaned.csv
│
├── model/
│   └── heart_disease_model.pkl
│
├── notebook/
│   └── heart_disease_analysis.ipynb
│
├── screenshots/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### Prerequisites

* Python installed on your system
* Git
* A code editor such as VS Code

### 1. Clone the Repository

```bash
git clone https://github.com/Omm13/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

Replace the repository URL if your GitHub repository uses a different name or URL.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
venv\Scripts\activate.bat
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application should open in your default browser. If it does not, open the local URL displayed in the terminal.

## Requirements

The `requirements.txt` file should contain the libraries required to run the project:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
joblib
jupyter
```

## Limitations

* The dataset contains only 303 records from the Cleveland subset.
* The model has been evaluated using a single train-test split.
* No independent external clinical validation has been performed.
* Model performance may vary across populations and healthcare settings.
* The displayed probability is a model output, not a confirmed medical probability.
* The application is not intended for diagnosis, treatment decisions, or emergency assessment.

## Future Scope

Potential improvements include:

* Comparing Logistic Regression with other classification algorithms.
* Performing cross-validation and hyperparameter tuning.
* Evaluating the model on independent datasets.
* Exploring model explainability techniques such as SHAP.
* Improving visualizations and the user interface.
* Adding patient-history visualization.
* Investigating probability calibration and fairness across patient groups.
* Exploring secure deployment for educational demonstrations.

## Conclusion

This project demonstrates an end-to-end AI for Healthcare workflow using a structured healthcare dataset, Scikit-learn, Logistic Regression, and Streamlit.

The model achieved 88.52% accuracy, 92.86% recall, and 88.14% F1-score on the held-out test set. The trained pipeline was integrated into a web application, and a validation test confirmed that the saved model and application produced consistent outputs for identical inputs.

The project provides practical experience in data preprocessing, exploratory analysis, supervised learning, model evaluation, serialization, and interactive application development.

## Acknowledgements

* UCI Machine Learning Repository for providing the Heart Disease Dataset.
* The open-source Python ecosystem, including Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib, and Streamlit.
