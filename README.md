# Smoker Status Prediction Using Biosignals

This repository implements a supervised machine learning pipeline to predict
smoking status based on medical biosignal and anthropometric features.
The project focuses on feature analysis, model comparison, and handling
moderate class imbalance in a healthcare classification setting.

## Dataset
- Source: Kaggle – *Smoker Status Prediction Using Biosignals*
- Samples: 38,984
- Features: 22 numerical predictors + 1 binary target
- Target: Smoking status (0 = Non-smoker, 1 = Smoker)
- Class distribution:
  - Non-smokers: ~63%
  - Smokers: ~37%
- Missing values: None

## Exploratory Data Analysis (EDA)
- Distribution analysis of all biosignal features
- Outlier identification in triglycerides, liver enzymes (ALT, AST, GTP), and blood pressure
- Correlation analysis revealing strong relationships among liver enzymes and lipid markers
- Identification of key smoking-related features such as GTP, triglycerides, ALT, AST, and waist circumference

## Preprocessing
- Feature–target separation
- Stratified train–test split (80/20) to preserve class distribution
- Standard scaling applied to all features
- Outliers retained due to their medical relevance

## Models Implemented
- Logistic Regression
- Support Vector Machine (Linear Kernel)
- Support Vector Machine (RBF Kernel)
- Multi-Layer Perceptron (Neural Network)

Both default and hyperparameter-tuned versions of each model are evaluated.

## Model Optimization
- Hyperparameter tuning performed using GridSearchCV
- F1-score used as the primary optimization metric due to class imbalance
- Class weighting applied where applicable to improve recall for smokers

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score

F1-score and recall are emphasized to reduce false negatives in a medical context.

## Results Summary
- SVM with RBF kernel achieved the best balanced performance (highest F1-score)
- Tuned neural network achieved the highest overall accuracy
- Linear models showed strong recall but weaker precision
- Non-linear models captured complex relationships in biosignal data more effectively

## Files
- `SmokerStatusPrediction.ipynb` – Main analysis and model training notebook
- `SStest.ipynb` – Test and evaluation notebook
- `SmokerStatusPrediction.pdf` – Detailed project report

## Tools & Libraries
- Python
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn

## Author
Sathvik Teja Moturi
