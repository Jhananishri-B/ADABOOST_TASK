
APP LINK :    https://adaboosttask-mvx5yukas8zgqvzanrsjzf.streamlit.app/


# Credit Card Default Prediction with AdaBoost

## Overview

This project predicts credit card default risk using AdaBoost ensemble learning. It uses the [UCI_Credit_Card.csv](UCI_Credit_Card.csv) dataset and demonstrates data preprocessing, model training, evaluation, and visualization.

## Repository Structure

- [app.py](app.py): Streamlit app for interactive model demo.
- [code.ipynb](code.ipynb): Main notebook for AdaBoost modeling and evaluation.
- [preprocessing.ipynb](preprocessing.ipynb): Data exploration and preprocessing steps.
- [visuals.ipynb](visuals.ipynb): Data visualization and feature analysis.
- [requirements.txt](requirements.txt): Python dependencies.
- [UCI_Credit_Card.csv](UCI_Credit_Card.csv): Dataset.

## Dataset

The dataset contains credit card client information and whether they defaulted on payment next month. Features include demographic info, payment history, bill amounts, and more.

## Setup

1. **Clone the repository**
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run notebooks**
   - Open notebooks in VS Code or Jupyter and run cells sequentially.

## Model Pipeline

1. **Data Loading**
   - Load data using pandas.
2. **Preprocessing**
   - Drop irrelevant columns (`ID`)
   - Handle class imbalance with SMOTE.
3. **Splitting**
   - Train-test split (80/20), stratified by target.
4. **Modeling**
   - Base estimator: DecisionTreeClassifier (max_depth=3)
   - AdaBoostClassifier with 500 estimators, learning rate 0.1.
5. **Training**
   - Fit on both original and SMOTE-resampled data.
6. **Evaluation**
   - Metrics: Accuracy, ROC-AUC, Precision, Recall, F1 Score.
   - Visualizations: Confusion matrix, ROC curve.

## Example Results

```
Accuracy: 0.8172
ROC-AUC: 0.7767
Precision: 0.6615
Recall: 0.3549
F1 Score: 0.4620
```

## Visualizations

- Confusion matrix heatmap
- ROC curve
- Distribution plots for features ([visuals.ipynb](visuals.ipynb))

## Streamlit App

To run the interactive app:
```sh
streamlit run app.py
```

## Requirements

See [requirements.txt](requirements.txt) for all dependencies:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- imbalanced-learn
- streamlit
- joblib

## References

- [UCI Credit Card Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- AdaBoost documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

## License

This project is for educational
