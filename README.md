# Hospital Mortality Analysis and Predictive Modeling Using Machine Learning

<div align="center">

<img src="./machinelearnig_health.jpg" alt="Healthcare Analytics and Machine Learning" width="650">

*Healthcare Data Analytics — Python, XGBoost, and exploratory analysis of historical hospital mortality records.*

</div>
# Hospital Mortality Analysis and Predictive Modeling Using Machine Learning

### Historical Mortality Trends, Exploratory Data Analysis, and XGBoost Classification with Python

**Project Type:** Academic Healthcare Data Analytics and Machine Learning Research  
**Field:** Data Science | Healthcare Analytics | Machine Learning | Predictive Modeling  
**Programming Language:** Python  
**Environment:** Google Colab | Jupyter Notebook  
**Author:** Data Engineer Luis Barahona

---

## 1. Project Overview

This project explores historical hospital mortality records using Python, exploratory data analysis, statistical visualizations, and machine learning.

The notebook analyzes records covering 2013–2023 to investigate differences in recorded patient counts across disease categories and historical periods.

The analysis compares three periods:

- Before the COVID-19 pandemic: 2013–2019
- During the COVID-19 pandemic: 2020–2022
- The year 2023

An experimental XGBoost classification model is subsequently developed to identify disease-year records associated with relatively high patient counts.

The project combines data preparation, ICD-10 category encoding, descriptive analysis, machine learning evaluation, and interactive data visualization.

**Research scope:** This is an exploratory analysis of aggregated hospital records, not an individual patient mortality prediction system.

---

## 2. Project Objectives

The main objectives are to:

1. Import and examine historical hospital mortality records.
2. Identify missing data and inspect dataset structure.
3. Organize disease categories and diagnostic descriptions.
4. Compare recorded patient counts across historical periods.
5. Analyze the most frequently represented disease categories.
6. Generate statistical visualizations.
7. Prepare numerical features for machine learning.
8. Define a binary classification target from the dataset.
9. Train and evaluate an XGBoost classifier.
10. Explore predictions for disease categories in 2024.
11. Identify methodological limitations before interpreting results.

---

## 3. Dataset Description

**Input file:** `Mortalidad.xls`

**Historical coverage:** 2013–2023

The saved notebook execution reports 1,121 observations and four original columns.

### Original Dataset Structure

| Column | Description |
|:---|:---|
| AÑO | Year of observation |
| C.I.E.10 | ICD-10 diagnostic classification code |
| MORTALIDAD | Recorded disease or mortality category |
| NUMERO DE PACIENTES | Recorded patient count |

### Data Quality

The saved notebook reports:

- 1,121 records.
- Four original columns.
- No missing values in the original fields.
- Observations covering eleven calendar years.

The dataset is aggregated by disease category rather than organized as individual patient-level clinical records.

The notebook does not establish the complete provenance, patient population, coverage, or reporting methodology of the original Excel file.

These details would be necessary for broader epidemiological or clinical interpretation.

---

## 4. Data Preparation

### 4.1 Importing Data

The original dataset is loaded using Pandas.

```python
import pandas as pd

df = pd.read_excel(
    "/content/Mortalidad.xls",
    sheet_name=True
)
```

The notebook then examines the first and last rows of the dataset.

### 4.2 Data Exploration

The workflow uses:

```python
df.info()
df.describe()
df.isna().sum()
```

These operations help identify the structure, data types, and descriptive characteristics of the observations.

### 4.3 Disease Category Normalization

The `MORTALIDAD` field contains disease descriptions that may include additional specifications separated by commas.

The notebook creates a new column to store part of the descriptive information:

```python
df["Especification"] = (
    df["MORTALIDAD"]
    .str.split(",")
    .str[1]
)
```

The main disease field is also simplified:

```python
df["MORTALIDAD"] = (
    df["MORTALIDAD"]
    .str.split(",")
    .str[0]
)
```

This transformation makes disease-level aggregation easier.

However, the operation preserves only part of the original description and may discard clinically meaningful distinctions.

The original ICD-10 codes should therefore remain available for more rigorous classification.

---

## 5. Exploratory Data Analysis

The notebook groups records by disease category and calculates the total recorded patient count.

```python
disease_counts = (
    df.groupby("MORTALIDAD")["NUMERO DE PACIENTES"]
    .sum()
    .sort_values(ascending=False)
)
```

This procedure identifies categories associated with the largest aggregated counts in the available data.

### Highest Recorded Counts — Entire Dataset

| Disease Category | Recorded Patient Count |
|:---|---:|
| COVID-19 | 608 |
| Pneumonia | 438 |
| Type 2 diabetes mellitus | 203 |

These are aggregated counts reported in the notebook, not population-adjusted mortality rates.

The results do not establish the probability that an individual patient will die from any of these conditions.

---

## 6. Historical Period Comparison

The notebook separates observations into three periods to explore changes in the distribution of recorded disease categories.

### 6.1 Before the COVID-19 Pandemic

**Period:** 2013–2019

The notebook identifies the following highest recorded categories:

| Disease Category | Recorded Patient Count |
|:---|---:|
| Pneumonia | 329 |
| Sepsis | 108 |
| Type 2 diabetes mellitus | 93 |

Pneumonia represents the highest count among these categories in the analyzed pre-pandemic records.

### 6.2 During the COVID-19 Pandemic

**Period:** 2020–2022

| Disease Category | Recorded Patient Count |
|:---|---:|
| COVID-19 | 602 |
| Type 2 diabetes mellitus | 84 |
| Pneumonia | 54 |

The results demonstrate a substantial concentration of recorded counts in the COVID-19 category during the selected period.

### 6.3 The Year 2023

The notebook identifies the following categories:

| Disease Category | Recorded Patient Count |
|:---|---:|
| Pneumonia | 55 |
| Type 2 diabetes mellitus | 26 |
| Extreme immaturity | 15 |

### Interpretation

The distribution of recorded patient counts differs across these three historical periods.

However, the notebook does not adjust for population size, hospital admissions, changes in reporting, or other potential confounding factors.

Therefore, the comparisons should be interpreted descriptively rather than as estimates of causal pandemic effects.

---

## 7. Data Visualization

The exploratory analysis uses several visualization techniques.

### Bar Charts

Bar charts compare recorded counts across disease categories.

The notebook generates top-ten and top-three category rankings for multiple periods.

### Three-Dimensional Scatter Plots

Three-dimensional plots display relationships between:

- Observation year.
- Recorded patient count.
- Binary classification target.

### Encoded Category Visualizations

The notebook illustrates numerical encodings of ICD-10 and disease categories.

### Interactive Plotly Visualizations

Plotly is used to construct interactive three-dimensional scatter plots and prediction charts.

### Prediction Heatmaps

Heatmaps display the model's estimated positive-class probabilities for selected disease categories.

These plots are exploratory model outputs and should not be interpreted as clinically calibrated risk estimates.

---

## 8. Machine Learning Problem Definition

The notebook defines an experimental binary classification target called:

```text
ALTA_MORTALIDAD
```

This label is calculated from the recorded patient count.

### Threshold Definition

The threshold is set to the 75th percentile:

```python
threshold = (
    df_modelo["NUMERO DE PACIENTES"]
    .quantile(0.75)
)
```

The saved notebook reports:

```text
Threshold = 2.0
```

### Target Construction

```python
df_modelo["ALTA_MORTALIDAD"] = (
    df_modelo["NUMERO DE PACIENTES"] >= threshold
).astype(int)
```

This creates two categories:

| Target | Interpretation |
|:---:|:---|
| 0 | Recorded patient count below 2 |
| 1 | Recorded patient count of 2 or more |

### Important Interpretation

The target represents a dataset-specific patient-count threshold.

It does not measure:

- Individual mortality risk.
- Case fatality rate.
- Disease severity.
- Probability of death.
- A clinically validated high-mortality definition.

Consequently, the classifier should be described as predicting a high-count category in aggregated mortality records.

---

## 9. Feature Engineering

The notebook prepares three input features:

1. Observation year.
2. Encoded ICD-10 code.
3. Encoded disease category.

### Categorical Encoding

Scikit-learn's LabelEncoder is used to convert categorical strings into numerical values.

```python
from sklearn.preprocessing import LabelEncoder

le_cie10 = LabelEncoder()
le_mortalidad = LabelEncoder()

df_modelo["C.I.E.10_CODE"] = (
    le_cie10.fit_transform(
        df_modelo["C.I.E.10"]
    )
)

df_modelo["MORTALIDAD_CODE"] = (
    le_mortalidad.fit_transform(
        df_modelo["MORTALIDAD"]
    )
)
```

These numerical representations allow the model to process the categorical fields.

However, integer encoding introduces an artificial numerical ordering between categories.

Alternative categorical encoding strategies should be evaluated in future versions.

---

## 10. Training and Testing

The notebook uses three predictive features:

```python
X = df_modelo[
    [
        "AÑO",
        "C.I.E.10_CODE",
        "MORTALIDAD_CODE"
    ]
]

y = df_modelo["ALTA_MORTALIDAD"]
```

### Dataset Split

An 80/20 train-test split is implemented.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)
```

### Recorded Split

| Dataset | Observations |
|:---|---:|
| Training | 896 |
| Testing | 225 |
| Total | 1,121 |

The random split allows records from different historical years and disease categories to appear in both training and testing sets.

This is not equivalent to a chronological validation of future-year predictions.

---

## 11. XGBoost Classification Model

The implemented machine learning model is:

**XGBoost — Extreme Gradient Boosting**

The notebook uses:

```python
import xgboost as xgb

model_xgb = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

model_xgb.fit(X_train, y_train)
```

### Why XGBoost?

XGBoost combines decision trees through gradient boosting.

The algorithm can model nonlinear relationships and interactions between input variables.

It is commonly used for structured tabular machine learning tasks.

### Model Scope

Although the notebook discusses Random Forest and Support Vector Machines conceptually, the evaluated predictive model is XGBoost.

The notebook does not report a complete comparative benchmark between those algorithms.

---

## 12. Model Evaluation

The trained XGBoost classifier is evaluated using the testing dataset.

```python
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

y_pred = model_xgb.predict(X_test)

print(
    accuracy_score(y_test, y_pred)
)

print(
    classification_report(
        y_test,
        y_pred
    )
)
```

### Saved Evaluation Results

| Metric | Recorded Value |
|:---|---:|
| Test accuracy | 73.33% |
| Class 0 precision | 0.82 |
| Class 0 recall | 0.79 |
| Class 0 F1-score | 0.81 |
| Class 1 precision | 0.55 |
| Class 1 recall | 0.60 |
| Class 1 F1-score | 0.58 |
| Macro-average F1 | 0.69 |
| Weighted-average F1 | 0.74 |

### Class Support

| Class | Test Observations |
|:---:|---:|
| 0 | 157 |
| 1 | 68 |
| Total | 225 |

### Interpretation

The model achieved an overall test accuracy of 73.33%.

However, the performance for class 1 was lower than for class 0.

For the positive class:

- Precision: 0.55
- Recall: 0.60
- F1-score: 0.58

This means the classifier makes substantial errors when identifying records above the selected patient-count threshold.

Accuracy alone is insufficient to evaluate the quality of the model.

The recorded results do not establish suitability for clinical or public health decision-making.

---

## 13. Experimental Predictions for 2024

The notebook identifies the last available year:

```text
2023
```

It then constructs an experimental prediction dataset for:

```text
2024
```

The procedure creates one observation for each unique disease and ICD-10 category combination.

The trained model assigns:

- A predicted binary label.
- An estimated probability for class 1.

### Example

```python
df_pred[
    "PREDICCION_ALTA_MORTALIDAD"
] = model_xgb.predict(X_pred)

df_pred[
    "PROBABILIDAD_ALTA_MORTALIDAD"
] = model_xgb.predict_proba(
    X_pred
)[:, 1]
```

### Selected Top-Ranked Categories

The saved prediction results include:

| Disease Category | Model Score |
|:---|---:|
| Cholangitis | 0.989515 |
| Local skin and subcutaneous tissue infection | 0.988434 |
| Gastrointestinal hemorrhage | 0.988342 |
| Hepatic failure | 0.984473 |

The scores are model-generated probabilities for the binary target defined by the dataset.

They are not calibrated probabilities of death or estimates of future disease-specific mortality.

### Prediction Limitations

The notebook does not evaluate these predictions against independently observed 2024 data.

Its random train-test split also does not establish accuracy for forecasting unseen future years.

Therefore, this section should be interpreted as an exploratory extrapolation experiment.

---

## 14. Technologies Used

| Technology | Purpose |
|:---|:---|
| Python | Programming language |
| Pandas | Data cleaning and transformation |
| NumPy | Numerical operations |
| Matplotlib | Statistical visualization |
| Seaborn | Exploratory charts |
| Plotly | Interactive visualizations |
| Scikit-learn | Data preprocessing and model evaluation |
| LabelEncoder | Categorical encoding |
| XGBoost | Gradient boosting classification |
| Google Colab | Cloud notebook execution |
| Jupyter Notebook | Interactive development |

---

## 15. How to Run

### Google Colab

[Open the Original Notebook](https://colab.research.google.com/drive/1cjE0XpoiXhFaEG-uYRbWs-5BuWU2DXBk)

### Requirements

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- XGBoost
- openpyxl or another compatible Excel reader
- Google Colab or Jupyter Notebook
- The original `Mortalidad.xls` dataset

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost xlrd openpyxl notebook
```

### Execution Instructions

1. Open the notebook in Google Colab.
2. Upload `Mortalidad.xls`.
3. Confirm the Excel file path.
4. Execute the data import cells.
5. Inspect the dataset structure.
6. Review the historical mortality analysis.
7. Generate the period comparison charts.
8. Prepare the binary classification target.
9. Encode categorical features.
10. Split the data into training and testing sets.
11. Train the XGBoost classifier.
12. Review the classification report.
13. Explore the experimental 2024 model predictions.

### Important Data Requirement

The original notebook expects the file:

```text
/content/Mortalidad.xls
```

If the dataset is not available at that location, update the path before running the notebook.

---

## 16. Methodological Limitations

This project is an educational machine learning experiment.

### Dataset Provenance

The notebook does not provide sufficient source documentation to establish the complete reporting scope or representativeness of the hospital records.

### Aggregated Observations

The dataset contains disease-year records, not patient-level clinical profiles.

### Target Definition

The binary target is derived from a percentile of recorded patient counts rather than a clinical mortality-risk definition.

### Random Split

The evaluation uses a random train-test split, not chronological forecasting validation.

### Category Encoding

Integer-encoded categorical features may introduce unintended numerical relationships.

### Preprocessing Leakage

The categorical encoders and target threshold are determined before the train-test split.

A stricter evaluation would fit preprocessing and threshold definitions only within the training process.

### Class Performance

The positive class has an F1-score of 0.58 in the saved test results.

### No External Validation

The notebook does not evaluate the model against an independent hospital dataset.

### No Prospective Validation

The proposed 2024 classifications are not compared with independently observed 2024 outcomes.

### Probability Calibration

The notebook does not include calibration testing for model-generated probabilities.

### Medical Interpretation

The model does not predict individual patient survival or support medical diagnosis.

The analysis should not be used for patient care, hospital resource allocation, or public health decisions without further validation.

---

## 17. Future Improvements

Potential improvements include:

- Documenting the original dataset source and population.
- Preserving complete diagnostic descriptions.
- Validating ICD-10 category mappings.
- Using chronological train-test splits.
- Implementing time-based cross-validation.
- Comparing XGBoost with Random Forest and other baseline models.
- Applying preprocessing within Scikit-learn pipelines.
- Evaluating alternative categorical encoding strategies.
- Measuring ROC-AUC and precision-recall metrics where appropriate.
- Including model probability calibration.
- Comparing predictions with independently observed future data.
- Evaluating model stability across years.
- Adding explainability techniques such as SHAP.
- Creating a reproducible analytical dashboard.

---

## 18. Potential Applications

The analytical techniques explored in this project may support:

- Educational healthcare data analysis.
- Hospital reporting research.
- Historical disease-category analysis.
- Exploratory trend visualization.
- Machine learning classification education.
- Health data quality evaluation.
- Statistical reporting prototypes.

Any operational healthcare application would require a different validation standard and appropriate clinical oversight.

---

## 19. Conclusion

This project demonstrates the application of Python, exploratory data analysis, and XGBoost to historical hospital mortality-related records from 2013–2023.

The exploratory analysis identifies differences in recorded patient counts before, during, and after the main COVID-19 comparison period.

The machine learning experiment defines a high-count classification label using the dataset's 75th percentile and trains an XGBoost model using year and encoded disease categories.

The saved test evaluation reports 73.33% accuracy, with an F1-score of 0.58 for the positive class.

These findings demonstrate an educational classification workflow but do not establish clinically meaningful mortality risk prediction or prospective forecasting performance.

The project provides a foundation for future research involving data quality, temporal validation, statistical modeling, and responsible machine learning in healthcare analytics.

---

## 20. Project Resources

### Google Colab Notebook

[Open Hospital Mortality Analysis and Predictive Modeling](https://colab.research.google.com/drive/1cjE0XpoiXhFaEG-uYRbWs-5BuWU2DXBk)

### XGBoost Documentation

[XGBoost — Official Documentation](https://xgboost.readthedocs.io/)

### Scikit-learn Documentation

[Scikit-learn — Machine Learning in Python](https://scikit-learn.org/stable/)

### World Health Organization

[WHO — International Classification of Diseases](https://www.who.int/standards/classifications/classification-of-diseases)

These resources provide technical and classification background. They do not constitute validation or endorsement of this independent notebook.

---

## Author

**Data Engineer Luis Barahona**

Mexican Data Engineer and 2026 graduate of Universidad del Caribe, interested in Data Engineering, Data Science, Artificial Intelligence, Machine Learning, and technology applied to healthcare analytics.

---

## Contact

| Platform | Contact |
|:---|:---|
| Email | [190300395@ucaribe.edu.mx](mailto:190300395@ucaribe.edu.mx) |
| GitHub | [github.com/LuisBarahonaMX](https://github.com/LuisBarahonaMX) |
| LinkedIn | [linkedin.com/in/luis-barahona-mx](https://www.linkedin.com/in/luis-barahona-mx/) |
| Instagram | [@luisbarahonamx](https://www.instagram.com/luisbarahonamx/) |
| X | [@luisbarahonaMX](https://x.com/luisbarahonaMX) |
| Website | [SIKO AI Platform](https://sikoai.web.app/) |
