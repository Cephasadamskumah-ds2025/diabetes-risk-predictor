# 🏥 Diabetes Risk Prediction: Early Detection System

> An end-to-end clinical risk prediction web application using XGBoost and SHAP explainability, trained on 229,474 CDC survey respondents and deployed as an interactive Streamlit app.

**🔗 Live App:** [Click here to try the app](#)  
<!-- Replace # with your Streamlit URL after deployment -->

---

## 📌 Project Overview

Diabetes remains one of the most important chronic disease challenges, and early identification of high-risk individuals can support timely screening, lifestyle intervention, and preventive care.

This project builds a complete machine learning pipeline that predicts diabetes risk from **21 behavioural, clinical, lifestyle, and socioeconomic indicators** from the CDC BRFSS 2015 Diabetes Health Indicators dataset.

The final solution includes:

- Exploratory data analysis to identify major diabetes risk factors
- Feature engineering and stratified train-test preparation
- Logistic Regression baseline modelling
- XGBoost final model training
- SHAP explainability for model interpretation
- An interactive Streamlit web application for real-time risk prediction

The app returns a diabetes risk probability, classifies the user as low, moderate, or high risk, and explains which features contributed most to the prediction.

> **Note:** This application is for educational and portfolio demonstration purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical advice.

---

## ❓ Business Question

**Can we accurately predict diabetes risk from behavioural, clinical, and socioeconomic survey data, and explain the prediction in clinically meaningful terms?**

---

## 🖥️ Live Application

The Streamlit application allows users to:

- Enter 21 patient indicators across 5 categories
- Receive an instant diabetes risk score classified as Low, Moderate, or High
- View a SHAP explanation showing which factors increased or decreased the prediction
- Review the full patient input summary

**App features:**

- 🟢 Low Risk, 🟡 Moderate Risk, and 🔴 High Risk colour-coded output
- Dynamic SHAP explanation that updates when input values change
- Clinically labelled input fields with human-readable options
- Medical disclaimer for responsible and educational use

---

## 📁 Repository Structure

```
diabetes-risk-predictor/
│
├── app/
│   └── app.py                      ← Streamlit web application
│
├── model/
│   ├── xgb_model.pkl               ← Trained XGBoost model
│   ├── explainer.pkl               ← SHAP TreeExplainer
│   └── feature_names.pkl           ← Feature column names
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   ← EDA, risk factors, correlations
│   ├── 02_feature_engineering.ipynb ← Split, scale, export
│   └── 03_modelling.ipynb          ← LR baseline, XGBoost, SHAP
│
├── data/
│   └── feature_importance.csv      ← XGBoost feature importance scores
│
├── visuals/
│   ├── target_distribution.png
│   ├── bmi_age_analysis.png
│   ├── lifestyle_factors.png
│   ├── healthcare_access.png
│   ├── correlation_heatmap.png
│   ├── roc_curve_comparison.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   └── shap_waterfall.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🗂️ Dataset

**Source:** [CDC Diabetes Health Indicators - Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

| Attribute | Detail |
|---|---|
| Raw records | 253,680 |
| After deduplication | **229,474** |
| Features | 21 |
| Target | `Diabetes_binary` (0 = No Diabetes, 1 = Diabetes) |
| Class balance | 84.7% No Diabetes · 15.3% Diabetes |
| Missing values | None |
| Data types | All float64 — no encoding required |

**Feature categories:**

| Category | Features |
|---|---|
| Clinical | HighBP, HighChol, CholCheck, BMI, Stroke, HeartDiseaseorAttack |
| Lifestyle | PhysActivity, Fruits, Veggies, HvyAlcoholConsump, Smoker |
| Healthcare Access | AnyHealthcare, NoDocbcCost |
| General Health | GenHlth, MentHlth, PhysHlth, DiffWalk |
| Demographics | Sex, Age, Education, Income |

> ⚠️ **Data not included** due to Kaggle terms. Download `diabetes_binary_health_indicators_BRFSS2015.csv` from the link above and place it in the project root before running notebooks.

---

## 🔍 Exploratory Analysis: Key Findings

| Finding | Detail |
|---|---|
| **BMI gap** | Respondents with diabetes had an average BMI of **32.0**, compared with **28.1** for respondents without diabetes, a difference of **+3.9 BMI points** |
| **Age risk** | Diabetes rates were relatively low among younger respondents aged **18–44**, then increased noticeably from middle age onward, peaking around age groups **9–11**, representing **60–74 years** |
| **Physical inactivity** | Respondents with no physical activity had a higher diabetes rate than active respondents, **21.3% vs 13.1%** |
| **Income gradient** | The lowest income group had approximately **2.5x** the diabetes rate of the highest income group, **24.3% vs 9.8%** |
| **Education gradient** | Diabetes rates were higher among respondents with lower education levels, with the highest rate at education level 2, **29.3%**, compared with **11.6%** for college graduates |
| **Strongest correlator** | `GenHlth` had the strongest Pearson correlation with diabetes, **r = 0.277**, followed by `HighBP`, **r = 0.254** |

These findings suggest that diabetes risk in the dataset is shaped by a combination of clinical, lifestyle, demographic, and socioeconomic factors.

---

## 🤖 Modelling

### Train / Test Split

A **stratified 80/20 split** preserves the original 84.7/15.3 class ratio in both sets.

| Set | Rows | No Diabetes | Diabetes |
|---|---|---|---|
| Training | 183,579 | 155,501 (84.7%) | 28,078 (15.3%) |
| Testing | 45,895 | 38,876 (84.7%) | 7,019 (15.3%) |

**Stratification** ensures that both the training and testing sets contain representative proportions of diabetic and non-diabetic respondents, allowing for fairer model training and evaluation.

### Models Trained

**1. Logistic Regression** (baseline)
```python
LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
```

**2. XGBoost** (final model)
```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=5.54,
    random_state=42
)
```

### Class Imbalance Handling

- **Logistic Regression:** `class_weight='balanced'`
- **XGBoost:** `scale_pos_weight=5.54` (ratio of negative to positive cases)

---

## 📈 Results

| Metric | Logistic Regression | XGBoost | Winner |
|---|---|---|---|
| ROC-AUC | 0.8106 | **0.8179** | ✅ XGBoost |
| Recall (Diabetes) | 0.76 | **0.78** | ✅ XGBoost |
| Precision (Diabetes) | 0.32 | 0.32 | Draw |
| F1 (Diabetes) | 0.45 | 0.45 | Draw |

**XGBoost selected as the final model.**

> **Why Recall matters most:** A diabetes recall of **0.78** means the model correctly identifies **78% of respondents with diabetes** in the test set. For an early screening tool, this helps reduce false negatives, which are the most costly error because missed high-risk individuals may not receive timely follow-up.

---

## 🔍 SHAP Explainability

| Rank | Feature | SHAP Impact | Interpretation |
|---|---:|---:|---|
| 1 | `GenHlth` | 0.55 | Poorer self-reported general health has the strongest average impact on model predictions |
| 2 | `HighBP` | 0.48 | High blood pressure is one of the most important clinical drivers of predicted diabetes risk |
| 3 | `BMI` | 0.27 | Higher BMI contributes meaningfully to increased predicted diabetes risk |
| 4 | `Age` | 0.25 | Older age groups have a stronger influence on higher predicted diabetes risk |
| 5 | `HighChol` | 0.19 | High cholesterol is an important metabolic risk indicator in the model |

SHAP confirms that the model is driven mainly by clinically meaningful features, especially general health, blood pressure, BMI, age, and cholesterol status.

Unlike standard feature importance, SHAP explains how much each feature contributes to model predictions. This makes the model more transparent and helps users understand which factors are increasing or decreasing an individual risk score.

---

## 💼 Business Impact

| Finding | Clinical Implication |
|---|---|
| 78% Recall | Model correctly flags 78% of diabetic patients for follow-up |
| GenHlth as top predictor | Simple self-reported health questions are highly predictive |
| HighBP dominates clinical features | Blood pressure is one of the most important clinical indicators in this model |
| Income and Education gradients | Diabetes interventions should prioritise lower socioeconomic groups |
| SHAP explainability | Predictions are more transparent, making it suitable for clinical decision support |

These findings show how machine learning can support early risk identification while remaining interpretable for healthcare-focused decision making.



---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|---|---|
| Python | Data processing, modelling, app development |
| pandas / numpy | Data manipulation |
| scikit-learn | Logistic Regression, train/test split, metrics |
| XGBoost | Gradient boosting classifier |
| SHAP | Model explainability |
| matplotlib / seaborn | EDA visualisations |
| Streamlit | Web application framework |
| Streamlit Cloud | Free deployment and hosting |

---

## ▶️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Cephasadamskumah-ds2025/diabetes-risk-predictor.git
cd diabetes-risk-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `diabetes_binary_health_indicators_BRFSS2015.csv` from
[Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
and place it in the project root folder.

### 4. Run notebooks in order
```
01_data_exploration.ipynb     → EDA and risk factor analysis
02_feature_engineering.ipynb  → Split, scale, export train/test sets
03_modelling.ipynb            → Train models, SHAP analysis, export model
```

### 5. Launch the app
```bash
streamlit run app/app.py
```

---

## 🚀 Deploy to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository, branch `main`, file `app/app.py`
5. Click **Deploy**

---

## 🔑 Key Learnings

- **Stratified splits are essential** for imbalanced data — random splits risk placing all diabetic cases in one set
- **scale_pos_weight in XGBoost** is cleaner than SMOTE for handling class imbalance in tree models
- **SHAP TreeExplainer** is significantly faster than KernelExplainer for tree-based models
- **Recall over Accuracy** — accuracy is misleading with 85/15 class imbalance
- **GenHlth outranks BMI** — self-reported general health is more predictive than body weight alone
- **Socioeconomic features matter** — Income and Education carry real predictive signal

---

## 👤 Author

**Cephas Adams Kumah** 
| Data Science Graduate | Healthcare & Open Industries
[LinkedIn](https://linkedin.com/cephas-adams-kumah) · [GitHub](https://github.com/Cephasadamskumah-ds2025) · [Live App](#)

---

## 📄 Licence

MIT Licence

> ⚠️ **Medical Disclaimer:** This tool is for educational and portfolio demonstration purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
