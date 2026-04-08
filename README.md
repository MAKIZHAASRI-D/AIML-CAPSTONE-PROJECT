# Titanic Survival Prediction 🚢

This project implements a Machine Learning model to predict the survival of passengers on the Titanic using the classic Kaggle Titanic dataset. It utilizes a Random Forest Classifier and includes full data preprocessing and feature engineering.

---

## 📊 Dataset Overview

The dataset contains information about passengers such as Age, Sex, Class, Fare, etc.

- **Source:** Titanic Dataset (CSV)
- **Target Variable:** Survived  
  - 0 = No  
  - 1 = Yes  

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:**
  - pandas & numpy (Data Manipulation)
  - seaborn & matplotlib (Visualization)
  - scikit-learn (Machine Learning)

---

## 🚀 Workflow

### 1. Data Cleaning
- **Missing Values:**
  - Filled missing *Age* values with the median
  - Filled missing *Embarked* values with the mode
- **Feature Dropping:**
  - Removed irrelevant columns: `Cabin`, `Name`, `Ticket`, `PassengerId`

### 2. Feature Engineering
- **Label Encoding:**
  - Converted `Sex` into numerical format  
    - Male = 0  
    - Female = 1  
- **One-Hot Encoding:**
  - Converted `Embarked` into dummy variables

### 3. Modeling
- **Algorithm:** Random Forest Classifier  
- **Parameters:**
  - `n_estimators = 100`
  - `random_state` set for reproducibility  
- **Train/Test Split:**
  - 80% Training
  - 20% Testing

---

## 📈 Results

The model performance is evaluated using:

- **Accuracy Score:** Overall percentage of correct predictions  
- **Confusion Matrix:**  
  - True Positives  
  - True Negatives  
  - False Positives  
  - False Negatives  
- **Classification Report:**
  - Precision  
  - Recall  
  - F1-Score  

---

## ⚙️ How to Run

1. Ensure `titanic.csv` is in the root directory

2. Install dependencies:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn
3.run
```bash
python survival_chance.py
