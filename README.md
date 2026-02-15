# 🔍 Smart Feature Engineering Tool (ML + Deep Learning)

## 📌 Overview

This project is a complete tabular data processing pipeline that:

- Preprocesses raw CSV data  
- Computes feature importance using Machine Learning  
- Generates learned embeddings using a Neural Network (PyTorch)  
- Exports a transformed dataset ready for advanced modeling  

It supports both:

- Classification  
- Regression  

The final output is saved as `transformed_dataset.csv`.

---

## 🚀 Key Features

✔ Automatic preprocessing  
✔ Label encoding for categorical features  
✔ Feature scaling using StandardScaler  
✔ Random Forest feature importance  
✔ Mutual Information scoring (classification only)  
✔ Neural Network-based feature embeddings  
✔ GPU support (if available)  

---

## 🛠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- PyTorch  

---

## 📂 Project Workflow

### 1️⃣ Data Preprocessing

- Removes `ID` column (if present)  
- Encodes categorical variables using LabelEncoder  
- Standardizes numerical features  
- Encodes target variable (for classification)  

Function used:

```python
preprocess_data()
```

---

### 2️⃣ Feature Importance (Machine Learning)

Uses:

- RandomForestClassifier (for classification)  
- RandomForestRegressor (for regression)  
- Mutual Information (classification only)  

Final ranking is based on:

```
Average Score = (RF Importance + Mutual Info) / 2
```

Function used:

```python
ml_feature_importance()
```

---

### 3️⃣ Deep Learning Embeddings

A custom PyTorch Neural Network generates learned feature embeddings.

### Neural Network Architecture

Input → 64 neurons → 31-dimensional embedding → Output layer  

- Activation: ReLU  
- Optimizer: Adam  
- Loss:
  - CrossEntropyLoss (classification)
  - MSELoss (regression)

After training, the 31-dimensional embeddings are extracted and added to the dataset.

Class used:

```python
TabularNN
```

Training function:

```python
train_nn()
```

---

## 📦 Installation

Install required libraries:

```bash
pip install pandas numpy scikit-learn torch
```

---

## ▶️ How to Run

```bash
python your_script_name.py
```

Then enter:

- CSV file path  
- Target column name  
- Task type (classification or regression)  

---

## 📊 Output

The script generates:

- Feature importance ranking (top 10 printed)  
- Neural Network training logs  
- Final transformed dataset saved as:

```
transformed_dataset.csv
```

This dataset includes:

- Scaled original features  
- 31 learned embedding features (Embed_1 → Embed_31)  

---

## 🎯 Use Cases

- Advanced feature engineering  
- Dimensionality transformation  
- Representation learning for tabular data  
- Preprocessing before XGBoost / LightGBM / Deep Learning models  
- Kaggle-style ML workflows  

---

## 👨‍💻 Author

Built as part of an advanced ML + Deep Learning experimentation workflow.

If you found this useful, consider starring the repository ⭐
