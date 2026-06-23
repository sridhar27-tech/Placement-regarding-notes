# Pandas Library Usage in Machine Learning

**Pandas** is a powerful, open-source Python library built on top of NumPy, providing high-performance data structures and data analysis tools. In machine learning workflows, Pandas is indispensable for **data ingestion**, **cleaning**, **exploration**, **feature engineering**, and **preprocessing** before feeding data into ML models (e.g., scikit-learn, TensorFlow, PyTorch).

## Why Pandas in Machine Learning?

- **DataFrame**: Tabular data structure similar to Excel/SQL tables — ideal for datasets with mixed types.
- **Efficient operations**: Vectorized computations, missing value handling, grouping, merging, and reshaping.
- **Integration**: Seamless with scikit-learn (e.g., `train_test_split`, pipelines), Matplotlib/Seaborn for EDA, and ML frameworks.
- Common ML tasks:
  - Loading CSV/Excel/JSON/SQL data
  - Handling missing values (`fillna`, `dropna`)
  - Encoding categorical variables (`get_dummies`, `LabelEncoder`)
  - Feature scaling/normalization
  - Splitting data, creating interaction features, outlier detection

---

## Python Example 1: Data Loading, Exploration & Preprocessing

This example demonstrates loading a dataset, basic EDA, cleaning, and preparing features for an ML model.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# 2. Basic exploration
print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nSummary statistics:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())

# 3. Data cleaning
# Handle missing values (example)
df = df.fillna(df.mean(numeric_only=True))

# 4. Feature engineering
df['sepal_ratio'] = df['sepal_length'] / df['sepal_width']  # New feature
df = pd.get_dummies(df, columns=['species'], prefix='sp')     # One-hot encoding

# 5. Prepare for ML
X = df.drop(['sepal_ratio'], axis=1)   # Features
y = df['sepal_ratio']                  # Target (example regression)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nPreprocessed training data shape:", X_train_scaled.shape)
--
## Python Example 2: Data Loading, Exploration & Preprocessing

import pandas as pd
import numpy as np

# Sample customer churn dataset
data = {
    'customer_id': range(1, 1001),
    'age': np.random.randint(18, 70, 1000),
    'tenure_months': np.random.randint(1, 120, 1000),
    'monthly_charges': np.random.uniform(20, 150, 1000),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000),
    'churn': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
}
df = pd.DataFrame(data)

# Feature engineering with groupby
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])
avg_charges = df.groupby('age_group')['monthly_charges'].transform('mean')
df['charges_vs_avg'] = df['monthly_charges'] - avg_charges

# Encoding & interactions
df = pd.get_dummies(df, columns=['contract_type'], drop_first=True)
df['high_value_tenure'] = (df['monthly_charges'] > 80) & (df['tenure_months'] > 24)

# Final feature matrix
features = ['age', 'tenure_months', 'monthly_charges', 'charges_vs_avg', 
            'contract_type_One year', 'contract_type_Two year', 'high_value_tenure']
X = df[features]
y = df['churn']

print("Feature correlation with target:\n", X.corrwith(y).sort_values(ascending=False))
print("\nSample engineered features:\n", df.head()[['age_group', 'charges_vs_avg', 'high_value_tenure']])
