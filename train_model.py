import pandas as pd
import numpy as np
import joblib
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score

# ==============================
# 1. LOAD DATA 
# ==============================

zip_path = "loan.csv.zip"   

with zipfile.ZipFile(zip_path) as z:
    print("Files inside ZIP:", z.namelist())
    
    # Select correct file
    file_name = [f for f in z.namelist() if "accepted" in f and f.endswith(".csv")][0]
    
    with z.open(file_name) as f:
        df = pd.read_csv(f, low_memory=False)

print("Dataset Loaded Successfully!")

# ==============================
# 2. DATA CLEANING
# ==============================

# Keep only required loan statuses
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]

# Convert target variable
df['loan_status'] = df['loan_status'].map({
    'Fully Paid': 0,
    'Charged Off': 1
})

# Drop columns with too many missing values
df = df.dropna(axis=1, thresh=len(df)*0.6)

# Fill remaining missing values
df = df.ffill()

# ==============================
# 3. FEATURE SELECTION (IMPORTANT ⚡)
# ==============================

# Select useful columns (avoids huge dataset issues)
selected_cols = [
    'loan_amnt', 'term', 'int_rate', 'installment',
    'annual_inc', 'dti', 'delinq_2yrs',
    'revol_util', 'total_acc', 'loan_status'
]

df = df[selected_cols]

# ==============================
# 4. ENCODING
# ==============================

# Convert term (e.g., "36 months")
df['term'] = df['term'].astype(str).str.extract(r'(\d+)').astype(int)

# Convert interest rate (remove %)
if df['int_rate'].dtype == 'object':
    df['int_rate'] = df['int_rate'].str.replace('%', '').astype(float)
else:
    df['int_rate'] = df['int_rate'].astype(float)

# Encode categorical if any
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# ==============================
# 5. SPLIT DATA
# ==============================

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 6. TRAIN MODEL
# ==============================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# 7. EVALUATION
# ==============================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# ==============================
# 8. SAVE MODEL
# ==============================

joblib.dump(model, "model.pkl")

print("\n✅ Model saved as model.pkl")
# Save test data for dashboard
joblib.dump((X_test, y_test), "test_data.pkl")