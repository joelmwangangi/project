# loan_default_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pickle

# ========= 1. Load Dataset =========
df = pd.read_csv("Loan_Default.csv")

# Assume target column is "Status" (1 = default, 0 = non-default)
target = "Status"
X = df.drop(columns=[target])
y = df[target]

# Identify numeric and categorical columns
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# ========= 2. Preprocessing =========
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# ========= 3. Model =========
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# ========= 4. Train/Test Split =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ========= 5. Train =========
pipeline.fit(X_train, y_train)

# ========= 6. Evaluation =========
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# ========= 7. Save Model =========
with open("loan_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model training complete. Saved as loan_model.pkl")
