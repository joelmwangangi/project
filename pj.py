import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ================================
# Streamlit UI
# ================================
st.title("📊 Loan Default Prediction (Deep Learning)")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    # ================================
    # Load data
    # ================================
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Dataset:", df.head())

    # Choose target column
    target_col = st.selectbox("Select Target Column (default = Status)", df.columns, index=len(df.columns)-1)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify numeric & categorical features
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    # ================================
    # Preprocessing
    # ================================
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_processed, y)

    # Split into train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X_res, y_res, test_size=0.3, random_state=42, stratify=y_res)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # ================================
    # Build Model
    # ================================
    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    # ================================
    # Training
    # ================================
    st.write("### Training Model...")
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        callbacks=callbacks,
        verbose=0
    )

    st.success("✅ Training complete!")

    # ================================
    # Evaluation
    # ================================
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_prob = model.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    st.write("### Model Evaluation")
    st.json(report)
    st.write("ROC AUC:", roc_auc_score(y_test, y_prob))

    # ================================
    # Save Model + Preprocessor
    # ================================
    model.save("loan_dnn_model.h5")
    with open("preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    st.info("📂 Model saved as `loan_dnn_model.h5` and preprocessor as `preprocessor.pkl`")

    # ================================
    # Prediction Form
    # ================================
    st.write("### Predict New Applicant")

    new_app = {}
    for col in num_cols:
        new_app[col] = st.number_input(f"{col}", value=float(df[col].mean()))

    for col in cat_cols:
        new_app[col] = st.selectbox(f"{col}", df[col].unique())

    if st.button("Predict Default Probability"):
        new_df = pd.DataFrame([new_app])
        new_processed = preprocessor.transform(new_df)
        prob = model.predict(new_processed)[0][0]
        st.write(f"**Default Probability: {prob:.2f}**")
        st.write("Prediction:", "⚠️ Likely to Default" if prob > 0.5 else "✅ Likely to Repay")
