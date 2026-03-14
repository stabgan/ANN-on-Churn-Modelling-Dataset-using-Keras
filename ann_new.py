# Artificial Neural Network — Customer Churn Prediction
# Trains a 2-hidden-layer ANN on the Churn Modelling dataset.

# ── Imports ──────────────────────────────────────────────────────────
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense


def main():
    # ── Part 1 — Data Preprocessing ─────────────────────────────────

    # Importing the dataset (resolve path relative to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = pd.read_csv(os.path.join(script_dir, "Churn_Modelling.csv"))
    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values

    # Encoding categorical data
    # Column 1 = Geography (multi-class) → LabelEncode then OneHot
    # Column 2 = Gender (binary)         → LabelEncode only
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    labelencoder_X_2 = LabelEncoder()
    X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

    # Modern ColumnTransformer replaces the removed
    # OneHotEncoder(categorical_features=...) API
    ct = ColumnTransformer(
        transformers=[("geo", OneHotEncoder(), [1])],
        remainder="passthrough",
    )
    X = ct.fit_transform(X).astype(float)
    # Drop first dummy column to avoid the dummy-variable trap
    X = X[:, 1:]

    # Splitting the dataset into Training and Test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # ── Part 2 — Building the ANN ───────────────────────────────────

    classifier = Sequential()

    # Input layer + first hidden layer
    classifier.add(
        Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11)
    )
    # Second hidden layer
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
    # Output layer
    classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

    classifier.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )

    # Training
    classifier.fit(X_train, y_train, batch_size=10, epochs=100)

    # ── Part 3 — Predictions & Evaluation ────────────────────────────

    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)

    # Single-customer prediction
    # Geography: France, CreditScore: 600, Gender: Male, Age: 40,
    # Tenure: 3, Balance: 60000, NumOfProducts: 2, HasCrCard: Yes,
    # IsActiveMember: Yes, EstimatedSalary: 50000
    new_prediction = classifier.predict(
        sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
    )
    new_prediction = new_prediction > 0.5
    print(f"Single-customer churn prediction: {bool(new_prediction[0][0])}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")


if __name__ == "__main__":
    main()
