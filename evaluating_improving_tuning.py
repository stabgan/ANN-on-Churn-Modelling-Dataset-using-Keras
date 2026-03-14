# Artificial Neural Network — Evaluating, Improving & Tuning
# Cross-validation and GridSearchCV on the Churn Modelling ANN.
#
# Requires: pip install scikeras
#   scikeras replaces the removed keras.wrappers.scikit_learn module.

# ── Imports ──────────────────────────────────────────────────────────
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# scikeras is the modern replacement for keras.wrappers.scikit_learn
from scikeras.wrappers import KerasClassifier


def main():
    # ── Part 1 — Data Preprocessing ─────────────────────────────────

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = pd.read_csv(os.path.join(script_dir, "Churn_Modelling.csv"))
    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values

    # Encoding categorical data
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
    X = X[:, 1:]  # drop first dummy to avoid dummy-variable trap

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # ── Part 2 — Building the ANN ───────────────────────────────────

    classifier = Sequential()
    classifier.add(
        Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11)
    )
    # Dropout regularisation (rate, not the old 'p' kwarg)
    # classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
    # classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

    classifier.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    classifier.fit(X_train, y_train, batch_size=10, epochs=100)

    # ── Part 3 — Predictions & Evaluation ────────────────────────────

    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)

    # Single-customer prediction
    new_prediction = classifier.predict(
        sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
    )
    new_prediction = new_prediction > 0.5
    print(f"Single-customer churn prediction: {bool(new_prediction[0][0])}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    # ── Part 4 — Cross-Validation ────────────────────────────────────

    def build_classifier():
        model = Sequential()
        model.add(
            Dense(
                units=6,
                kernel_initializer="uniform",
                activation="relu",
                input_dim=11,
            )
        )
        model.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
        model.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    # scikeras API: model= instead of build_fn=
    cv_classifier = KerasClassifier(
        model=build_classifier, batch_size=10, epochs=100, verbose=0
    )
    accuracies = cross_val_score(
        estimator=cv_classifier, X=X_train, y=y_train, cv=10, n_jobs=-1
    )
    mean = accuracies.mean()
    variance = accuracies.std()
    print(f"CV accuracy: {mean:.4f} (+/- {variance:.4f})")

    # ── Part 5 — Grid Search Tuning ──────────────────────────────────

    def build_classifier_tuned(optimizer="adam"):
        model = Sequential()
        model.add(
            Dense(
                units=6,
                kernel_initializer="uniform",
                activation="relu",
                input_dim=11,
            )
        )
        model.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
        model.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    gs_classifier = KerasClassifier(model=build_classifier_tuned, verbose=0)
    parameters = {
        "batch_size": [25, 32],
        "epochs": [100, 500],
        "model__optimizer": ["adam", "rmsprop"],
    }
    grid_search = GridSearchCV(
        estimator=gs_classifier, param_grid=parameters, scoring="accuracy", cv=10
    )
    grid_search = grid_search.fit(X_train, y_train)
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    print(f"Best parameters: {best_parameters}")
    print(f"Best accuracy:   {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
