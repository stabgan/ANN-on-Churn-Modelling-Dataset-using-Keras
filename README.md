# ANN Customer Churn Prediction

Predict whether a bank customer will leave (churn) using a Keras artificial neural network trained on 10 000 records.

## Overview

Two scripts build, evaluate, and tune a binary-classification ANN on the [Churn Modelling dataset](Churn_Modelling.csv):

| File | What it does |
|---|---|
| `ann_new.py` | Preprocesses data, trains a 2-hidden-layer ANN (6-6-1), predicts on the test set and a single new customer |
| `evaluating_improving_tuning.py` | Same base model **plus** 10-fold cross-validation and `GridSearchCV` hyperparameter tuning (batch size, epochs, optimizer) |

### Model architecture

```
Input (11 features) → Dense(6, relu) → Dense(6, relu) → Dense(1, sigmoid)
```

Optimizer: Adam · Loss: binary cross-entropy · Epochs: 100 · Batch size: 10

### Dataset

`Churn_Modelling.csv` — 10 000 rows, 14 columns (credit score, geography, gender, age, tenure, balance, etc.). Target column: `Exited` (1 = churned).

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- TensorFlow / Keras

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## Usage

```bash
# Train and predict
python ann_new.py

# Train, cross-validate, and grid-search
python evaluating_improving_tuning.py
```

## Known Bugs & Deprecations

| Issue | Location | Fix |
|---|---|---|
| `OneHotEncoder(categorical_features=[1])` removed in scikit-learn ≥ 0.22 | both scripts | Use `ColumnTransformer` + `OneHotEncoder` instead |
| `from keras.wrappers.scikit_learn import KerasClassifier` removed in Keras 3 / TF 2.16+ | `evaluating_improving_tuning.py` | Use `scikeras.wrappers.KerasClassifier` (`pip install scikeras`) |
| `Dropout(p=0.1)` — parameter is `rate`, not `p` | `evaluating_improving_tuning.py` (commented out) | Change to `Dropout(rate=0.1)` |
| Top-level `import keras` may fail without standalone Keras installed | both scripts | Use `from tensorflow import keras` or install `keras` separately |

## License

[MIT](LICENSE)
