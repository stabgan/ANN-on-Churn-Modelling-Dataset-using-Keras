# 🧠 ANN Customer Churn Prediction

Predict whether a bank customer will leave (churn) using an Artificial Neural Network built with TensorFlow/Keras, trained on 10,000 customer records.

## 📖 Description

This project builds, evaluates, and tunes a binary-classification ANN on the **Churn Modelling** dataset. It demonstrates a complete deep learning pipeline — from data preprocessing and encoding through model training, prediction, cross-validation, and hyperparameter tuning via grid search.

| File | Purpose |
|---|---|
| `ann_new.py` | Preprocesses data, trains a 2-hidden-layer ANN, predicts on the test set and a single new customer |
| `evaluating_improving_tuning.py` | Same base model **plus** 10-fold cross-validation and `GridSearchCV` hyperparameter tuning |

## 🏗️ Architecture

```
Input (11 features)
       │
       ▼
┌──────────────┐
│ Dense(6, ReLU)│  ← Hidden Layer 1
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Dense(6, ReLU)│  ← Hidden Layer 2
└──────┬───────┘
       │
       ▼
┌─────────────────┐
│Dense(1, Sigmoid) │  ← Output Layer
└─────────────────┘
```

- **Optimizer:** Adam
- **Loss:** Binary Cross-Entropy
- **Epochs:** 100 (default)
- **Batch Size:** 10 (default)

## 🛠️ Tech Stack

| Tool | Version |
|---|---|
| 🐍 Python | 3.8+ |
| 🔢 NumPy | latest |
| 🐼 Pandas | latest |
| 📊 Matplotlib | latest |
| 🤖 TensorFlow / Keras | 2.x |
| 🧪 scikit-learn | 1.0+ |
| 🔗 scikeras | latest |

## 📦 Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow scikeras
```

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/stabgan/ANN-on-Churn-Modelling-Dataset-using-Keras.git
   cd ANN-on-Churn-Modelling-Dataset-using-Keras
   ```

2. Install dependencies:

   ```bash
   pip install numpy pandas matplotlib scikit-learn tensorflow scikeras
   ```

3. Train the model and predict:

   ```bash
   python ann_new.py
   ```

4. Train, cross-validate, and grid-search:

   ```bash
   python evaluating_improving_tuning.py
   ```

## ⚠️ Known Issues

- `OneHotEncoder` with `ColumnTransformer` assumes column index 1 is the Geography column after label encoding. If the dataset schema changes, the transformer index must be updated.
- Grid search in `evaluating_improving_tuning.py` with `n_jobs = -1` can be memory-intensive. Reduce `cv` folds or run on a machine with sufficient RAM.
- The dataset (`Churn_Modelling.csv`) must be in the same directory as the scripts when running.

## 📄 License

[MIT](LICENSE)
