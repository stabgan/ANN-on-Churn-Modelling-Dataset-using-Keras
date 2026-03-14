# ANN Customer Churn Prediction

Predict whether a bank customer will leave (churn) using a Keras artificial neural network trained on 10 000 records.

## What It Does

Builds, trains, and evaluates a binary-classification ANN on the **Churn Modelling** dataset. Two scripts cover the full workflow:

| Script | Purpose |
|---|---|
| `ann_new.py` | Preprocess → train a 2-hidden-layer ANN → predict on test set + single customer |
| `evaluating_improving_tuning.py` | Same base model **plus** 10-fold cross-validation and `GridSearchCV` hyperparameter tuning |

## Model Architecture

```
Input (11 features) → Dense(6, relu) → Dense(6, relu) → Dense(1, sigmoid)
```

- **Optimizer:** Adam
- **Loss:** Binary cross-entropy
- **Epochs:** 100 · **Batch size:** 10

## Dataset

`Churn_Modelling.csv` — 10 000 rows × 14 columns (credit score, geography, gender, age, tenure, balance, etc.).
Target: `Exited` (1 = churned).

## 🛠 Tech Stack

| | Technology | Role |
|---|---|---|
| 🐍 | Python 3.9+ | Language |
| 🧠 | TensorFlow / Keras | Neural network |
| 📊 | scikit-learn | Preprocessing, CV, grid search |
| 🔗 | scikeras | sklearn ↔ Keras bridge |
| 🔢 | NumPy / Pandas | Data handling |
| 📈 | Matplotlib | Plotting |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train and predict
python ann_new.py

# Train, cross-validate, and grid-search
python evaluating_improving_tuning.py
```

## Notebooks

| Notebook | Description |
|---|---|
| `Q1 TSA modelling IBM stock price.ipynb` | Time-series analysis of daily IBM stock prices (1980–1992) |
| `nb/Gemma3_(4B).ipynb` | Gemma 3 (4B) fine-tuning with Unsloth on Colab |
| `nb/Qwen3_(14B)-Reasoning-Conversational.ipynb` | Qwen3 14B reasoning/conversational fine-tuning |
| `site/en/gemma/docs/core/lora_tuning.ipynb` | Gemma LoRA tuning reference notebook |

## ⚠️ Known Issues

- `evaluating_improving_tuning.py` grid search with `n_jobs=-1` may hang on some systems due to TensorFlow multiprocessing. Set `n_jobs=1` if that happens.
- Dropout layers are commented out; uncomment and tune `rate` if overfitting occurs.

## License

[MIT](LICENSE)
