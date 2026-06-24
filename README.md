# Student Performance Prediction System

Desktop application for predicting student final performance categories using a Random Forest classifier.

## Features

- Loads `dataset.csv`
- Trains a Random Forest model without external machine-learning packages
- Predicts `Low`, `Medium`, or `High` final performance
- Shows prediction confidence and class probabilities
- Provides a unified model evaluation page in the GUI
- Reports 5-fold cross-validation accuracy and standard deviation
- Displays per-class precision, recall, F1-score, and a confusion matrix
- Displays feature importance alongside the evaluation results
- Shows per-student feature influence from History > View Detail
- Saves prediction history to SQLite
- Supports single or batch deletion of saved history records
- Imports validated CSV/XLSX batch prediction records directly into History
- Exports saved history to CSV

## Run

```powershell
& "C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" run_app.py
```

Or with any Python installation that includes Tkinter:

```powershell
python run_app.py
```

## Test

```powershell
& "C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" self_test.py
```

## Prediction Meaning

The system predicts the final performance category from available stage-based student data:

- `Low`: final grade below 50
- `Medium`: final grade from 50 to 74
- `High`: final grade 75 or above

The GUI accepts `G1` and `G2` as 0-100 scores. Internally, the system converts them to the dataset's original 0-20 scale before prediction.

Batch import accepts `.csv` and `.xlsx` files. Each row must include:

- `name`
- `matric_no`
- `sex`
- `age`
- `study_time`
- `failures`
- `activities`
- `absences`
- `G1` as a 0-100 score
- `G2` as a 0-100 score

Friendly column labels such as `Student Name`, `Matric No.`, `Gender`, `Study Time`, `Previous Grade G1 (0-100)`, and `Midterm Grade G2 (0-100)` are also accepted.

The strongest expected predictors are usually `G1` and `G2`, because they represent previous and midterm academic performance.
