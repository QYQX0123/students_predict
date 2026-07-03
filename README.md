# Student Performance Prediction System

Desktop application for estimating whether a student is likely to pass the final assessment using a Random Forest classifier.

## Features

- Loads `dataset.csv`
- Trains a Random Forest model without external machine-learning packages
- Predicts `Pass` or `Fail` based on the final-grade pass threshold
- Shows the estimated pass probability and fail risk
- Displays key factors for the current prediction using lightweight feature perturbation
- Provides a unified model evaluation page in the GUI
- Reports 5-fold cross-validation accuracy and standard deviation
- Displays per-class precision, recall, F1-score, and a confusion matrix
- Displays feature importance alongside the evaluation results
- Saves prediction history to SQLite
- Supports single or batch deletion of saved history records
- Imports validated CSV/XLSX batch prediction records directly into History
- Exports saved history to CSV or styled XLSX with predicted G3 scores and Pass/Fail row colors

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

The system predicts pass/fail status, pass probability, and an estimated final G3 score from available stage-based student data:

- `Fail`: final grade below 50
- `Pass`: final grade 50 or above

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
