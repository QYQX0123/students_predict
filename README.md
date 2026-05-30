# Student Performance Prediction System

Desktop application for predicting student final performance categories using a Random Forest classifier.

## Features

- Loads `dataset.csv`
- Trains a Random Forest model without external machine-learning packages
- Predicts `Low`, `Medium`, or `High` final performance
- Shows prediction confidence and class probabilities
- Displays feature importance in the GUI
- Saves prediction history to SQLite
- Exports history and batch prediction results to CSV

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

- `Low`: final grade below 10
- `Medium`: final grade from 10 to 14
- `High`: final grade 15 or above

The strongest expected predictors are usually `G1` and `G2`, because they represent previous and midterm academic performance.
