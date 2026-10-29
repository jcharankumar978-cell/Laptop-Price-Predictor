# Laptop Price Predictor — Major Project

## Implemented workflow
- Downloads the CampusX laptop CSV.
- Extracts the four report features: brand, RAM (GB), processor family and total storage (GB).
- One-hot encodes categorical features and imputes missing feature values.
- Trains a Linear Regression pipeline with an 80/20 train/test split (`random_state=42`).
- Prints held-out MAE, RMSE and R² and saves the fitted pipeline with input options and metrics.
- Serves a Flask form using the saved model.

## Run locally
```bash
python -m pip install -r requirements.txt
python train_model.py
python app.py
```
Open http://127.0.0.1:5000 after training.

## Important
The training script downloads the dataset from the report's GitHub URL. This chat's execution environment had no outbound network/DNS access, so the dataset could not be downloaded here and no genuine test-set metrics are claimed in the accompanying report. Run `python train_model.py` in Google Colab or another internet-enabled Python environment to generate the real metrics and `laptop_price_model.joblib`.
